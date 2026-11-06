"""Bidirectional state block in the full-attention layers. On for checkpoints whose vev.json says "bidir": "state"
(configure()), or with VEV_BIDIR=state.

The state block of a row is its first state_len tokens: system prompt, "State:\n" and the state, i.e. the prefix that
the questions of one request share. With the block on, in the 8 full-attention layers of Qwen3.5 the tokens of the
state block attend to each other in both directions; every later token (question, options, answer hint, readout
position) stays causal; padding is never attended. The Gated DeltaNet layers are recurrent and unchanged. The prefix's
keys and values do not depend on any question, so a prefilled state cache stays shareable.

Mechanism: Qwen3_5TextModel builds its masks itself ({"full_attention": create_causal_mask(...), "linear_attention":
create_recurrent_attention_mask(...)}). While a state_blocks(...) context is active, the create_causal_mask it calls
gets one extra argument, block_sequence_ids (0 inside the state block, -1 after it), which transformers turns into
(causal OR same block) AND padding, in the format of the attention implementation (bool for sdpa, additive float for
eager). Everything else, including the linear-attention entry, is what the model computes on its own. A forward whose
queries all lie past the state block (question tails continuing from a prefilled cache) gets the unchanged causal mask.

Off: nothing is patched and nothing changes. Training and serving use this one implementation."""

from __future__ import annotations

import contextvars
import inspect
import os
from contextlib import contextmanager, nullcontext

import torch

MODE = os.environ.get("VEV_BIDIR", "")
if MODE not in ("", "state"):
    raise ValueError(f"VEV_BIDIR={MODE!r}: expected 'state' or unset")
ENABLED = MODE == "state"
SUPPORTED_ATTN = ("sdpa", "eager")  # flash attention would silently drop the block mask

_active: contextvars.ContextVar = contextvars.ContextVar("vev_bidir_active", default=None)
_causal: contextvars.ContextVar = contextvars.ContextVar("vev_bidir_causal", default=False)


def block_ids(state_len: torch.Tensor, total: int, device: torch.device) -> torch.Tensor:
    """[B, total] block id per absolute position: 0 inside the row's state block, -1 (causal) after it."""
    pos = torch.arange(total, device=device)
    return torch.where(pos[None, :] < state_len.to(device)[:, None], 0, -1)


def _wrap(orig):
    sig = inspect.signature(orig)

    def create_causal_mask(*args, **kwargs):
        st = _active.get()
        if st is None:
            return orig(*args, **kwargs)
        a = sig.bind(*args, **kwargs).arguments
        config, emb = a["config"], a["inputs_embeds"]
        if config._attn_implementation not in SUPPORTED_ATTN:
            raise RuntimeError(f"VEV_BIDIR=state supports attn_implementation {SUPPORTED_ATTN}, "
                               f"not {config._attn_implementation!r}")
        if a.get("block_sequence_ids") is not None:
            raise RuntimeError("VEV_BIDIR=state: the model already passes block_sequence_ids")
        lens = st["lens"]
        bsz, q_len = emb.shape[:2]
        if lens.shape[0] != bsz:
            raise RuntimeError(f"VEV_BIDIR=state: {lens.shape[0]} state lengths for a batch of {bsz}")
        pkv, mask = a.get("past_key_values"), a.get("attention_mask")
        past = pkv.get_seq_length() if pkv is not None else 0
        if isinstance(mask, torch.Tensor) and mask.ndim == 2 and mask.shape[-1] != past + q_len:
            raise RuntimeError(f"VEV_BIDIR=state: 2D mask of {mask.shape[-1]} for {past} cached + {q_len} new tokens")
        st["calls"] += 1
        if not bool((lens > past).any()):  # every query lies past the state block: plain causal mask
            return orig(*args, **kwargs)
        a["block_sequence_ids"] = block_ids(lens, past + q_len, emb.device)
        return orig(**a)

    create_causal_mask._vev_bidir = True
    return create_causal_mask


def _install() -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m

    if not getattr(m.create_causal_mask, "_vev_bidir", False):
        m.create_causal_mask = _wrap(m.create_causal_mask)


if ENABLED:
    _install()


def configure(mode: str) -> None:
    """Switch the state block on ("state") or off ("") at run time, e.g. from a checkpoint's vev.json."""
    global MODE, ENABLED
    if mode not in ("", "state"):
        raise ValueError(f"bidir={mode!r}: expected 'state' or ''")
    MODE, ENABLED = mode, mode == "state"
    if ENABLED:
        _install()


@contextmanager
def state_blocks(state_len):
    """Forwards inside see state blocks of state_len[i] tokens in row i (absolute positions, cached tokens included).
    No-op when the flag is off or inside causal()."""
    if not ENABLED or _causal.get():
        yield
        return
    st = {"lens": torch.as_tensor(state_len, dtype=torch.long).reshape(-1).cpu(), "calls": 0}
    token = _active.set(st)
    try:
        yield
    finally:
        _active.reset(token)
    if st["calls"] == 0:
        raise RuntimeError("VEV_BIDIR=state: no full-attention mask was built inside state_blocks(); "
                           "the transformers version does not go through the patched create_causal_mask")


def for_rows(rows):
    """state_blocks for encoded rows carrying "state_len"; a null context when the flag is off."""
    return state_blocks([r["state_len"] for r in rows]) if ENABLED else nullcontext()


@contextmanager
def causal():
    """Inside, state_blocks() is a no-op: plain causal forwards (the frozen base model as KL teacher in training)."""
    token = _causal.set(True)
    try:
        yield
    finally:
        _causal.reset(token)


def forced_causal() -> bool:
    return bool(_causal.get())
