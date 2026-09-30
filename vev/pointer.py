"""Pointer readout: one causal row per question = chat-template prefix (system + state) followed by a
branch (question, options as <opt>…</opt> spans, <decide>); the head scores the <decide> hidden state against every
</opt> hidden state. Rendering and the head are shared by training (train/) and serving (PointerEngine)."""

from __future__ import annotations

import copy
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from vev.readout import DTYPES, MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, Result, choice_labels, confidence
from vev.readout import SYSTEM_PROMPT as ZEROSHOT_SYSTEM_PROMPT
from vev.readout import render_desc as _render_desc
from vev.state import ImageSlot, InvalidRequest, render_desc, serialize_state
from vev.tokens import count_output_tokens, count_state_tokens

# Rarely used Qwen special tokens reused as delimiters (kev does the same): no new embedding rows, LoRA learns their role.
SPECIAL = {
    "q": "<|fim_middle|>",
    "opts": "<|fim_pad|>",
    "opt": "<|box_start|>",
    "opt_end": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}
TYPE_TAG = {"choice": "[choice]", "noul": "[noul]", "score": "[score]"}
SYSTEM_PROMPT = "You are a careful judge. Read the state, then decide the answer to each question about it."

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def sanitize(text: str) -> str:
    """Caller text can never produce a special token (`<|name|>` becomes `<¦name¦>`), so option boundaries are unforgeable."""
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


def option_spans(q: dict[str, Any]) -> list[tuple[str, str]]:
    """(label, description) per option in rendered order; noul is the two-option question yes/no."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        return [(str(label), render_desc(desc)) for label, desc in crit.items()]
    if q["type"] == "score":
        return [(str(i), render_desc(desc)) for i, desc in enumerate(crit)]
    crit = crit or {}
    return [("yes", render_desc(crit.get("true"))), ("no", render_desc(crit.get("false")))]


PRIOR_HINT = {
    "choice": "\nAnswer with the letter of the single best option.",
    "noul": "\nAnswer Yes or No.",
    "score": "\nAnswer with the level number only.",
}


def prior_labels(q: dict[str, Any], letters: list[str]) -> list[str]:
    """Single-token answer strings per option, in rendered order (the zero-shot readout's scheme)."""
    k = len(option_spans(q))
    if q["type"] == "choice":
        return letters[:k]
    if q["type"] == "score":
        return [str(i) for i in range(k)]
    return ["Yes", "No"]


def branch_text(q: dict[str, Any], letters: list[str] | None = None) -> str:
    """letters: when given (prior mode), every option is prefixed with its single-token answer label and the
    branch ends with the zero-shot answer hint, so the LM's next-token logits at <decide> are a usable prior."""
    instr = sanitize((q.get("instructions") or "").strip())
    parts = ["\n", SPECIAL["q"], TYPE_TAG[q["type"]], " ", instr, SPECIAL["opts"]]
    spans = option_spans(q)
    labs = prior_labels(q, letters) if letters is not None else None
    for j, (label, desc) in enumerate(spans):
        if labs is None:
            body = sanitize(label) + (": " + sanitize(desc) if desc else "")
        elif q["type"] == "noul" or q["type"] == "score":
            body = labs[j] + (": " + sanitize(desc) if desc else "")
        else:
            body = f"{labs[j]}. " + sanitize(label) + (": " + sanitize(desc) if desc else "")
        parts += [SPECIAL["opt"], body, SPECIAL["opt_end"]]
    if labs is not None:
        parts.append(PRIOR_HINT[q["type"]])
    return "".join(parts)


def row_text(processor, segments: list, q: dict[str, Any], letters: list[str] | None = None) -> str:
    """Chat-template text of one row: system, user(state + branch), assistant-turn opener, <decide>."""
    content: list[dict[str, Any]] = [{"type": "text", "text": "State:\n"}]
    for seg in segments:
        content.append({"type": "image"} if isinstance(seg, ImageSlot) else {"type": "text", "text": sanitize(seg)})
    content.append({"type": "text", "text": branch_text(q, letters)})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return text + SPECIAL["decide"]


def target_vector(q: dict[str, Any], target: Any) -> list[float]:
    """Soft target over the rendered options."""
    if q["type"] == "noul":
        p = float(target)
        return [p, 1.0 - p]
    if q["type"] == "choice":
        return [float(target[label]) for label in q["criteria"]]
    return [float(target[str(i)]) for i in range(len(q["criteria"]))]


def special_ids(tok) -> dict[str, int]:
    ids = {}
    for name, text in SPECIAL.items():
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) != 1:
            raise RuntimeError(f"delimiter {text!r} is not a single token: {enc}")
        ids[name] = enc[0]
    return ids


def prior_row(processor, segments: list, q: dict[str, Any], letters: list[str]) -> tuple[str, list[int]]:
    """Prior-mode row: exactly the zero-shot prompt of vev.readout (system prompt, 'State:', 'Question:', option lines,
    answer hint, generation prompt) with NO delimiter tokens, so the LM's next-token logits at the last position are
    the zero-shot readout by construction. Returns (text, char offsets just past the end of each option span); the
    pointer head keys on the token holding each span's last character, and queries the last token of the row."""
    labs = prior_labels(q, letters)
    crit = q.get("criteria")
    ends: list[int] = []
    if q["type"] == "choice":
        head = "Options:\n"
        items = [f"{labs[j]}. {sanitize(str(label))}" + (f" - {sanitize(d)}" if (d := _render_desc(desc)) else "")
                 for j, (label, desc) in enumerate(crit.items())]
        tail_hint = "\n\nAnswer with the letter of the single best option."
    elif q["type"] == "score":
        head = f"Scale, from lowest (0) to highest ({len(crit) - 1}):\n"
        items = [f"{j}. {sanitize(_render_desc(d))}".rstrip() for j, d in enumerate(crit)]
        tail_hint = "\n\nAnswer with the level number only."
    else:
        crit = crit or {}
        head = "Answer Yes or No."
        items = []
        if crit.get("true") is not None:
            items.append(f"Yes means: {sanitize(_render_desc(crit['true']))}")
        if crit.get("false") is not None:
            items.append(f"No means: {sanitize(_render_desc(crit['false']))}")
        tail_hint = ""
    fmt = head
    if q["type"] == "noul":
        if items:
            for it in items:
                fmt += "\n" + it
                ends.append(len(fmt))
            if len(items) == 1:  # only one side described: key both options on the 'Yes'/'No' words of the hint
                ends = [len("Answer Yes"), len("Answer Yes or No")]
        else:
            ends = [len("Answer Yes"), len("Answer Yes or No")]
    else:
        fmt += "\n".join(items)
        pos = len(head)
        for it in items:
            pos += len(it)
            ends.append(pos)
            pos += 1  # newline
        fmt += tail_hint
    instructions = sanitize((q.get("instructions") or "").strip())
    tail = "\n\n" + (f"Question: {instructions}\n\n" if instructions else "") + fmt
    content: list[dict[str, Any]] = [{"type": "text", "text": "State:\n"}]
    for seg in segments:
        content.append({"type": "image"} if isinstance(seg, ImageSlot) else {"type": "text", "text": sanitize(seg)})
    content.append({"type": "text", "text": tail})
    messages = [{"role": "system", "content": [{"type": "text", "text": ZEROSHOT_SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    start = text.rfind(tail)
    if start < 0:
        raise RuntimeError("chat template altered the question text")
    base = start + len(tail) - len(fmt)  # fmt is the suffix of tail
    return text, [base + e for e in ends]


def encode_prior_row(processor, text: str, images: list[Image.Image], char_ends: list[int]) -> dict[str, Any]:
    """Tokenize a prior-mode row and map option char ends to token positions (offset mapping on the plain text, then
    shifted by the image-pad expansion, which sits entirely before the question)."""
    tok = processor.tokenizer
    plain = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    offs = plain["offset_mapping"]
    def tok_index(char_end: int) -> int:
        idx = 0
        for i, (a, b) in enumerate(offs):
            if a < char_end:
                idx = i
        return idx
    enc = processor(text=[text], images=images or None, return_tensors="pt")
    row: dict[str, Any] = {}
    for key, v in enc.items():
        if key in ("input_ids", "attention_mask", "mm_token_type_ids") or (v.dim() == 2 and v.shape[0] == 1 and key != "image_grid_thw"):
            row[key] = v[0]
        else:
            row[key] = v
    shift = int(row["input_ids"].shape[0]) - len(plain["input_ids"])
    opt_idx = []
    for ce in char_ends:
        i = tok_index(ce)
        j = i + shift
        if int(row["input_ids"][j]) != plain["input_ids"][i]:
            raise RuntimeError("token alignment failed between plain and expanded encodings")
        opt_idx.append(j)
    row["decide"], row["opt_idx"] = int(row["input_ids"].shape[0]) - 1, opt_idx
    return row


class LabelTokens:
    """Single-token answer labels of the tokenizer: letters for choice (A..Z, AA..), Yes/No, digits."""

    def __init__(self, tok):
        self.letters = choice_labels(tok, MAX_CHOICE_OPTIONS)
        self.ids: dict[str, int] = {}
        for s in self.letters + ["Yes", "No"] + [str(i) for i in range(MAX_SCORE_LEVELS)]:
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"answer label {s!r} is not a single token: {enc}")
            self.ids[s] = enc[0]

    def answer_ids(self, q: dict[str, Any]) -> list[int]:
        return [self.ids[s] for s in prior_labels(q, self.letters)]


def find_positions(ids: torch.Tensor, sid: dict[str, int], k: int) -> tuple[int, list[int]]:
    """(index of <decide>, indices of the K </opt> tokens) in one un-padded row."""
    opt_end = (ids == sid["opt_end"]).nonzero(as_tuple=True)[0].tolist()
    decide = (ids == sid["decide"]).nonzero(as_tuple=True)[0].tolist()
    if len(opt_end) != k or len(decide) != 1 or decide[0] != int(ids.shape[0]) - 1:
        raise ValueError(f"row layout mismatch: {len(opt_end)} option ends for {k} options, decide at {decide} of {ids.shape[0]}")
    return decide[0], opt_end


class PointerHead(nn.Module):
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1.0 / math.sqrt(dp)
        nn.init.xavier_uniform_(self.q.weight)
        nn.init.xavier_uniform_(self.k.weight)
        nn.init.zeros_(self.q.bias)
        nn.init.zeros_(self.k.bias)

    def zero_output(self) -> None:
        """Start the head at exactly zero logits (prior mode): W_q = 0 keeps dz/dW_q nonzero through k."""
        nn.init.zeros_(self.q.weight)
        nn.init.zeros_(self.q.bias)

    def forward(self, h_decide: torch.Tensor, h_opts: torch.Tensor) -> torch.Tensor:  # [d], [K, d] -> [K]
        # parameter-free LayerNorm on the inputs: the backbone's final hidden states have outlier dimensions, so raw
        # dot products start at |z| ~ 50 (initial NLL ~ 10); normalized inputs start the head near uniform
        d = h_decide.shape[-1]
        h_decide = F.layer_norm(h_decide, (d,))
        h_opts = F.layer_norm(h_opts, (d,))
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


LORA_TARGETS = (
    r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))"
)
# retention ablation (kev "attn"): attention and DeltaNet projections only, MLPs untouched
LORA_TARGETS_ATTN = (
    r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))"
)


def load_backbone(base_id: str, dtype: torch.dtype, attn_implementation: str | None = None):
    """The multimodal backbone (vision tower + language model) without lm_head: we never generate text."""
    from transformers import AutoModelForImageTextToText

    kw = {"dtype": dtype}
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    full = AutoModelForImageTextToText.from_pretrained(base_id, **kw)
    backbone = full.model
    lm_head_weight = full.lm_head.weight  # tied to the input embeddings on Qwen3.5; a reference either way
    del full
    return backbone, lm_head_weight


class PointerModel(nn.Module):
    """Backbone (+ optional LoRA on the language model) + fp32 pointer head."""

    def __init__(self, base_id: str, dtype: torch.dtype = torch.bfloat16, lora_r: int = 0, lora_alpha: int | None = None,
                 lora_dropout: float = 0.05, dp: int = 256, gradient_checkpointing: bool = False,
                 attn_implementation: str | None = None, prior: bool = False, lora_targets: str = "all"):
        super().__init__()
        self.backbone, lm_head_weight = load_backbone(base_id, dtype, attn_implementation)
        self.prior = prior
        self.head_scale = 1.0  # prior mode: final = prior + head_scale * head (alpha scan without retraining)
        # lm_head rows are read for the zero-shot prior only; kept as a plain attribute so it is not a trainable
        # parameter of this module (it is the frozen, tied embedding matrix on Qwen3.5)
        self._lm_head_weight = lm_head_weight if prior else None
        self.hidden_size = int(self.backbone.config.get_text_config().hidden_size)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lora_r = lora_r
        if lora_r:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=lora_dropout,
                             target_modules=LORA_TARGETS_ATTN if lora_targets == "attn" else LORA_TARGETS, bias="none")
            self.backbone = get_peft_model(self.backbone, cfg)
        self.head = PointerHead(self.hidden_size, dp).float()
        if prior:
            self.head.zero_output()

    def hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.backbone(**batch, use_cache=False)
        return out.last_hidden_state

    def prior_logits(self, h_decide: torch.Tensor, answer_ids: list[int]) -> torch.Tensor:
        w = self._lm_head_weight[torch.tensor(answer_ids, device=h_decide.device)]
        return F.linear(h_decide.float(), w.float())

    def forward(self, batch: dict[str, torch.Tensor], readouts: list[tuple[int, list[int]]],
                answer_ids: list[list[int]] | None = None, return_prior: bool = False):
        """readouts[i] = (decide index, option-end indices) of row i. Returns one logits tensor per row (fp32);
        in prior mode the logits are prior + head, and return_prior=True also returns the prior logits."""
        return self.readout(self.hidden(batch), readouts, answer_ids, return_prior)

    def readout(self, h: torch.Tensor, readouts: list[tuple[int, list[int]]], answer_ids: list[list[int]] | None = None,
                return_prior: bool = False):
        """Head (+ prior) on backbone hidden states h [B, L, d]; readouts index into each row of h."""
        out, priors = [], []
        with torch.autocast(device_type=h.device.type, enabled=False):  # head stays fp32 even under bf16 autocast
            for i, (d, opts) in enumerate(readouts):
                hi = h[i].float()
                z = self.head(hi[d], hi[torch.tensor(opts, device=h.device)])
                if self.prior:
                    if answer_ids is None:
                        raise ValueError("prior mode needs answer_ids per row")
                    # prior rows carry no delimiter tokens: d is the last token of the generation prompt, exactly where
                    # the zero-shot readout reads its label logits
                    pz = self.prior_logits(hi[d], answer_ids[i])
                    priors.append(pz)
                    z = self.head_scale * z + pz
                out.append(z)
        return (out, priors) if return_prior else out

    def trainable_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        head = list(self.head.parameters())
        head_ids = {id(p) for p in head}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in head_ids]
        return rest, head

    def save(self, out_dir: str | Path, meta: dict[str, Any]) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.lora_r:
            self.backbone.save_pretrained(str(out / "adapter"))
        torch.save(self.head.state_dict(), out / "head.pt")
        (out / "vev_pointer.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_checkpoint(ckpt: str | Path) -> Path:
    """A local checkpoint directory, or a Hugging Face repo id downloaded to the local cache."""
    p = Path(ckpt)
    if (p / "vev_pointer.json").is_file():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(ckpt)))


def is_checkpoint(ckpt: str | Path) -> bool:
    """True for a directory or Hugging Face repo that carries vev_pointer.json."""
    if (Path(ckpt) / "vev_pointer.json").is_file():
        return True
    if Path(ckpt).exists() or str(ckpt).count("/") != 1:
        return False
    from huggingface_hub import file_exists, try_to_load_from_cache

    try:
        if isinstance(try_to_load_from_cache(str(ckpt), "vev_pointer.json"), str):   # works offline
            return True
        return file_exists(str(ckpt), "vev_pointer.json")
    except Exception:
        return False


def base_source(ckpt: Path, meta: dict[str, Any]) -> str:
    """Where the backbone and processor come from: the base model id, or the checkpoint itself for merged releases."""
    return str(ckpt) if meta["base"] == "." else meta["base"]


def load_pointer_checkpoint(ckpt_dir: str | Path, dtype: torch.dtype, device: torch.device,
                            attn_implementation: str | None = None) -> tuple[PointerModel, dict[str, Any]]:
    """A checkpoint written by PointerModel.save (base + adapter + head) or a merged release (base "."), in eval mode."""
    ckpt = resolve_checkpoint(ckpt_dir)
    meta = json.loads((ckpt / "vev_pointer.json").read_text(encoding="utf-8"))
    model = PointerModel(base_source(ckpt, meta), dtype=dtype, lora_r=0, dp=int(meta.get("dp", 256)),
                         attn_implementation=attn_implementation, prior=bool(meta.get("prior", False)))
    if (ckpt / "adapter").exists():
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(model.backbone, str(ckpt / "adapter"))
        model.backbone = peft_model.merge_and_unload()
    model.head.load_state_dict(torch.load(ckpt / "head.pt", map_location="cpu"))
    model.to(device).eval()
    return model, meta


def collate_rows(rows: list[dict[str, Any]], pad_id: int) -> tuple[dict[str, torch.Tensor], list[tuple[int, list[int]]]]:
    """Right-pad encoded rows into one batch; pixel_values / image_grid_thw are concatenated in row order (the model
    matches image placeholders to grid entries sequentially across the batch)."""
    L = max(int(r["input_ids"].shape[0]) for r in rows)
    batch: dict[str, torch.Tensor] = {}
    seq_keys = [k for k, v in rows[0].items() if isinstance(v, torch.Tensor) and v.dim() == 1 and k not in ("target",)
                and v.shape[0] == rows[0]["input_ids"].shape[0]]
    for key in seq_keys:
        fill = pad_id if key == "input_ids" else 0
        t = torch.full((len(rows), L), fill, dtype=rows[0][key].dtype)
        for i, r in enumerate(rows):
            t[i, : r[key].shape[0]] = r[key]
        batch[key] = t
    if "attention_mask" not in batch:
        m = torch.zeros((len(rows), L), dtype=torch.long)
        for i, r in enumerate(rows):
            m[i, : r["input_ids"].shape[0]] = 1
        batch["attention_mask"] = m
    pv = [r["pixel_values"] for r in rows if r.get("pixel_values") is not None]
    if pv:
        batch["pixel_values"] = torch.cat(pv, 0)
        batch["image_grid_thw"] = torch.cat([r["image_grid_thw"] for r in rows if r.get("image_grid_thw") is not None], 0)
    readouts = [(r["decide"], r["opt_idx"]) for r in rows]
    return batch, readouts


def encode_row(processor, text: str, images: list[Image.Image], sid: dict[str, int], k: int) -> dict[str, Any]:
    """Tokenize one row (with its images) and locate the readout positions. Tensors are un-padded, batch dim dropped."""
    enc = processor(text=[text], images=images or None, return_tensors="pt")
    row: dict[str, Any] = {}
    for key, v in enc.items():
        if key in ("input_ids", "attention_mask", "mm_token_type_ids") or (v.dim() == 2 and v.shape[0] == 1 and key not in ("image_grid_thw",)):
            row[key] = v[0]
        else:
            row[key] = v
    row["decide"], row["opt_idx"] = find_positions(row["input_ids"], sid, k)
    return row


# One empty-instruction question per type: rows built from them share exactly the state part of any real row.
PROBE_QUESTIONS = [{"type": "choice", "instructions": "", "criteria": {"x": None}},
                   {"type": "noul", "instructions": "", "criteria": None},
                   {"type": "score", "instructions": "", "criteria": ["x"]}]


def common_prefix_len(seqs: list[list[int]]) -> int:
    n = min(len(x) for x in seqs)
    for i in range(n):
        if any(x[i] != seqs[0][i] for x in seqs[1:]):
            return i
    return n


def fork_cache(cache, n: int = 1):
    """Copy of a prefilled hybrid cache that n branches (batch rows) continue independently. Attention K/V are only
    ever replaced via torch.cat, never written in place, so they are shared (expanded to n rows); DeltaNet conv and
    recurrent states are updated in place (copy_), so they are cloned."""
    new = copy.copy(cache)
    new.layers = []
    for layer in cache.layers:
        lay = copy.copy(layer)
        for k, v in vars(layer).items():
            if isinstance(v, dict):
                setattr(lay, k, dict(v))
        for states in (getattr(lay, "conv_states", None), getattr(lay, "recurrent_states", None)):
            for i, t in (states or {}).items():
                if t is not None:
                    states[i] = t.repeat(n, *([1] * (t.dim() - 1)))
        if isinstance(getattr(layer, "keys", None), torch.Tensor):
            lay.keys = layer.keys.expand(n, *layer.keys.shape[1:])
            lay.values = layer.values.expand(n, *layer.values.shape[1:])
        new.layers.append(lay)
    return new


class PointerEngine:
    """Serving backend for a pointer checkpoint: one row per question. With prefix_share the state prefix
    is prefilled once and every question continues from a fork of that cache; branches run one at a time unless
    branch_batch (right-padded batch: faster, but not bit-identical to running alone in bf16)."""

    def __init__(self, ckpt_dir: str, dtype: str = "bf16", device: str = "cuda",
                 temperatures: dict[str, float] | None = None, attn_implementation: str | None = None,
                 head_scale: float | None = None, prefix_share: bool = True, branch_batch: bool = False,
                 prefix_min_tokens: int = 4096):
        from transformers import AutoProcessor

        self.device = torch.device(device)
        self.model, self.meta = load_pointer_checkpoint(ckpt_dir, DTYPES[dtype], self.device, attn_implementation)
        if head_scale is not None:
            self.model.head_scale = float(head_scale)
        self.base_id = self.meta.get("base_model") or self.meta["base"]
        self.processor = AutoProcessor.from_pretrained(base_source(resolve_checkpoint(ckpt_dir), self.meta))
        self.tok = self.processor.tokenizer
        self.merge = self.processor.image_processor.merge_size
        self.sid = special_ids(self.tok)
        self.labels = LabelTokens(self.tok) if self.model.prior else None
        self.temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0, **(self.meta.get("temperatures") or {}),
                             **(temperatures or {})}
        self.autocast_dtype = DTYPES[dtype] if dtype != "fp32" else None
        self.prefix_share = prefix_share
        self.branch_batch = branch_batch
        self.prefix_min_tokens = prefix_min_tokens  # share only states at least this long: depends on the state alone, so isolation stays exact
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")

    def _autocast(self):
        on = self.autocast_dtype is not None and self.device.type == "cuda"
        return torch.autocast("cuda", dtype=self.autocast_dtype or torch.bfloat16, enabled=on)

    @torch.inference_mode()
    def _logits(self, row: dict[str, Any]) -> torch.Tensor:
        batch, readouts = collate_rows([row], self.tok.pad_token_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        ids = [row["answer_ids"]] if self.model.prior else None
        with self._autocast():
            z = self.model(batch, readouts, ids)[0]
        return z.float().cpu()

    def _row_text(self, segments: list, q: dict[str, Any]) -> tuple[str, list[int] | None]:
        if self.labels:
            return prior_row(self.processor, segments, q, self.labels.letters)
        return row_text(self.processor, segments, q), None

    def _prefix_len(self, segments: list, rows: list[dict[str, Any]]) -> int:
        """Length of the shared prefix, taken from state-only probe rows so it depends on the state alone: a question
        then goes through the same computation whatever its siblings are, and isolation stays exact. Falls back to
        the rows' common prefix if some row tokenizes the state boundary differently."""
        grid = rows[0].get("image_grid_thw")
        counts = [] if grid is None else [int(g.prod()) // self.merge**2 for g in grid]

        def expand(ids: list[int]) -> list[int]:  # the processor's image-pad expansion
            it, out = iter(counts), []
            for t in ids:
                out.extend([t] * next(it) if t == self.image_pad_id else [t])
            return out

        probes = [expand(self.tok(self._row_text(segments, q)[0], add_special_tokens=False)["input_ids"])
                  for q in PROBE_QUESTIONS]
        ids = [r["input_ids"].tolist() for r in rows]
        n = common_prefix_len(probes)
        if any(x[:n] != probes[0][:n] for x in ids):
            n = common_prefix_len(probes + ids)
        mm = rows[0].get("mm_token_type_ids")
        if mm is not None and bool((mm[n:] != 0).any()):
            raise RuntimeError("image tokens outside the shared state prefix")
        if any(n > min(min(r["opt_idx"]), r["decide"]) for r in rows):
            raise RuntimeError("readout position inside the shared prefix")
        return n

    def _positions(self, row: dict[str, Any]) -> torch.Tensor:
        """Full-row position ids exactly as the un-cached forward builds them: 4 x L (text + 3 mrope) for text rows,
        3 x L mrope for rows with images. Passed explicitly so a cached branch never falls back to the model's
        stored rope_deltas (left over from whichever image request ran last)."""
        ids = row["input_ids"]
        if row.get("image_grid_thw") is not None:
            pos, _ = self.model.backbone.get_rope_index(ids[None], mm_token_type_ids=row["mm_token_type_ids"][None],
                                                        image_grid_thw=row["image_grid_thw"])
            return pos[:, 0]
        return torch.arange(ids.shape[0]).expand(4, -1)

    @torch.inference_mode()
    def _logits_shared(self, segments: list, rows: list[dict[str, Any]]) -> list[torch.Tensor]:
        from transformers import DynamicCache

        bb, dev = self.model.backbone, self.device
        n = self._prefix_len(segments, rows)
        pos = [self._positions(r) for r in rows]
        first = rows[0]
        pre = {"input_ids": first["input_ids"][None, :n], "attention_mask": torch.ones(1, n, dtype=torch.long),
               "position_ids": pos[0][:, None, :n]}
        if first.get("pixel_values") is not None:
            pre["pixel_values"], pre["image_grid_thw"] = first["pixel_values"], first["image_grid_thw"]
        answer_ids = [r["answer_ids"] for r in rows] if self.model.prior else None
        readouts = [(r["decide"] - n, [o - n for o in r["opt_idx"]]) for r in rows]
        with self._autocast():
            cache = DynamicCache(config=bb.config)
            bb(**{k: v.to(dev) for k, v in pre.items()}, past_key_values=cache, use_cache=True)
            if self.branch_batch:
                sfx = [r["input_ids"][n:] for r in rows]
                S = max(int(x.shape[0]) for x in sfx)
                ids = torch.full((len(rows), S), self.tok.pad_token_id, dtype=first["input_ids"].dtype)
                mask = torch.zeros((len(rows), n + S), dtype=torch.long)
                p = torch.zeros((pos[0].shape[0], len(rows), S), dtype=pos[0].dtype)
                for i, x in enumerate(sfx):
                    ids[i, : x.shape[0]] = x
                    mask[i, : n + x.shape[0]] = 1
                    p[:, i, : x.shape[0]] = pos[i][:, n:]
                h = bb(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=p.to(dev),
                       past_key_values=fork_cache(cache, len(rows)), use_cache=True).last_hidden_state
                zs = self.model.readout(h, readouts, answer_ids)
            else:
                zs = []
                for i, r in enumerate(rows):
                    x = r["input_ids"][n:]
                    mask = torch.ones(1, n + x.shape[0], dtype=torch.long, device=dev)
                    h = bb(input_ids=x[None].to(dev), attention_mask=mask, position_ids=pos[i][:, None, n:].to(dev),
                           past_key_values=fork_cache(cache), use_cache=True).last_hidden_state
                    zs += self.model.readout(h, [readouts[i]], [answer_ids[i]] if answer_ids else None)
        return [z.float().cpu() for z in zs]

    def run(self, state: Any, questions: dict[str, dict[str, Any]], max_images: int = 8, max_pixels: int = 1024 * 1024,
            allow_remote: bool = False, max_state_tokens: int = 32768, max_request_tokens: int = 65536) -> Result:
        t0 = time.perf_counter()
        segments, images = serialize_state(state, max_images=max_images, max_pixels=max_pixels, allow_remote=allow_remote)
        rows = {}
        for name, q in questions.items():
            text, ends = self._row_text(segments, q)
            if self.labels:
                rows[name] = encode_prior_row(self.processor, text, images, ends)
                rows[name]["answer_ids"] = self.labels.answer_ids(q)
            else:
                rows[name] = encode_row(self.processor, text, images, self.sid, len(option_spans(q)))
        first = next(iter(rows.values()))
        state_text, state_image = count_state_tokens(self.tok, segments, first, self.merge)
        state_tokens = state_text + state_image
        row_lens = {name: int(r["input_ids"].shape[0]) for name, r in rows.items()}
        longest = max(row_lens.values())
        if longest > max_state_tokens:
            raise InvalidRequest(400, "invalid_request_error", f"state + longest question is {longest} tokens (max {max_state_tokens})")
        question_tokens = {name: n - state_tokens for name, n in row_lens.items()}
        input_tokens = state_tokens + sum(question_tokens.values())
        if input_tokens > max_request_tokens:
            raise InvalidRequest(400, "invalid_request_error", f"request is {input_tokens} tokens (max {max_request_tokens})")
        t1 = time.perf_counter()
        answers: dict[str, Any] = {}
        raw_logits: dict[str, list[float]] = {}
        if self.prefix_share and state_tokens >= self.prefix_min_tokens:
            logits = dict(zip(rows, self._logits_shared(segments, list(rows.values()))))
        else:
            logits = {name: self._logits(r) for name, r in rows.items()}
        for name, q in questions.items():
            z = logits[name]
            probs = torch.softmax(z / self.temperatures.get(q["type"], 1.0), dim=0)
            raw_logits[name] = z.tolist()
            answers[name] = self._answer(q, probs.tolist())
        t2 = time.perf_counter()
        return Result(
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=count_output_tokens(self.tok, answers),
            extensions={
                "logits": raw_logits,
                "timing": {"prefill_ms": round((t1 - t0) * 1000, 2), "questions_ms": round((t2 - t1) * 1000, 2),
                           "total_ms": round((t2 - t0) * 1000, 2)},
                "tokens": {"state_text": state_text, "state_image": state_image, "questions": question_tokens},
            },
        )

    @staticmethod
    def _answer(q: dict[str, Any], probs: list[float]) -> dict[str, Any]:
        if q["type"] == "noul":
            return {"type": "noul", "noul": probs[0]}
        k = len(probs)
        best = max(range(k), key=lambda i: (probs[i], -i))
        if q["type"] == "choice":
            labels = list(q["criteria"].keys())
            return {"type": "choice", "choice": labels[best], "confidence": confidence(probs[best], k),
                    "probabilities": dict(zip(labels, probs))}
        return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": confidence(probs[best], k),
                "probabilities": {str(i): p for i, p in enumerate(probs)}, "legend": {str(i): d for i, d in enumerate(q["criteria"])}}

    def warmup(self) -> None:
        from vev.readout import Engine

        Engine.warmup(self)  # same three-question probe; only uses self.run


def image_loader_for(base_dir: Path, max_pixels: int) -> Callable[[dict], Image.Image]:
    def load(value: dict) -> Image.Image:
        img = Image.open(base_dir / value["path"])
        img.load()
        img = img.convert("RGB")
        w, h = img.size
        if w * h > max_pixels:
            s = math.sqrt(max_pixels / (w * h))
            img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.BICUBIC)
        return img
    return load


def row_loss(z: torch.Tensor, t: torch.Tensor, qtype: str, brier_w: float = 0.1, score_w: float = 0.05) -> torch.Tensor:
    """Soft-label cross-entropy + brier_w * Brier (+ score_w * squared normalized expectation gap for score)."""
    logp = F.log_softmax(z, -1)
    p = logp.exp()
    loss = -(t * logp).sum() + brier_w * (p - t).square().sum()
    k = z.shape[0]
    if qtype == "score" and score_w > 0 and k > 1:
        idx = torch.arange(k, device=z.device, dtype=z.dtype)
        loss = loss + score_w * (((p * idx).sum() - (t * idx).sum()) / (k - 1)).square()
    return loss
