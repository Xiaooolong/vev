"""Vev checkpoints: prompt rendering, the model used for training, checkpoint loading and the serving engine.

A question is one row: the zero-shot prompt of vev.readout (system prompt, state, question, options, answer hint),
and its answer probabilities are the LM-head logits of the answer tokens at the last position, renormalised over the
allowed answers. Training (train/) and serving (CheckpointEngine) share the rendering below."""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from vev import bidir
from vev.readout import DTYPES, SYSTEM_PROMPT, LabelTokens, Result, make_answer, windows_sdpa_workaround
from vev.state import ImageSlot, InvalidRequest, render_desc, serialize_state
from vev.tokens import count_output_tokens, count_state_tokens

MARKER = "vev.json"
log = logging.getLogger("vev")
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def sanitize(text: str) -> str:
    """Caller text can never produce a special token: `<|name|>` becomes `<¦name¦>`."""
    return _SPECIAL_RE.sub(r"<¦\1¦>", text)


def question_text(q: dict[str, Any], labels: list[str]) -> str:
    """Question, options and answer hint, as in vev.readout.answer_format but with caller text sanitized."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        items = [f"{labels[j]}. {sanitize(str(label))}" + (f" - {sanitize(d)}" if (d := render_desc(desc)) else "")
                 for j, (label, desc) in enumerate(crit.items())]
        fmt = "Options:\n" + "\n".join(items) + "\n\nAnswer with the letter of the single best option."
    elif q["type"] == "score":
        items = [f"{j}. {sanitize(render_desc(d))}".rstrip() for j, d in enumerate(crit)]
        fmt = (f"Scale, from lowest (0) to highest ({len(crit) - 1}):\n" + "\n".join(items)
               + "\n\nAnswer with the level number only.")
    else:
        crit = crit or {}
        fmt = "Answer Yes or No."
        if crit.get("true") is not None:
            fmt += f"\nYes means: {sanitize(render_desc(crit['true']))}"
        if crit.get("false") is not None:
            fmt += f"\nNo means: {sanitize(render_desc(crit['false']))}"
    instructions = sanitize((q.get("instructions") or "").strip())
    return "\n\n" + (f"Question: {instructions}\n\n" if instructions else "") + fmt


def render_row(processor, segments: list, q: dict[str, Any], labels: LabelTokens, image_mode: str = "",
               read_twice: str = "", read_twice_max: int = 3000) -> str:
    """The prompt of one question. image_mode (requests with images and one question): "twice" writes the state, the
    question's instructions, the state again (its images included) and then the question; "qfirst" writes the
    instructions before the state. read_twice (text-only states of at most read_twice_max tokens): "state" writes the
    state twice before the question; "full" writes the state, the instructions, the state again and the question.
    All off: the v0.1 layout."""
    def state_part(head: str) -> list[dict[str, Any]]:
        part: list[dict[str, Any]] = [{"type": "text", "text": head}]
        for seg in segments:
            part.append({"type": "image"} if isinstance(seg, ImageSlot) else {"type": "text", "text": sanitize(seg)})
        return part

    instructions = sanitize((q.get("instructions") or "").strip())
    if image_mode == "qfirst":
        content = state_part((f"Question: {instructions}\n\n" if instructions else "") + "State:\n")
        content.append({"type": "text", "text": question_text(q, labels.letters)})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if image_mode == "twice":
        content = state_part("State:\n")
        if instructions:
            content.append({"type": "text", "text": f"\n\nQuestion: {instructions}"})
        content += state_part("\n\nState (repeated):\n")
        content.append({"type": "text", "text": question_text(q, labels.letters)})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": content}]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    content = state_part("State:\n")
    twice = read_twice in ("state", "full") and not any(isinstance(s, ImageSlot) for s in segments) and \
        sum(len(processor.tokenizer.encode(sanitize(s), add_special_tokens=False)) for s in segments) <= read_twice_max
    if twice and read_twice == "full":
        if instructions:
            content.append({"type": "text", "text": f"\n\nQuestion: {instructions}"})
    if twice:
        content += state_part("\n\nState (repeated):\n")
    content.append({"type": "text", "text": question_text(q, labels.letters)})
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": content}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def encode_row(processor, text: str, images: list[Image.Image]) -> dict[str, Any]:
    """Tokenize one row with its images; tensors are un-padded with the batch dim dropped. The answer is read at the
    last position (`decide`)."""
    enc = processor(text=[text], images=images or None, return_tensors="pt")
    row: dict[str, Any] = {}
    for key, v in enc.items():
        if key in ("input_ids", "attention_mask", "mm_token_type_ids") or (v.dim() == 2 and v.shape[0] == 1 and key != "image_grid_thw"):
            row[key] = v[0]
        else:
            row[key] = v
    row["decide"] = int(row["input_ids"].shape[0]) - 1
    return row


def image_inputs(processor, images: list[Image.Image], device: torch.device | str | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """pixel_values and image_grid_thw for images, as processor(images=...) returns them, with each distinct image
    object going through the image processor once (the image-read-twice layout lists every image twice) and its rows
    repeated. With a device, the processor resizes on the CPU as usual but returns uint8 patches, and its own fused
    rescale/normalize then runs on the device: one subtraction and one division per element, the same float32 values,
    a quarter of the bytes to copy. image_grid_thw stays on the CPU."""
    seen: dict[int, int] = {}
    distinct, index = [], []
    for im in images:
        if id(im) not in seen:
            seen[id(im)] = len(distinct)
            distinct.append(im)
        index.append(seen[id(im)])
    ip = processor.image_processor
    late = device is not None and ip.do_rescale and ip.do_normalize and hasattr(ip, "rescale_and_normalize")
    enc = ip(distinct, return_tensors="pt", **({"do_rescale": False, "do_normalize": False} if late else {}))
    pv, grid = enc["pixel_values"], enc["image_grid_thw"]
    if late:
        n, channels = pv.shape[0], pv.shape[1] // (ip.temporal_patch_size * ip.patch_size ** 2)
        x = pv.to(device).view(n, channels, -1, 1)  # channel-major patches, normalized like an image tensor
        mean, std = (tuple(v) if isinstance(v, list) else v for v in (ip.image_mean, ip.image_std))  # as preprocess()
        pv = ip.rescale_and_normalize(x, ip.do_rescale, ip.rescale_factor, ip.do_normalize, mean, std).reshape(n, -1)
    elif device is not None:
        pv = pv.to(device)
    if len(index) == len(distinct):
        return pv, grid
    parts = pv.split(grid.prod(-1).tolist())
    return torch.cat([parts[i] for i in index]), grid[index]


def encode_rows(processor, texts: list[str], images: list[Image.Image],
                device: torch.device | str | None = None) -> list[dict[str, Any]]:
    """encode_row for several prompts that contain the same images. Without images, the first row goes through the
    full processor. With images, image_inputs() runs the image processor once per distinct image (on `device`, see
    there), the prompts are tokenized with a single <|image_pad|> per image, and each placeholder position is then
    repeated grid_k / merge² times in every field, which is what the processor's expansion produces. All rows share
    the same pixel_values and image_grid_thw."""
    if not images:
        first = encode_row(processor, texts[0], images)
        if len(texts) == 1:
            return [first]
        todo = texts[1:]
        rows, pixel_values, grid, counts = [first], None, None, []
    else:
        pixel_values, grid = image_inputs(processor, images, device)
        counts = [int(g.prod()) // processor.image_processor.merge_size ** 2 for g in grid]
        todo, rows = texts, []
    enc = processor(text=todo)
    pad_id = processor.image_token_id
    for i in range(len(todo)):
        ids = enc["input_ids"][i]
        where = [p for p, x in enumerate(ids) if x == pad_id]
        if len(where) != len(counts):
            raise RuntimeError("image placeholders do not match the images")
        row: dict[str, Any] = {}
        for k in enc.keys():
            seq, out, last = enc[k][i], [], 0
            for p, c in zip(where, counts):
                out += seq[last:p] + [seq[p]] * c
                last = p + 1
            row[k] = torch.tensor(out + seq[last:], dtype=torch.long)
        if images:
            row["pixel_values"], row["image_grid_thw"] = pixel_values, grid
        row["decide"] = int(row["input_ids"].shape[0]) - 1
        rows.append(row)
    return rows


def collate_rows(rows: list[dict[str, Any]], pad_id: int) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Right-pad encoded rows into one batch; pixel_values / image_grid_thw are concatenated in row order (the model
    matches image placeholders to grid entries sequentially across the batch)."""
    L = max(int(r["input_ids"].shape[0]) for r in rows)
    batch: dict[str, torch.Tensor] = {}
    seq_keys = [k for k, v in rows[0].items() if isinstance(v, torch.Tensor) and v.dim() == 1 and k != "target"
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
    return batch, [r["decide"] for r in rows]


LORA_TARGETS = (
    r"language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))"
)


class VevModel(nn.Module):
    """Qwen3.5 backbone (vision tower + language model, frozen) with optional LoRA on the language model. Only the LM
    head rows of the answer tokens are ever computed."""

    def __init__(self, base_id: str, dtype: torch.dtype = torch.bfloat16, lora_r: int = 0, lora_alpha: int | None = None,
                 lora_dropout: float = 0.05, gradient_checkpointing: bool = False, attn_implementation: str | None = None):
        super().__init__()
        from transformers import AutoModelForImageTextToText

        kw: dict[str, Any] = {"dtype": dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        full = AutoModelForImageTextToText.from_pretrained(base_id, **kw)
        self.backbone = full.model
        self.lm_head_weight = full.lm_head.weight  # tied to the input embeddings on Qwen3.5-4B, separate on 9B
        del full
        self.lm_head_weight.requires_grad_(False)
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.lora_r = lora_r
        if lora_r:
            from peft import LoraConfig, get_peft_model

            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha or 2 * lora_r, lora_dropout=lora_dropout,
                             target_modules=LORA_TARGETS, bias="none")
            self.backbone = get_peft_model(self.backbone, cfg)

    def hidden(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.backbone(**batch, use_cache=False).last_hidden_state

    def forward(self, batch: dict[str, torch.Tensor], decide: list[int], answer_ids: list[list[int]]) -> list[torch.Tensor]:
        return self.readout(self.hidden(batch), decide, answer_ids)

    def readout(self, h: torch.Tensor, decide: list[int], answer_ids: list[list[int]]) -> list[torch.Tensor]:
        """fp32 answer-token logits at position decide[i] of row i of h [B, L, d]."""
        out = []
        with torch.autocast(device_type=h.device.type, enabled=False):
            for i, d in enumerate(decide):
                w = self.lm_head_weight[torch.tensor(answer_ids[i], device=h.device)]
                out.append(F.linear(h[i].float()[d], w.float()))
        return out

    def save(self, out_dir: str | Path, meta: dict[str, Any]) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.lora_r:
            self.backbone.save_pretrained(str(out / "adapter"))
        (out / MARKER).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_checkpoint(ckpt: str | Path, revision: str | None = None) -> Path:
    """A local checkpoint directory, or a Hugging Face repo id (at `revision`) downloaded to the local cache."""
    p = Path(ckpt)
    if (p / MARKER).is_file():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(ckpt), revision=revision))


def is_checkpoint(ckpt: str | Path, revision: str | None = None) -> bool:
    """True for a directory or Hugging Face repo that carries vev.json."""
    if (Path(ckpt) / MARKER).is_file():
        return True
    if Path(ckpt).exists() or str(ckpt).count("/") != 1:
        return False
    from huggingface_hub import file_exists, try_to_load_from_cache

    try:
        if isinstance(try_to_load_from_cache(str(ckpt), MARKER, revision=revision), str):  # works offline
            return True
        return file_exists(str(ckpt), MARKER, revision=revision)
    except Exception as e:
        log.warning("could not check whether %s is a Vev checkpoint (%s: %s); loading it as a base model",
                    ckpt, type(e).__name__, e)
        return False


def base_source(ckpt: Path, meta: dict[str, Any]) -> str:
    """Where the backbone and processor come from: the base model id, or the checkpoint itself for merged releases."""
    return str(ckpt) if meta["base"] == "." else meta["base"]


def load_checkpoint(ckpt: str | Path, dtype: torch.dtype, device: torch.device, attn_implementation: str | None = None,
                    revision: str | None = None) -> tuple[VevModel, dict[str, Any], Path]:
    """A checkpoint written by VevModel.save (base id + adapter/) or a merged release (base "."), in eval mode."""
    path = resolve_checkpoint(ckpt, revision)
    meta = json.loads((path / MARKER).read_text(encoding="utf-8"))
    model = VevModel(base_source(path, meta), dtype=dtype, attn_implementation=attn_implementation)
    if (path / "adapter").exists():
        from peft import PeftModel

        model.backbone = PeftModel.from_pretrained(model.backbone, str(path / "adapter")).merge_and_unload()
    model.to(device).eval()
    return model, meta, path


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
    """Copy of a prefilled hybrid cache that n questions (batch rows) continue independently. Attention K/V are only
    ever replaced via torch.cat, never written in place, so they are shared (expanded to n rows); DeltaNet conv and
    recurrent states are updated in place, so they are copied."""
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
                    states[i] = t.clone() if n == 1 else t.repeat(n, *([1] * (t.dim() - 1)))
        if n > 1 and isinstance(getattr(layer, "keys", None), torch.Tensor):
            lay.keys = layer.keys.expand(n, *layer.keys.shape[1:])
            lay.values = layer.values.expand(n, *layer.values.shape[1:])
        new.layers.append(lay)
    return new


def vision_once_per_image(forward):
    """Wraps a vision tower's forward. When the second half of a forward's images repeats the first half (same grids,
    same pixel rows: the image-read-twice layout), it encodes the first half and repeats the outputs. The tower attends
    only within each image (cu_seqlens) and its other layers act patch by patch, so this equals encoding every copy."""
    def wrapped(hidden_states, grid_thw=None, **kwargs):
        n = 0 if grid_thw is None else int(grid_thw.shape[0])
        if n >= 2 and n % 2 == 0 and torch.equal(grid_thw[: n // 2], grid_thw[n // 2:]):
            m = int(grid_thw[: n // 2].prod(-1).sum())
            if hidden_states.shape[0] == 2 * m and torch.equal(hidden_states[:m], hidden_states[m:]):
                out = forward(hidden_states[:m], grid_thw=grid_thw[: n // 2], **kwargs)
                if any(v is not None for k, v in out.items() if k not in ("last_hidden_state", "pooler_output")):
                    return forward(hidden_states, grid_thw=grid_thw, **kwargs)  # outputs this wrapper does not know
                out.last_hidden_state = torch.cat([out.last_hidden_state] * 2)
                out.pooler_output = torch.cat([out.pooler_output] * 2)
                return out
        return forward(hidden_states, grid_thw=grid_thw, **kwargs)
    return wrapped


class CheckpointEngine:
    """Serving backend for a Vev checkpoint, one row per question.

    A single question runs alone (for states of at least prefix_min_tokens tokens, from a prefilled state cache).
    Several questions run as one batch. When whole rows would recompute a long state many times over, and always for
    states with images, the state is prefilled once and the question suffixes continue from copies of that cache as a
    right-padded batch; otherwise the whole rows are right-padded into one batch. Batches are split into chunks that fit the free GPU memory. Rows are ordered by (length, name), so the
    order of the questions in a request does not change what is computed. With images, the image processor and the
    vision tower run once per distinct image, also when the image-read-twice layout lists every image twice."""

    def __init__(self, ckpt: str, dtype: str = "bf16", device: str = "cuda", attn_implementation: str | None = None,
                 prefix_min_tokens: int = 4096, revision: str | None = None):
        from transformers import AutoProcessor

        windows_sdpa_workaround()
        self.device = torch.device(device)
        self.model, self.meta, path = load_checkpoint(ckpt, DTYPES[dtype], self.device, attn_implementation, revision)
        # Reading layout, from the checkpoint's vev.json; the VEV_* environment variables override it (experiments).
        mode = os.environ.get("VEV_BIDIR", self.meta.get("bidir", ""))
        if self.meta.get("bidir") and mode != self.meta["bidir"]:
            raise RuntimeError(f"{ckpt} was trained with bidir={self.meta['bidir']}; serve it with the same setting")
        if mode and not self.meta.get("bidir"):
            log.warning("bidir=%s on %s, which was trained without it", mode, ckpt)
        bidir.configure(mode)
        self.read_twice = os.environ.get("VEV_READ_TWICE", self.meta.get("read_twice", ""))
        self.read_twice_max = int(os.environ.get("VEV_READ_TWICE_MAX", self.meta.get("read_twice_max", 3000)))
        self.image_read = os.environ.get("VEV_IMAGE_READ", self.meta.get("image_read", ""))
        self.max_image_pixels = int(os.environ.get("VEV_MAX_IMAGE_PIXELS",
                                                   self.meta.get("max_image_pixels", 1024 * 1024)))
        self.base_id = self.meta.get("base_model") or self.meta["base"]
        self.processor = AutoProcessor.from_pretrained(base_source(path, self.meta))
        self.tok = self.processor.tokenizer
        self.merge = self.processor.image_processor.merge_size
        self.labels = LabelTokens(self.tok)
        self.autocast_dtype = DTYPES[dtype] if dtype != "fp32" else None
        self.prefix_min_tokens = prefix_min_tokens
        # several questions: share the state prefix when whole rows would recompute the state at least this many times
        # over, in tokens ((questions - 1) x state tokens); otherwise batch whole rows
        self.share_min_saved_tokens = 2048
        self.max_batch_tokens = 32768  # padded tokens per forward pass
        self.cache_memory_fraction = 0.5  # of the free GPU memory, for the per-row copies of the state cache
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        vision = getattr(self.model.backbone, "visual", None)
        if vision is not None:
            vision.forward = vision_once_per_image(vision.forward)

    def _autocast(self):
        on = self.autocast_dtype is not None and self.device.type == "cuda"
        return torch.autocast("cuda", dtype=self.autocast_dtype or torch.bfloat16, enabled=on)

    @torch.inference_mode()
    def _logits(self, row: dict[str, Any]) -> torch.Tensor:
        batch, decide = collate_rows([row], self.tok.pad_token_id)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self._autocast(), bidir.for_rows([row]):
            z = self.model(batch, decide, [row["answer_ids"]])[0]
        return z.float().cpu()

    def _prefix_len(self, segments: list, rows: list[dict[str, Any]], image_mode: str = "",
                    probe_instructions: str | None = None, read_twice: str | None = None) -> int:
        """Length of the shared prefix, taken from state-only probe rows. Falls back to the rows' common prefix if
        some row tokenizes the state boundary differently."""
        if "state_len" in rows[0]:  # VEV_BIDIR=state: computed by this method in run() for the same rows
            return rows[0]["state_len"]
        grid = rows[0].get("image_grid_thw")
        counts = [] if grid is None else [int(g.prod()) // self.merge**2 for g in grid]

        def expand(ids: list[int]) -> list[int]:  # the processor's image-pad expansion
            it, out = iter(counts), []
            for t in ids:
                out.extend([t] * next(it) if t == self.image_pad_id else [t])
            return out

        pqs = PROBE_QUESTIONS if probe_instructions is None else [dict(q, instructions=probe_instructions)
                                                                  for q in PROBE_QUESTIONS]
        read_twice = self.read_twice if read_twice is None else read_twice
        probes = [expand(self.tok(render_row(self.processor, segments, q, self.labels, image_mode, read_twice,
                                             self.read_twice_max), add_special_tokens=False)["input_ids"]) for q in pqs]
        ids = [r["input_ids"].tolist() for r in rows]
        n = common_prefix_len(probes)
        if any(x[:n] != probes[0][:n] for x in ids):
            n = common_prefix_len(probes + ids)
        mm = rows[0].get("mm_token_type_ids")
        if not image_mode and mm is not None and bool((mm[n:] != 0).any()):
            raise RuntimeError("image tokens outside the shared state prefix")
        if any(n > r["decide"] for r in rows):
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
        zs = []
        with self._autocast():
            cache = DynamicCache(config=bb.config)
            with bidir.state_blocks([n]):
                bb(**{k: v.to(dev) for k, v in pre.items()}, past_key_values=cache, use_cache=True)
            for i, r in enumerate(rows):
                x = r["input_ids"][n:]
                mask = torch.ones(1, n + x.shape[0], dtype=torch.long, device=dev)
                h = bb(input_ids=x[None].to(dev), attention_mask=mask, position_ids=pos[i][:, None, n:].to(dev),
                       past_key_values=fork_cache(cache), use_cache=True).last_hidden_state
                zs += self.model.readout(h, [r["decide"] - n], [r["answer_ids"]])
        return [z.float().cpu() for z in zs]

    def run(self, state: Any, questions: dict[str, dict[str, Any]], max_images: int = 8, max_pixels: int | None = None,
            allow_remote: bool = False, max_state_tokens: int = 32768, max_request_tokens: int = 65536) -> Result:
        t0 = time.perf_counter()
        max_pixels = max_pixels or self.max_image_pixels
        segments, images = serialize_state(state, max_images=max_images, max_pixels=max_pixels, allow_remote=allow_remote)
        names = list(questions)

        def encode(image_mode: str, read_twice: str) -> dict[str, dict[str, Any]]:
            texts = [render_row(self.processor, segments, questions[k], self.labels, image_mode, read_twice,
                                self.read_twice_max) for k in names]
            encoded = encode_rows(self.processor, texts, images * (2 if image_mode == "twice" else 1),
                                  device=self.device if self.device.type == "cuda" else None)
            return dict(zip(names, encoded))

        def lengths(rows: dict[str, dict[str, Any]]) -> tuple[int, int, dict[str, int], str]:
            state_text, state_image = count_state_tokens(self.tok, segments, next(iter(rows.values())), self.merge)
            row_lens = {name: int(r["input_ids"].shape[0]) for name, r in rows.items()}
            longest, total = max(row_lens.values()), sum(row_lens.values()) - (len(row_lens) - 1) * (state_text + state_image)
            error = (f"state + longest question is {longest} tokens (max {max_state_tokens})" if longest > max_state_tokens
                     else f"request is {total} tokens (max {max_request_tokens})" if total > max_request_tokens else "")
            return state_text, state_image, row_lens, error

        # The limits are on the caller's request: one that is over them only because the layout repeats the state (and
        # the instructions) is read once, in the v0.1 layout.
        image_mode = self.image_read if images and len(questions) == 1 else ""
        read_twice = self.read_twice
        rows = encode(image_mode, read_twice)
        state_text, state_image, row_lens, error = lengths(rows)
        if error and (image_mode or read_twice):
            image_mode = read_twice = ""
            rows = encode(image_mode, read_twice)
            state_text, state_image, row_lens, error = lengths(rows)
        if error:
            raise InvalidRequest(400, "invalid_request_error", error)
        for name, q in questions.items():
            rows[name]["answer_ids"] = self.labels.answer_ids(q)
        state_tokens = state_text + state_image
        question_tokens = {name: n - state_tokens for name, n in row_lens.items()}
        input_tokens = state_tokens + sum(question_tokens.values())
        if bidir.ENABLED or read_twice != self.read_twice:
            # VEV_BIDIR=state: one state-block length per request, used by every path below; also the prefix of a
            # request read once by the fallback above, whose probes must be rendered the same way
            n_state = self._prefix_len(segments, list(rows.values()), image_mode,
                                       (questions[names[0]].get("instructions") or "").strip()
                                       if image_mode == "qfirst" else None, read_twice)
            for r in rows.values():
                r["state_len"] = n_state
        t1 = time.perf_counter()
        if len(rows) == 1 and state_tokens >= self.prefix_min_tokens and not image_mode:
            logits = dict(zip(rows, self._logits_shared(segments, list(rows.values()))))
        elif len(rows) == 1:
            logits = {name: self._logits(r) for name, r in rows.items()}
        elif images or (len(rows) - 1) * state_tokens >= self.share_min_saved_tokens:
            logits = self._logits_shared_batched(segments, rows)
        else:
            logits = self._logits_rows_batched(rows)
        answers = {name: make_answer(q, torch.softmax(logits[name], dim=0).tolist()) for name, q in questions.items()}
        t2 = time.perf_counter()
        return Result(
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=count_output_tokens(self.tok, answers),
            extensions={
                "logits": {name: z.tolist() for name, z in logits.items()},
                "timing": {"prefill_ms": round((t1 - t0) * 1000, 2), "questions_ms": round((t2 - t1) * 1000, 2),
                           "total_ms": round((t2 - t0) * 1000, 2)},
                "tokens": {"state_text": state_text, "state_image": state_image, "questions": question_tokens},
            },
        )

    @staticmethod
    def _order(rows: dict[str, dict[str, Any]], start: int = 0) -> list[str]:
        return sorted(rows, key=lambda k: (int(rows[k]["input_ids"].shape[0]) - start, k))

    @staticmethod
    def _chunks(names: list[str], length: dict[str, int], fits) -> list[list[str]]:
        """Consecutive chunks of names (sorted by length), each as large as fits(count, max_length) allows."""
        out: list[list[str]] = []
        for k in names:
            if out and fits(len(out[-1]) + 1, max(length[x] for x in out[-1] + [k])):
                out[-1].append(k)
            else:
                out.append([k])
        return out

    def _readout(self, h: torch.Tensor, chunk: list[str], decide: list[int], rows) -> dict[str, torch.Tensor]:
        ids = [rows[k]["answer_ids"] for k in chunk]
        zs = self.model.readout(h, decide, ids)
        flat = torch.cat([z.float() for z in zs]).cpu()
        return dict(zip(chunk, flat.split([len(i) for i in ids])))

    @torch.inference_mode()
    def _logits_rows_batched(self, rows: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
        order = self._order(rows)
        length = {k: int(rows[k]["input_ids"].shape[0]) for k in order}
        out: dict[str, torch.Tensor] = {}
        for chunk in self._chunks(order, length, lambda b, s: b * s <= self.max_batch_tokens):
            batch, decide = collate_rows([rows[k] for k in chunk], self.tok.pad_token_id)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self._autocast(), bidir.for_rows([rows[k] for k in chunk]):
                h = self.model.hidden(batch)
                out.update(self._readout(h, chunk, decide, rows))
        return out

    @staticmethod
    def _cache_bytes(cache) -> tuple[int, int]:
        """(attention K/V bytes, DeltaNet conv + recurrent state bytes) of a batch-1 cache."""
        kv = state = 0
        for layer in cache.layers:
            for x in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                if isinstance(x, torch.Tensor):
                    kv += x.numel() * x.element_size()
            for states in (getattr(layer, "conv_states", None), getattr(layer, "recurrent_states", None)):
                for x in (states or {}).values():
                    if isinstance(x, torch.Tensor):
                        state += x.numel() * x.element_size()
        return kv, state

    @torch.inference_mode()
    def _logits_shared_batched(self, segments: list, rows: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
        from transformers import DynamicCache

        bb, dev = self.model.backbone, self.device
        n = self._prefix_len(segments, list(rows.values()))
        order = self._order(rows)
        pos = {k: self._positions(rows[k]) for k in order}
        first = rows[order[0]]
        pre = {"input_ids": first["input_ids"][None, :n], "attention_mask": torch.ones(1, n, dtype=torch.long),
               "position_ids": pos[order[0]][:, None, :n]}
        if first.get("pixel_values") is not None:
            pre["pixel_values"], pre["image_grid_thw"] = first["pixel_values"], first["image_grid_thw"]
        out: dict[str, torch.Tensor] = {}
        with self._autocast():
            cache = DynamicCache(config=bb.config)
            with bidir.state_blocks([n]):
                bb(**{k: v.to(dev) for k, v in pre.items()}, past_key_values=cache, use_cache=True)
            kv, state = self._cache_bytes(cache)
            per_token = kv / max(1, n)
            budget = (torch.cuda.mem_get_info(dev)[0] * self.cache_memory_fraction if dev.type == "cuda" else float("inf"))
            length = {k: int(rows[k]["input_ids"].shape[0]) - n for k in order}

            def fits(b: int, s: int) -> bool:
                return b * s <= self.max_batch_tokens and b * (per_token * (n + s) + state) <= budget

            for chunk in self._chunks(order, length, fits):
                b, s = len(chunk), max(length[k] for k in chunk)
                dims = pos[chunk[0]].shape[0]
                ids = torch.full((b, s), self.tok.pad_token_id, dtype=first["input_ids"].dtype)
                mask = torch.zeros((b, n + s), dtype=torch.long)
                p = torch.zeros((dims, b, s), dtype=pos[chunk[0]].dtype)
                for i, k in enumerate(chunk):
                    x = rows[k]["input_ids"][n:]
                    ids[i, : x.shape[0]] = x
                    mask[i, : n + x.shape[0]] = 1
                    p[:, i, : x.shape[0]] = pos[k][:, n:]
                h = bb(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=p.to(dev),
                       past_key_values=fork_cache(cache, b), use_cache=True).last_hidden_state
                out.update(self._readout(h, chunk, [rows[k]["decide"] - n for k in chunk], rows))
        return out

    def warmup(self) -> None:
        """vev.warmup: requests in the board's shapes, so Triton compiles and autotunes before the first request."""
        from vev import warmup

        warmup.run(self)
