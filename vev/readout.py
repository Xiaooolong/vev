"""Zero-shot readout: next-token logits of a Qwen3.5 model over the allowed answer tokens, one forward per question.
Used to serve a base model without fine-tuning; vev.model serves trained checkpoints with the same prompt."""

from __future__ import annotations

import base64
import io
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.integrations import sdpa_attention

from vev.state import ImageSlot, InvalidRequest, render_desc, serialize_state
from vev.tokens import count_output_tokens, count_state_tokens

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

SYSTEM_PROMPT = (
    "You are a careful judge. Read the state, then answer the question about it. "
    "Reply with only the answer token, nothing else."
)


def windows_sdpa_workaround() -> None:
    # Windows torch has no flash kernel and SDPA with enable_gqa=True has no memory-efficient one, so it falls back to
    # the math kernel (full L x L scores, ~3.7 GB at 7k tokens); expanding KV heads avoids that.
    if sys.platform == "win32":
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False


def choice_labels(tok, k: int) -> list[str]:
    """A..Z, then two-letter AA, AB, ... keeping only strings the tokenizer encodes as a single token."""
    two = [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    out = []
    for s in list(string.ascii_uppercase) + two:
        if len(tok.encode(s, add_special_tokens=False)) == 1:
            out.append(s)
            if len(out) == k:
                break
    return out


def confidence(p_max: float, k: int) -> float:
    return 1.0 if k == 1 else (p_max - 1.0 / k) / (1.0 - 1.0 / k)


class LabelTokens:
    """Single-token answer strings of the tokenizer: letters for choice (A..Z, AA..), Yes/No, level digits."""

    def __init__(self, tok):
        self.letters = choice_labels(tok, MAX_CHOICE_OPTIONS)
        if len(self.letters) < MAX_CHOICE_OPTIONS:
            raise RuntimeError(f"tokenizer has only {len(self.letters)} single-token choice labels (< {MAX_CHOICE_OPTIONS})")
        self.ids: dict[str, int] = {}
        for s in self.letters + ["Yes", "No"] + [str(i) for i in range(MAX_SCORE_LEVELS)]:
            enc = tok.encode(s, add_special_tokens=False)
            if len(enc) != 1:
                raise RuntimeError(f"answer string {s!r} is not a single token: {enc}")
            self.ids[s] = enc[0]

    def answer_strings(self, q: dict[str, Any]) -> list[str]:
        if q["type"] == "choice":
            return self.letters[: len(q["criteria"])]
        if q["type"] == "score":
            return [str(i) for i in range(len(q["criteria"]))]
        return ["Yes", "No"]

    def answer_ids(self, q: dict[str, Any]) -> list[int]:
        return [self.ids[s] for s in self.answer_strings(q)]


def answer_format(q: dict[str, Any], letters: list[str]) -> str:
    """Options and answer instructions for one question."""
    crit = q.get("criteria")
    if q["type"] == "choice":
        lines = []
        for letter, (label, desc) in zip(letters, crit.items()):
            d = render_desc(desc)
            lines.append(f"{letter}. {label}" + (f" - {d}" if d else ""))
        return "Options:\n" + "\n".join(lines) + "\n\nAnswer with the letter of the single best option."
    if q["type"] == "score":
        n = len(crit)
        lines = [f"{i}. {render_desc(d)}".rstrip() for i, d in enumerate(crit)]
        return f"Scale, from lowest (0) to highest ({n - 1}):\n" + "\n".join(lines) + "\n\nAnswer with the level number only."
    crit = crit or {}
    parts = ["Answer Yes or No."]
    if crit.get("true") is not None:
        parts.append(f"Yes means: {render_desc(crit['true'])}")
    if crit.get("false") is not None:
        parts.append(f"No means: {render_desc(crit['false'])}")
    return "\n".join(parts)


def build_messages(segments: list, q: dict[str, Any], fmt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": "State:\n"}]
    for seg in segments:
        content.append({"type": "image"} if isinstance(seg, ImageSlot) else {"type": "text", "text": seg})
    instructions = (q.get("instructions") or "").strip()
    tail = "\n\n" + (f"Question: {instructions}\n\n" if instructions else "") + fmt
    content.append({"type": "text", "text": tail})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


@dataclass
class Result:
    answers: dict[str, Any]
    input_tokens: int
    output_tokens: int
    extensions: dict[str, Any] = field(default_factory=dict)


def make_answer(q: dict[str, Any], probs: list[float]) -> dict[str, Any]:
    if q["type"] == "noul":
        return {"type": "noul", "noul": probs[0]}
    k = len(probs)
    best = max(range(k), key=lambda i: (probs[i], -i))
    if q["type"] == "choice":
        labels = list(q["criteria"].keys())
        return {"type": "choice", "choice": labels[best], "confidence": confidence(probs[best], k),
                "probabilities": dict(zip(labels, probs))}
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)), "confidence": confidence(probs[best], k),
            "probabilities": {str(i): p for i, p in enumerate(probs)},
            "legend": {str(i): d for i, d in enumerate(q["criteria"])}}


def warmup(engine) -> None:
    """One image request and one text request with all three question types."""
    img = Image.new("RGB", (512, 512), (200, 200, 200))
    ImageDraw.Draw(img).ellipse([128, 128, 384, 384], fill=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    questions = {
        "n": {"type": "noul", "instructions": "Is there a red circle?", "criteria": None},
        "c": {"type": "choice", "instructions": "Shape?", "criteria": {"circle": None, "square": None}},
        "s": {"type": "score", "instructions": "How red?", "criteria": ["not", "somewhat", "very"]},
    }
    engine.run({"note": "warmup", "image": {"url": url}}, questions)
    engine.run("warmup", questions)


class Engine:
    def __init__(self, model_id: str, dtype: str = "bf16", device: str = "cuda", revision: str | None = None):
        windows_sdpa_workaround()
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.tok = self.processor.tokenizer
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPES[dtype], revision=revision)
        self.model = self.model.to(self.device).eval()
        self.merge = self.processor.image_processor.merge_size
        self.labels = LabelTokens(self.tok)

    def _encode(self, messages: list[dict[str, Any]], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return self.processor(text=[text], images=images or None, return_tensors="pt")

    @torch.inference_mode()
    def _forward(self, enc: dict[str, torch.Tensor], answer_ids: list[int]) -> torch.Tensor:
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**inputs, logits_to_keep=1, use_cache=False)
        return out.logits[0, -1, answer_ids].float().cpu()

    def run(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        max_images: int = 8,
        max_pixels: int = 1024 * 1024,
        allow_remote: bool = False,
        max_state_tokens: int = 32768,
        max_request_tokens: int = 65536,
    ) -> Result:
        t0 = time.perf_counter()
        segments, images = serialize_state(state, max_images=max_images, max_pixels=max_pixels, allow_remote=allow_remote)
        rows = {}
        for name, q in questions.items():
            fmt = answer_format(q, self.labels.letters)
            rows[name] = self._encode(build_messages(segments, q, fmt), images)
        state_text, state_image = count_state_tokens(self.tok, segments, next(iter(rows.values())), self.merge)
        state_tokens = state_text + state_image
        row_lens = {name: int(enc["input_ids"].shape[1]) for name, enc in rows.items()}
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
        for name, q in questions.items():
            ids = self.labels.answer_ids(q)
            if q["type"] != "noul" and len(ids) == 1:
                probs = torch.ones(1)
                raw_logits[name] = []
            else:
                logits = self._forward(rows[name], ids)
                probs = torch.softmax(logits, dim=0)
                raw_logits[name] = logits.tolist()
            answers[name] = make_answer(q, probs.tolist())
        t2 = time.perf_counter()

        return Result(
            answers=answers,
            input_tokens=input_tokens,
            output_tokens=count_output_tokens(self.tok, answers),
            extensions={
                "logits": raw_logits,
                "timing": {
                    "prefill_ms": round((t1 - t0) * 1000, 2),
                    "questions_ms": round((t2 - t1) * 1000, 2),
                    "total_ms": round((t2 - t0) * 1000, 2),
                },
                "tokens": {"state_text": state_text, "state_image": state_image, "questions": question_tokens},
            },
        )

    def warmup(self) -> None:
        warmup(self)
