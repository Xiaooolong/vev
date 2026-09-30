"""Write a prior-only checkpoint: the base model, no adapter, a zero-output head, prior mode on.

    python -m train.make_prior_ckpt --base Qwen/Qwen3.5-4B --out runs/prior-only-4b

Serving this directory must reproduce the zero-shot label-token readout; check it before a prior-mode
training run. It is also what a prior-mode training run starts from."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vev.pointer import SPECIAL, SYSTEM_PROMPT, PointerHead


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dp", type=int, default=256)
    a = ap.parse_args(argv)
    from transformers import AutoConfig

    hidden = int(AutoConfig.from_pretrained(a.base).get_text_config().hidden_size)
    head = PointerHead(hidden, a.dp).float()
    head.zero_output()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), out / "head.pt")
    (out / "vev_pointer.json").write_text(json.dumps({
        "base": a.base, "dp": a.dp, "lora_r": 0, "prior": True, "special": SPECIAL, "system_prompt": SYSTEM_PROMPT,
        "temperatures": {}, "note": "prior-only: zero head, no adapter; equals the zero-shot label-token readout"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[prior-ckpt] {out}")


if __name__ == "__main__":
    main()
