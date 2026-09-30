"""Write the two release forms of a trained checkpoint.

    python -m train.export_release --ckpt runs/vev-4b/export --name vev-4b --out release/

release/vev-4b-lora/   adapter/ + vev.json               (loads the base model from Hugging Face)
release/vev-4b/        full merged model + processor + vev.json with base "."

vev.json keeps the base model, the version and the training arguments, with local paths reduced to their last
component."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from vev.model import MARKER

PATH_KEYS = ("build", "out", "anchor_build", "resume")


def clean_args(args: dict) -> dict:
    return {k: Path(v).name if k in PATH_KEYS and isinstance(v, str) and v else v for k, v in args.items()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint directory written by train.train")
    ap.add_argument("--name", required=True, help="release name, e.g. vev-4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--skip-merged", action="store_true")
    a = ap.parse_args(argv)
    ckpt, out = Path(a.ckpt), Path(a.out)
    meta = json.loads((ckpt / MARKER).read_text(encoding="utf-8"))
    base = meta["base"]
    rel = {"base_model": base, "vev_version": a.version, "train_args": clean_args(meta.get("train_args", {}))}

    lora = out / f"{a.name}-lora"
    lora.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpt / "adapter", lora / "adapter", dirs_exist_ok=True)
    (lora / "adapter" / "README.md").unlink(missing_ok=True)  # peft's empty model-card template
    cfg = lora / "adapter" / "adapter_config.json"
    c = json.loads(cfg.read_text(encoding="utf-8"))
    c["base_model_name_or_path"] = base
    cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
    (lora / MARKER).write_text(json.dumps({**rel, "base": base}, indent=2), encoding="utf-8")
    print(f"[export] {lora}")

    if a.skip_merged:
        return
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    full = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    # the adapter was trained on the backbone (full.model); wrap that same module so the target names line up
    full.model = PeftModel.from_pretrained(full.model, str(ckpt / "adapter")).merge_and_unload()
    merged = out / a.name
    full.save_pretrained(str(merged), safe_serialization=True)
    AutoProcessor.from_pretrained(base).save_pretrained(str(merged))
    (merged / MARKER).write_text(json.dumps({**rel, "base": "."}, indent=2), encoding="utf-8")
    print(f"[export] {merged}")


if __name__ == "__main__":
    main()
