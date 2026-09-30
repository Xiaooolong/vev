"""Write the two release forms of a trained checkpoint.

    python -m train.export_release --ckpt runs/.../export --name vev-4b --out release/

release/vev-4b-lora/   adapter/ + head.pt + vev_pointer.json   (loads the base model from Hugging Face)
release/vev-4b/        full merged model + processor + head.pt + vev_pointer.json with base "."

vev_pointer.json keeps only what loading needs plus the training recipe with local paths reduced to their last
component, so no machine paths leave the training cluster."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

KEEP = ("dp", "prior", "temperatures")
PATH_KEYS = ("build", "out", "anchor_build", "resume")


def clean_args(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if k in PATH_KEYS and isinstance(v, str) and v:
            v = Path(v).name
        out[k] = v
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint directory written by train.train (vev_pointer.json)")
    ap.add_argument("--name", required=True, help="release name, e.g. vev-4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--skip-merged", action="store_true")
    a = ap.parse_args(argv)
    ckpt, out = Path(a.ckpt), Path(a.out)
    meta = json.loads((ckpt / "vev_pointer.json").read_text(encoding="utf-8"))
    base = meta["base"]
    rel = {k: meta[k] for k in KEEP if k in meta}
    rel.update({"base_model": base, "vev_version": a.version})
    if "train_args" in meta:
        rel["train_args"] = clean_args(meta["train_args"])

    lora = out / f"{a.name}-lora"
    lora.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ckpt / "adapter", lora / "adapter", dirs_exist_ok=True)
    cfg = lora / "adapter" / "adapter_config.json"
    c = json.loads(cfg.read_text(encoding="utf-8"))
    c["base_model_name_or_path"] = base
    cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
    shutil.copy2(ckpt / "head.pt", lora / "head.pt")
    (lora / "vev_pointer.json").write_text(json.dumps({**rel, "base": base}, indent=2), encoding="utf-8")
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
    shutil.copy2(ckpt / "head.pt", merged / "head.pt")
    (merged / "vev_pointer.json").write_text(json.dumps({**rel, "base": "."}, indent=2), encoding="utf-8")
    print(f"[export] {merged}")


if __name__ == "__main__":
    main()
