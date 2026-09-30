"""LoRA fine-tuning of the answer-token readout on a build directory (see TRAINING.md for the release recipe).

    python -m train.train --build data/build/v1-research --base Qwen/Qwen3.5-4B --out runs/vev-4b \
        --limit 100000 --max-steps 2500 --anchor-kl 0.3 --anchor-build data/build/anchor-v3 --anchor-weight 2

Writes <out>/metrics.jsonl (train + validation curves), <out>/ckpt/step-N/ (resumable), <out>/export/ (servable)."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.data import RowDataset, micro_batches
from vev.model import VevModel
from vev.readout import DTYPES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="training records (grids) to use; 0 = all")
    ap.add_argument("--sources", default=None, help="comma list of sources to keep")
    ap.add_argument("--max-steps", type=int, required=True, help="optimizer steps; data cycles over epochs as needed")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=0, help="0 = 2r")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--anchor-kl", type=float, default=0.0,
                    help="weight of KL(base || adapted) on training rows; the base is the same model with adapters disabled")
    ap.add_argument("--anchor-build", default=None, help="build dir of unlabeled anchor rows: KL to the base only, no CE")
    ap.add_argument("--anchor-weight", type=float, default=1.0, help="KL weight on anchor rows")
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-q", type=int, default=4)
    ap.add_argument("--max-row-tokens", type=int, default=4096)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--brier", type=float, default=0.1)
    ap.add_argument("--score-w", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-pct", type=float, default=0.1)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-gradient-checkpointing", action="store_true")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--attn", default=None, help="attn_implementation override")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-rows", type=int, default=1000)
    ap.add_argument("--eval-records", type=int, default=2000, help="validation records the eval rows are drawn from")
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help="ckpt/step-N directory to continue from")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args(argv)


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


def row_metrics(z: torch.Tensor, t: torch.Tensor, qtype: str) -> dict[str, float]:
    p = torch.softmax(z.float(), -1)
    if qtype == "noul":
        acc = float((p[0] > 0.5) == (t[0] > 0.5))
    else:
        acc = float(int(p.argmax()) == int(t.argmax()))
    return {"acc": acc, "brier": float((p - t).square().sum()), "nll": float(-(t * torch.log(p.clamp_min(1e-12))).sum())}


AUX = ("anchor", "anchorset")


class Meter:
    def __init__(self):
        self.sums: dict[str, dict[str, float]] = {}
        self.n: dict[str, int] = {}

    def add(self, qtype: str, m: dict[str, float]) -> None:
        s = self.sums.setdefault(qtype, {})
        for k, v in m.items():
            s[k] = s.get(k, 0.0) + v
        self.n[qtype] = self.n.get(qtype, 0) + 1

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        tot: dict[str, float] = {}
        n_all = sum(n for k, n in self.n.items() if k not in AUX)
        for qtype, s in self.sums.items():
            out[qtype] = {k: round(v / self.n[qtype], 4) for k, v in s.items()}
            out[qtype]["n"] = self.n[qtype]
            if qtype in AUX:  # KL bookkeeping, not questions: keep them out of the per-question totals
                continue
            for k, v in s.items():
                tot[k] = tot.get(k, 0.0) + v
        out["all"] = {k: round(v / max(1, n_all), 4) for k, v in tot.items()}
        out["all"]["n"] = n_all
        return out


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def base_kl(model: VevModel, dev_batch, decide, ids, logits, autocast) -> torch.Tensor:
    """KL(base || adapted) of the answer distribution; the base is the same rows with adapters disabled, no grad."""
    with torch.no_grad(), model.backbone.disable_adapter(), autocast:
        base = model(dev_batch, decide, ids)
    kls = [F.kl_div(torch.log_softmax(z, -1), torch.softmax(b, -1), reduction="sum") for z, b in zip(logits, base)]
    return torch.stack(kls).mean()


def run_batch(model: VevModel, batch, decide, rows, args, device, autocast, meter: Meter) -> torch.Tensor:
    dev_batch = to_device(batch, device)
    ids = [r["answer_ids"] for r in rows]
    with autocast:
        logits = model(dev_batch, decide, ids)
    losses = []
    for z, r in zip(logits, rows):
        t = r["target"].to(device)
        losses.append(row_loss(z, t, r["qtype"], args.brier, args.score_w))
        meter.add(r["qtype"], {**row_metrics(z.detach(), t, r["qtype"]), "loss": float(losses[-1].detach())})
    loss = torch.stack(losses).mean()
    if args.anchor_kl > 0:
        kl = base_kl(model, dev_batch, decide, ids, logits, autocast)
        meter.add("anchor", {"kl": float(kl.detach())})
        loss = loss + args.anchor_kl * kl
    return loss


def anchor_batch(model: VevModel, batch, decide, rows, device, autocast, meter: Meter) -> torch.Tensor:
    """Unlabeled anchor rows: only the KL of the adapted answer distribution to the base; targets are ignored."""
    dev_batch = to_device(batch, device)
    ids = [r["answer_ids"] for r in rows]
    with autocast:
        logits = model(dev_batch, decide, ids)
    kl = base_kl(model, dev_batch, decide, ids, logits, autocast)
    meter.add("anchorset", {"kl": float(kl.detach())})
    return kl


@torch.no_grad()
def evaluate(model: VevModel, args, processor, device, autocast) -> dict[str, Any]:
    model.eval()
    ds = RowDataset(args.build, "validation", processor, limit=args.eval_records, seed=args.seed, max_q=2, augment=False,
                    max_pixels=args.max_pixels, max_row_tokens=args.max_row_tokens)
    loader = DataLoader(ds, batch_size=None, num_workers=min(args.workers, 4))
    meter = Meter()
    seen = 0
    for batch, decide, rows in micro_batches(iter(loader), args.micro_tokens, processor.tokenizer.pad_token_id):
        with autocast:
            logits = model(to_device(batch, device), decide, [r["answer_ids"] for r in rows])
        for z, r in zip(logits, rows):
            t = r["target"].to(device)
            m = row_metrics(z, t, r["qtype"])
            m["loss"] = float(row_loss(z, t, r["qtype"], args.brier, args.score_w))
            meter.add(r["qtype"], m)
        seen += len(rows)
        if seen >= args.eval_rows:
            break
    model.train()
    return meter.summary()


def save_checkpoint(model: VevModel, opt, sched, step: int, epoch: int, out: Path, meta: dict) -> Path:
    d = out / "ckpt" / f"step-{step}"
    model.save(d, {**meta, "step": step})
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step, "epoch": epoch,
                "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "python": random.getstate()}}, d / "train_state.pt")
    return d


def load_checkpoint(model: VevModel, opt, sched, path: Path) -> tuple[int, int]:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model.backbone, load_file(str(path / "adapter" / "adapter_model.safetensors")))
    st = torch.load(path / "train_state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["optimizer"])
    sched.load_state_dict(st["scheduler"])
    torch.set_rng_state(st["rng"]["torch"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    random.setstate(st["rng"]["python"])
    return int(st["step"]), int(st["epoch"])


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoProcessor

    args = parse_args(argv)
    if args.lora_r <= 0:
        raise SystemExit("--lora-r must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    autocast = torch.autocast("cuda", dtype=dtype) if (device.type == "cuda" and dtype != torch.float32) else torch.autocast("cpu", enabled=False)

    processor = AutoProcessor.from_pretrained(args.base)
    processor.tokenizer.padding_side = "right"
    pad_id = processor.tokenizer.pad_token_id
    model = VevModel(args.base, dtype=dtype, lora_r=args.lora_r, lora_alpha=args.lora_alpha or None,
                     lora_dropout=args.lora_dropout, gradient_checkpointing=not args.no_gradient_checkpointing,
                     attn_implementation=args.attn).to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[train] trainable: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.max_steps, pct_start=args.warmup_pct,
                                                anneal_strategy="cos", div_factor=25.0, final_div_factor=100.0)
    meta = {"base": args.base, "lora_r": args.lora_r, "train_args": vars(args)}
    step, epoch = 0, 0
    if args.resume:
        step, epoch = load_checkpoint(model, opt, sched, Path(args.resume))
        print(f"[train] resumed at step {step}, epoch {epoch}", flush=True)

    sources = set(args.sources.split(",")) if args.sources else None
    ds = RowDataset(args.build, "train", processor, limit=args.limit, seed=args.seed, max_q=args.max_q,
                    augment=not args.no_augment, max_pixels=args.max_pixels, max_row_tokens=args.max_row_tokens, sources=sources)
    print(f"[train] {len(ds)} training records; augment={not args.no_augment}", flush=True)
    anchor_ds = None
    if args.anchor_build:
        anchor_ds = RowDataset(args.anchor_build, "train", processor, seed=args.seed, max_q=args.max_q, augment=False,
                               max_pixels=args.max_pixels, max_row_tokens=args.max_row_tokens)
        print(f"[train] {len(anchor_ds)} anchor records (KL only, weight {args.anchor_weight})", flush=True)
    metrics_path = out / "metrics.jsonl"

    def log(rec: dict[str, Any]) -> None:
        rec["time"] = time.time()
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def new_epoch_iter(ep: int):
        ds.set_epoch(ep)
        loader = DataLoader(ds, batch_size=None, num_workers=args.workers, prefetch_factor=4 if args.workers else None,
                            persistent_workers=False)
        return micro_batches(iter(loader), args.micro_tokens, pad_id)

    if step == 0:
        ev = evaluate(model, args, processor, device, autocast)
        log({"kind": "eval", "step": 0, "epoch": epoch, **ev})
        print(f"[eval] step 0 {json.dumps(ev['all'])}", flush=True)

    mb_iter = new_epoch_iter(epoch)
    anchor_epoch = 0

    def new_anchor_iter(ep: int):
        anchor_ds.set_epoch(ep)
        return micro_batches(iter(DataLoader(anchor_ds, batch_size=None, num_workers=min(2, args.workers))), args.micro_tokens, pad_id)

    anchor_iter = new_anchor_iter(anchor_epoch) if anchor_ds is not None else None
    meter = Meter()
    tokens_window, rows_window, t_window = 0, 0, time.perf_counter()
    while step < args.max_steps:
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            try:
                batch, decide, rows = next(mb_iter)
            except StopIteration:
                epoch += 1
                print(f"[train] epoch {epoch} begins at step {step}; skipped so far {ds.skipped}", flush=True)
                mb_iter = new_epoch_iter(epoch)
                batch, decide, rows = next(mb_iter)
            loss = run_batch(model, batch, decide, rows, args, device, autocast, meter)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            (loss / args.accum).backward()
            tokens_window += int(batch["attention_mask"].sum())
            rows_window += len(rows)
            if anchor_iter is not None:
                try:
                    abatch, adecide, arows = next(anchor_iter)
                except StopIteration:
                    anchor_epoch += 1
                    anchor_iter = new_anchor_iter(anchor_epoch)
                    abatch, adecide, arows = next(anchor_iter)
                akl = anchor_batch(model, abatch, adecide, arows, device, autocast, meter)
                if not torch.isfinite(akl):
                    raise RuntimeError(f"non-finite anchor KL at step {step}")
                (args.anchor_weight * akl / args.accum).backward()
                tokens_window += int(abatch["attention_mask"].sum())
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == args.max_steps:
            dt = time.perf_counter() - t_window
            s = meter.summary()
            rec = {"kind": "train", "step": step, "epoch": epoch, "lr": sched.get_last_lr(), "tokens_per_s": round(tokens_window / dt, 1),
                   "rows_per_step": round(rows_window / args.log_every, 1), "step_s": round(dt / args.log_every, 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None, **s}
            log(rec)
            print(f"[train] step {step} loss {s['all'].get('loss')} acc {s['all'].get('acc')} brier {s['all'].get('brier')} "
                  f"tok/s {rec['tokens_per_s']} step_s {rec['step_s']} peak {rec['peak_gb']}GB lr {rec['lr'][0]:.2e}", flush=True)
            meter = Meter()
            tokens_window, rows_window, t_window = 0, 0, time.perf_counter()
        if step % args.eval_every == 0 and step < args.max_steps:
            ev = evaluate(model, args, processor, device, autocast)
            log({"kind": "eval", "step": step, "epoch": epoch, **ev})
            print(f"[eval] step {step} {json.dumps(ev['all'])} | " + " ".join(f"{k}:{v.get('acc')}" for k, v in ev.items() if k != "all"), flush=True)
        if step % args.save_every == 0 and step < args.max_steps:
            d = save_checkpoint(model, opt, sched, step, epoch, out, meta)
            print(f"[train] saved {d}", flush=True)

    ev = evaluate(model, args, processor, device, autocast)
    log({"kind": "eval", "step": step, "epoch": epoch, "final": True, **ev})
    print(f"[eval] final {json.dumps(ev, ensure_ascii=False)}", flush=True)
    save_checkpoint(model, opt, sched, step, epoch, out, meta)
    model.save(out / "export", {**meta, "step": step, "final_eval": ev})
    (out / "summary.json").write_text(json.dumps({"step": step, "epoch": epoch, "final_eval": ev, "skipped": ds.skipped,
                                                  "args": vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] done; export at {out / 'export'}", flush=True)


if __name__ == "__main__":
    main()
