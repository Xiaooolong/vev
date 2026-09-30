"""The prior path must reproduce the zero-shot readout's label logits element-wise.

    python -m train.check_prior_equivalence --base Qwen/Qwen3.5-0.8B --ckpt runs/prior-only-0.8b --records evals/data/text/judgekit.jsonl --n 20

Loads the zero-shot Engine and the prior-only PointerEngine on the same device, runs the same requests through both,
and reports the max |Δlogit| and argmax agreement over the label logits (both in bf16, so ~1e-2 is expected noise)."""

from __future__ import annotations

import argparse
import json

import torch

from evals.schema import iter_records
from vev.pointer import PointerEngine
from vev.readout import Engine


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt", required=True, help="prior-only checkpoint (train.make_prior_ckpt)")
    ap.add_argument("--records", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    zs = Engine(a.base, dtype="bf16", device=a.device)
    pe = PointerEngine(a.ckpt, dtype="bf16", device=a.device)
    assert pe.model.prior, "checkpoint is not in prior mode"
    worst, agree, n = 0.0, 0, 0
    for i, rec in enumerate(iter_records(a.records)):
        if i >= a.n:
            break
        r1 = zs.run(rec["state"], rec["questions"])
        r2 = pe.run(rec["state"], rec["questions"])
        for name in rec["questions"]:
            z1 = torch.tensor(r1.extensions["logits"][name])
            z2 = torch.tensor(r2.extensions["logits"][name])
            if z1.numel() == 0:
                continue
            d = float((z1 - z2).abs().max())
            worst = max(worst, d)
            agree += int(z1.argmax() == z2.argmax())
            n += 1
    print(json.dumps({"questions": n, "max_abs_logit_diff": round(worst, 4), "argmax_agreement": round(agree / max(1, n), 4)}))


if __name__ == "__main__":
    main()
