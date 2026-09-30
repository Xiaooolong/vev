"""Release gate: paired per-question comparison of a trained model against the zero-shot reference.
    python -m evals.gate --model results/vev-9b --ref results/qwen3.5-9b \
        --sets judgekit jevbench nimble kev_transfer_v4 mmbench_en pope mmstar --out gate.md
Each set is scored on the questions both results answered. A paired bootstrap (--boot resamples, seeded) gives 95%
intervals for the accuracy and Brier differences; when the records carry meta.cluster (same image / prompt / app),
whole clusters are resampled instead of questions. Verdict per set:
  FAIL  accuracy CI upper < 0 (significantly worse), or accuracy point drop > --max-drop points (only when
        n >= --min-n; below that the point drop is noise), or Brier CI lower > 0 (significantly worse calibration)
  PASS  otherwise (reported as 'better' when the accuracy CI lower > 0).
Both result files must have been produced on the same records file (records_sha256), which must be the one on disk;
otherwise ids may silently pair different questions. --allow-sha-mismatch downgrades that to a warning.
--ref entries are directories searched in order for <set>.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def record_file(s: str) -> Path:
    for kind in ("image", "text"):
        p = Path("evals/data") / kind / f"{s}.jsonl"
        if p.exists():
            return p
    raise SystemExit(f"no records file for set {s} under evals/data/{{image,text}}/")


def per_question(path: Path, recs: dict) -> tuple[dict, str | None]:
    res = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for e in res["raw"]:
        if "error" in e or e["id"] not in recs:
            continue
        r = recs[e["id"]]
        for nm, q in e["questions"].items():
            if "pred" not in q:
                continue
            qq, t, p = r["questions"][nm], r["targets"][nm], q["pred"]
            if qq["type"] == "noul":
                p, t = float(p), float(t)
                out[(e["id"], nm)] = (float((p >= 0.5) == (t >= 0.5)), 2 * (p - t) ** 2)   # ties count as "true", as in evals.metrics
                continue
            labs = list(qq["criteria"]) if qq["type"] == "choice" else [str(i) for i in range(len(qq["criteria"]))]
            pd, td = dict(p), dict(t)
            ok = max(labs, key=lambda k: (float(pd.get(k, 0)), -labs.index(k))) == max(labs, key=lambda k: (float(td.get(k, 0)), -labs.index(k)))
            out[(e["id"], nm)] = (float(ok), sum((float(pd.get(k, 0)) - float(td.get(k, 0))) ** 2 for k in labs))
    return out, res.get("records_sha256")


def find_ref(refs: list[str], s: str) -> Path | None:
    for d in refs:
        p = Path(d) / f"{s}.json"
        if p.exists():
            return p
    return None


def ci(diffs: list[float], boot: int, rng: random.Random, clusters: list | None = None) -> tuple[float, float]:
    """95% bootstrap interval of the mean; with clusters, whole clusters are resampled (sizes kept)."""
    if clusters is None:
        n = len(diffs)
        means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot))
        return means[int(0.025 * boot)], means[int(0.975 * boot) - 1]
    groups: dict = {}
    for d, c in zip(diffs, clusters):
        groups.setdefault(c, [0.0, 0])
        groups[c][0] += d
        groups[c][1] += 1
    sums = list(groups.values())
    m = len(sums)
    means = []
    for _ in range(boot):
        s = n = 0
        for _ in range(m):
            a, b = sums[rng.randrange(m)]
            s += a
            n += b
        means.append(s / n)
    means.sort()
    return means[int(0.025 * boot)], means[int(0.975 * boot) - 1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="directory holding <set>.json results of the model under test")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--max-drop", type=float, default=3.0, help="accuracy points")
    ap.add_argument("--min-n", type=int, default=600, help="the point-drop rule applies only from this many questions")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-sha-mismatch", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    lines = [f"# Release gate: {a.model} (reference = zero-shot; FAIL = accuracy significantly worse, or a drop > {a.max_drop:g} points (n >= {a.min_n}), or Brier significantly worse)", "",
             "| set | n | clusters | ref acc | acc | Δacc | 95% CI | ref Brier | Brier | ΔBrier 95% CI | better/worse | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fails = []
    for s in a.sets:
        mp, rp = Path(a.model) / f"{s}.json", find_ref(a.ref, s)
        if not mp.exists() or rp is None:
            lines.append(f"| {s} | - | missing result (model {mp.exists()} / ref {rp is not None}) |||||||||||")
            continue
        rf = record_file(s)
        recs = {}
        for line in rf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                recs[r["id"]] = r
        (m, sha_m), (r, sha_r) = per_question(mp, recs), per_question(rp, recs)
        sha_disk = hashlib.sha256(rf.read_bytes()).hexdigest()
        if not (sha_m == sha_r == sha_disk):
            msg = f"[gate] {s}: records_sha256 differs (model {str(sha_m)[:8]} / ref {str(sha_r)[:8]} / on disk {sha_disk[:8]}); ids may pair different questions"
            if not a.allow_sha_mismatch:
                lines.append(f"| {s} | - | sha mismatch ({str(sha_m)[:8]} / {str(sha_r)[:8]} / on disk {sha_disk[:8]}); pass --allow-sha-mismatch to compare anyway |||||||||||")
                print(msg, file=sys.stderr)
                fails.append(s)
                continue
            print(msg + " (allowed)", file=sys.stderr)
        keys = sorted(set(m) & set(r))
        n = len(keys)
        if n == 0:
            lines.append(f"| {s} | 0 | no shared questions |||||||||||")
            fails.append(s)
            continue
        da = [m[k][0] - r[k][0] for k in keys]
        db = [m[k][1] - r[k][1] for k in keys]
        clusters = [((recs[k[0]].get("meta") or {}).get("cluster")) for k in keys]
        use_clusters = any(c is not None for c in clusters)
        cl = [c if c is not None else k[0] for c, k in zip(clusters, keys)] if use_clusters else None
        if cl is not None and len(set(cl)) < 20:   # too few clusters for a cluster bootstrap: back to questions
            cl = None
        mean_a = sum(da) / n
        alo, ahi = ci(da, a.boot, rng, cl)
        blo, bhi = ci(db, a.boot, rng, cl)
        acc_m, acc_r = sum(m[k][0] for k in keys) / n, sum(r[k][0] for k in keys) / n
        br_m, br_r = sum(m[k][1] for k in keys) / n, sum(r[k][1] for k in keys) / n
        why = []
        if ahi < 0:
            why.append("accuracy significantly worse")
        if n >= a.min_n and mean_a * 100 < -a.max_drop:
            why.append(f"dropped {-mean_a * 100:.1f} points")
        if blo > 0:
            why.append("Brier significantly worse")
        verdict = "**FAIL** (" + "; ".join(why) + ")" if why else ("PASS (significantly better)" if alo > 0 else "PASS")
        if not why and n < a.min_n:
            verdict += " (small n, CI only)"
        if why:
            fails.append(s)
        up, dn = sum(x > 0 for x in da), sum(x < 0 for x in da)
        ncl = f"{len(set(cl))}" if cl else "per-question"
        lines.append(f"| {s} | {n} | {ncl} | {acc_r:.3f} | {acc_m:.3f} | {mean_a * 100:+.1f} | [{alo * 100:+.1f}, {ahi * 100:+.1f}] | {br_r:.3f} | {br_m:.3f} | "
                     f"[{blo:+.3f}, {bhi:+.3f}] | {up}/{dn} | {verdict} |")
    lines += ["", f"Result: {'all sets pass' if not fails else 'FAIL: ' + ', '.join(fails)}",
              "", "clusters = number of clusters when resampling whole meta.cluster groups; \"per-question\" = questions resampled individually."]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
