"""Sanity checks over a built dataset before training.

    python -m data.check --build data/build/v1-research --out data/check/v1-research [--augment-sample 10000] [--montage 3]

Writes report.json + report.md, and (with --montage N) one PNG per source showing N random records:
image (if any), state text, questions, options and targets, for a human to eyeball.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import sys
import textwrap
from pathlib import Path

from data.augment import DEFAULT_CONFIG, Augmenter
from evals.schema import iter_records, validate_record


def _images(node, out: list) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "image" and isinstance(v, dict) and ("path" in v or "url" in v):
                out.append(v)
            else:
                _images(v, out)
    elif isinstance(node, list):
        for v in node:
            _images(v, out)


def _text(node) -> str:
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return " ".join(_text(v) for k, v in node.items() if k != "image")
    if isinstance(node, list):
        return " ".join(_text(v) for v in node)
    return str(node) if node is not None else ""


def check_split(path: Path, base_dir: Path, sample_per_source: int, rng: random.Random) -> tuple[dict, dict]:
    per: dict[str, dict] = collections.defaultdict(lambda: {
        "records": 0, "questions": 0, "by_type": collections.Counter(), "derived": 0, "images": 0,
        "missing_images": 0, "invalid": 0, "grid_sizes": [], "text_chars": [], "langs": collections.Counter(),
        "errors": [], "samples": []})
    seen_ids: set[str] = set()
    dup_ids = 0
    for rec in iter_records(path):
        s = per[rec["source"]]
        s["records"] += 1
        if rec["id"] in seen_ids:
            dup_ids += 1
        seen_ids.add(rec["id"])
        errs = validate_record(rec)
        if errs:
            s["invalid"] += 1
            if len(s["errors"]) < 3:
                s["errors"].append({"id": rec["id"], "errors": errs[:3]})
        qs = rec["questions"]
        s["questions"] += len(qs)
        s["grid_sizes"].append(len(qs))
        s["langs"][rec.get("lang", "?")] += 1
        s["derived"] += len((rec.get("meta") or {}).get("derived") or [])
        for q in qs.values():
            s["by_type"][q["type"]] += 1
        imgs: list = []
        _images(rec["state"], imgs)
        s["images"] += len(imgs)
        for im in imgs:
            if "path" in im and not (base_dir / im["path"]).exists():
                s["missing_images"] += 1
        s["text_chars"].append(len(_text(rec["state"])))
        if len(s["samples"]) < sample_per_source:
            s["samples"].append(rec)
        elif rng.random() < sample_per_source / s["records"]:  # reservoir
            s["samples"][rng.randrange(sample_per_source)] = rec
    report = {}
    for src, s in per.items():
        gs, tc = s["grid_sizes"], s["text_chars"]
        report[src] = {
            "records": s["records"], "questions": s["questions"], "by_type": dict(s["by_type"]),
            "derived_questions": s["derived"], "images": s["images"], "missing_images": s["missing_images"],
            "invalid_records": s["invalid"], "langs": dict(s["langs"]),
            "questions_per_record": {"mean": round(statistics.mean(gs), 2), "max": max(gs)},
            "state_chars": {"median": int(statistics.median(tc)), "p95": int(sorted(tc)[int(0.95 * (len(tc) - 1))])},
            "errors": s["errors"],
        }
    return report, {"dup_ids": dup_ids, "samples": {src: s["samples"] for src, s in per.items()}}


def augment_smoke(records: list[dict], aug_dir: Path, seed: int) -> dict:
    rng = random.Random(seed)
    text_pool = [_text(r["state"]) for r in records if isinstance(r["state"], str)][:2000]
    image_pool = []
    for r in records:
        imgs: list = []
        _images(r["state"], imgs)
        image_pool += [i["path"] for i in imgs if "path" in i]
    image_pool = image_pool[:2000]
    aug = Augmenter(DEFAULT_CONFIG, rng, text_pool, image_pool, aug_dir)
    applied: collections.Counter = collections.Counter()
    failures: list = []
    bad_targets = 0
    for r in records:
        try:
            out = aug.apply(r)
        except Exception as e:  # noqa: BLE001
            if len(failures) < 10:
                failures.append({"id": r["id"], "error": f"{type(e).__name__}: {str(e)[:160]}"})
            continue
        for a in out["meta"].get("aug", []):
            applied[a] += 1
        for name, q in out["questions"].items():
            t = out["targets"][name]
            if q["type"] == "noul":
                ok = 0.0 <= float(t) <= 1.0
            else:
                ok = abs(sum(t.values()) - 1.0) < 1e-6 and set(t) == (set(q["criteria"]) if q["type"] == "choice"
                                                                      else {str(i) for i in range(len(q["criteria"]))})
            if not ok:
                bad_targets += 1
        if validate_record(out):
            bad_targets += 1
    return {"n": len(records), "applied": dict(applied), "failures": failures, "bad_records_after_aug": bad_targets}


def montage(samples: list[dict], base_dir: Path, out_png: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    for cand in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "C:/Windows/Fonts/msyh.ttc"):
        if Path(cand).exists():
            font = ImageFont.truetype(cand, 12)
            break
    panels = []
    for rec in samples:
        imgs: list = []
        _images(rec["state"], imgs)
        thumbs = []
        for im in imgs[:2]:
            p = base_dir / im["path"] if "path" in im else None
            if p and p.exists():
                t = Image.open(p).convert("RGB")
                t.thumbnail((420, 420))
                thumbs.append(t)
        lines = [f"[{rec['id']}]  lang={rec.get('lang')}  aug={rec.get('meta', {}).get('aug', '-')}"]
        lines += textwrap.wrap("state: " + _text(rec["state"])[:600], 110)[:6]
        for name, q in rec["questions"].items():
            lines += textwrap.wrap(f"{name} ({q['type']}): {q.get('instructions', '')}", 110)[:3]
            t = rec["targets"][name]
            if q["type"] == "noul":
                lines.append(f"    P(true) = {float(t):.2f}")
            elif q["type"] == "choice":
                for lab, desc in list(q["criteria"].items())[:10]:
                    lines.append(f"    {t.get(lab, 0):.2f}  {lab}: {str(desc)[:80]}")
                if len(q["criteria"]) > 10:
                    lines.append(f"    ... {len(q['criteria'])} options")
            else:
                for i, desc in enumerate(q["criteria"]):
                    lines.append(f"    {t.get(str(i), 0):.2f}  level {i}: {str(desc)[:80]}")
        text_h = 14 * len(lines) + 10
        w = 900 + (sum(th.width for th in thumbs) + 10 * len(thumbs))
        h = max(text_h, max((th.height for th in thumbs), default=0)) + 10
        panel = Image.new("RGB", (w, h), (255, 255, 255))
        x = 5
        for th in thumbs:
            panel.paste(th, (x, 5))
            x += th.width + 10
        d = ImageDraw.Draw(panel)
        y = 5
        for ln in lines:
            d.text((x + 5, y), ln, fill=(0, 0, 0), font=font)
            y += 14
        panels.append(panel)
    W = max(p.width for p in panels)
    H = sum(p.height for p in panels) + 8 * len(panels)
    sheet = Image.new("RGB", (W, H), (200, 200, 200))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 8
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, format="PNG")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="data/build/<version>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--augment-sample", type=int, default=10000)
    ap.add_argument("--montage", type=int, default=0, help="records per source to draw into a PNG")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build, out = Path(a.build), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)

    report: dict = {"build": str(build), "splits": {}}
    samples_by_split: dict = {}
    for split in ("train", "calibration", "validation"):
        p = build / f"{split}.jsonl"
        if not p.exists():
            continue
        print(f"[check] {split} …", file=sys.stderr, flush=True)
        rep, extra = check_split(p, build, max(a.montage, 3), rng)
        report["splits"][split] = {"sources": rep, "dup_ids": extra["dup_ids"]}
        samples_by_split[split] = extra["samples"]

    # per-source samples for offline spot checks (a Sonnet pass over text sources, a human pass over montages)
    with (out / "samples.jsonl").open("w", encoding="utf-8") as fh:
        for recs in samples_by_split.get("train", {}).values():
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool = list(iter_records(build / "train.jsonl"))
    rng.shuffle(pool)
    print(f"[check] augment smoke on {a.augment_sample} records …", file=sys.stderr, flush=True)
    report["augment_smoke"] = augment_smoke(pool[:a.augment_sample], out / "aug_images", a.seed)

    # exact duplicate questions inside train (dedup only compares against held-out); the key includes image paths,
    # otherwise every "Is it raining?" on a different photo counts as a duplicate
    seen: dict = {}
    dups = 0
    for r in pool:
        imgs: list = []
        _images(r["state"], imgs)
        img_key = tuple(i.get("path") or i.get("url", "")[:64] for i in imgs)
        for q in r["questions"].values():
            key = (r["source"], (q.get("instructions") or "")[:200], _text(r["state"])[:200], img_key)
            if key in seen:
                dups += 1
            seen[key] = 1
    report["train_exact_duplicate_questions"] = dups

    if a.montage:
        for src, recs in samples_by_split.get("train", {}).items():
            try:
                montage(recs[: a.montage], build, out / "montage" / f"{src}.png")
            except Exception as e:  # noqa: BLE001
                report.setdefault("montage_errors", []).append(f"{src}: {type(e).__name__}: {str(e)[:120]}")

    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"# data check: {build}", ""]
    for split, rep in report["splits"].items():
        lines += [f"## {split} (dup ids: {rep['dup_ids']})", "",
                  "| source | records | questions | types | derived | images | missing img | invalid | q/rec mean/max | state chars p50/p95 | langs |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for src, s in sorted(rep["sources"].items()):
            lines.append(f"| {src} | {s['records']} | {s['questions']} | {s['by_type']} | {s['derived_questions']} | {s['images']} | "
                         f"{s['missing_images']} | {s['invalid_records']} | {s['questions_per_record']['mean']}/{s['questions_per_record']['max']} | "
                         f"{s['state_chars']['median']}/{s['state_chars']['p95']} | {s['langs']} |")
        lines.append("")
    sm = report["augment_smoke"]
    lines += ["## augment smoke", "", f"n={sm['n']} applied={sm['applied']} failures={len(sm['failures'])} bad_after_aug={sm['bad_records_after_aug']}", ""]
    for f in sm["failures"]:
        lines.append(f"- {f}")
    lines += ["", f"train exact duplicate questions (same source/state/instructions): {report['train_exact_duplicate_questions']}"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[check] wrote {out / 'report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
