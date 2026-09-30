"""VideoGameQA-Bench (single-image judgement tasks) -> three sets, each a noul with the upstream judgement text
as the instruction:

  game_glitch  BugDetection/GlitchDetection, 1,000 screenshots, glitch_detected balanced 500/500
  game_clip    ClippingDetection/ParametricTest, 686 renders, clipping_detected 384/302
  game_vr      VisualRegression (Unity 171 all false + Cutscene 79: 53 false / 26 true): reference + candidate image,
               the ACCEPTABLE / UNACCEPTABLE lists of the upstream prompt are the criteria. Diagnostic only (26 positives).

python -m evals.datasets.image_game_spec --out-dir evals/data/image --limit 20

Images are the repo's jpg files (2,486). Ids derive from the upstream custom_id; meta.cluster = media_source
(the video/game the frame came from)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evals.datasets._common import IMAGE_MANIFEST, ImageSetWriter

HF_ID = "taesiri/VideoGameQA-Bench"
REVISION = "main"
LICENSE = "commercial-ok"  # cc-by-4.0 annotations; frames belong to the games' publishers
CUT = re.compile(r"\n\s*(Provide your answer|Based on your analysis, respond|Respond using|Provide your response|Output)", re.I)


def judgement_text(q: str) -> str:
    """The upstream prompt without its output-format tail."""
    m = CUT.search(q)
    return (q[: m.start()] if m else q).strip()


def gt(row) -> dict | None:
    try:
        g = json.loads(row["ground_truth"])
    except Exception:
        return None
    return g if isinstance(g, dict) else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evals/data/image")
    ap.add_argument("--limit", type=int, default=0, help="records per set; 0 = all")
    ap.add_argument("--manifest", default=str(IMAGE_MANIFEST))
    ap.add_argument("--src", default=None, help="local snapshot of the HF repo (default: download)")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, snapshot_download
    rev = HfApi().dataset_info(HF_ID, revision=REVISION).sha
    if a.src:
        root = Path(a.src)
    else:
        root = Path(snapshot_download(HF_ID, repo_type="dataset", revision=rev,
                                      allow_patterns=["data/*", "images/*", "images/**"]))
    import pyarrow.parquet as pq
    table = pq.read_table(next(root.glob("data/*.parquet")))
    rows = table.to_pylist()
    ns = argparse.Namespace
    writers = {name: ImageSetWriter(name, ns(out=f"{a.out_dir}/{name}.jsonl", limit=a.limit, images_dir=None, manifest=a.manifest))
               for name in ("game_glitch", "game_clip", "game_vr")}
    n_full = {"game_glitch": 0, "game_clip": 0, "game_vr": 0}
    seen_text: dict[str, set] = {k: set() for k in writers}
    for row in rows:
        if row["media_type"] != "images":
            continue
        cats = set(row["question_categories"] or [])
        g = gt(row)
        if g is None:
            continue
        paths = [root / "images" / row["custom_id"] / p for p in (row["media_path"] or [])]
        if any(not p.exists() for p in paths):
            continue
        cid = row["custom_id"]
        # media_source names the video/game for the glitch rows; for the synthetic clipping renders and the Unity
        # regression pairs it is one constant, so fall back to the scene token in the id (sphere_<scene>-samples__…)
        # and finally to the row itself
        src = str(row.get("media_source") or "")
        m = re.match(r"sphere_([A-Za-z0-9]+)-", cid)
        cluster = m.group(1) if m else (src if src and not src.startswith(("Unity", "unity")) else cid)
        instr = judgement_text(row["question"])
        if "GlitchDetection" in cats and "glitch_detected" in g and len(paths) == 1:
            name, target, key_state = "game_glitch", bool(g["glitch_detected"]), None
        elif "ClippingDetection" in cats and "clipping_detected" in g and len(paths) == 1:
            name, target, key_state = "game_clip", bool(g["clipping_detected"]), None
        elif "VisualRegression" in cats and "test_pass" in g and len(paths) == 2:
            name, target, key_state = "game_vr", bool(g["test_pass"]), "pair"
        else:
            continue
        n_full[name] += 1
        w = writers[name]
        if w.full:
            continue
        seen_text[name].add(instr)
        raw = [{"bytes": p.read_bytes(), "path": str(p)} for p in paths]
        if key_state == "pair":
            ref, cand = w.image(raw[0], f"{cid}_q0"), w.image(raw[1], f"{cid}_q1")
            assert ref["image"]["path"] != cand["image"]["path"]
            state = {"reference": ref, "candidate": cand}
            subset = "unity" if "Unity" in cats else "cutscene"
        else:
            state = w.image(raw[0], cid)
            subset = None
        meta = {"grid_id": f"{name}/{cid}", "custom_id": cid, "cluster": cluster, "categories": sorted(cats),
                "orig_answer_type": "json_bool"}
        if subset:
            meta["subset"] = subset
        w.add({"id": f"{name}/test/{cid}", "source": name, "split": "test", "license": LICENSE, "lang": "en",
               "state": state, "questions": {"q1": {"type": "noul", "instructions": instr}},
               "targets": {"q1": 1.0 if target else 0.0}, "meta": meta})
    notes = {
        "game_glitch": "VideoGameQA-Bench BugDetection/GlitchDetection single-image rows; instruction = upstream prompt minus "
                       "the output-format tail (the definition of 'glitch' is the upstream one). meta.cluster = media_source.",
        "game_clip": "ClippingDetection/ParametricTest rows (synthetic white-sphere renders); instruction = upstream prompt "
                     "minus the output-format tail. meta.cluster = media_source.",
        "game_vr": "VisualRegression rows: state.reference / state.candidate; the upstream ACCEPTABLE/UNACCEPTABLE lists are "
                   "the criteria. Unity subset (171) is all test_pass=false; Cutscene (79) 53/26. Diagnostic set: report "
                   "Cutscene-only discrimination and Unity false-positive rate, not in the gate.",
    }
    for name, w in writers.items():
        w.finish({"hf_id": HF_ID, "hf_config": None, "hf_split": "test", "hf_revision": rev, "n_records_full": n_full[name],
                  "license": LICENSE, "lang": "en",
                  "notes": notes[name] + f" {len(seen_text[name])} distinct instruction texts. Data license cc-by-4.0."})


if __name__ == "__main__":
    main()
