import json
from pathlib import Path

from data import build
from data.coco_licenses import license_map, lookup
from data.coco_licenses import main as coco_main
from data.tests.synth import choice_q, draw_image, mixed_record, noul_q, rec, score_q
from data.tests.test_dedup import LONG_TEXT
from evals.schema import iter_records, validate_record

HELD_COCO = 555


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def make_world(tmp: Path) -> dict:
    # held-out: one image, one long text, one COCO val image id
    held = tmp / "held"
    draw_image(held / "images" / "h" / "0.png", seed=999)
    write_jsonl(held / "text" / "h.jsonl", [mixed_record("h", state=LONG_TEXT)])
    write_jsonl(held / "image" / "h.jsonl",
                [rec("hi", {"image": {"path": f"../images/h/COCO_val2014_{HELD_COCO:012d}.jpg"}},
                     {"n": noul_q()}, {"n": 1.0})])

    raw = tmp / "raw"
    text = [mixed_record(f"synth_text/raw/{i:06d}", source="synth_text", state=f"text state number {i}",
                         lang="zh" if i % 5 == 0 else "en") for i in range(120)]
    text.append(mixed_record("synth_text/raw/dup", source="synth_text", state=LONG_TEXT))
    write_jsonl(raw / "text" / "synth_text.jsonl", text)
    # converter wrote commercial-ok, licenses.json says non-commercial: licenses.json wins
    write_jsonl(raw / "text" / "synth_nc.jsonl",
                [mixed_record(f"synth_nc/raw/{i:06d}", source="synth_nc", state=f"nc {i}") for i in range(10)])

    imgs = []
    for i in range(40):
        draw_image(tmp / "images" / "synth_img" / f"{i}.png", seed=i)
        state = {"image": {"path": f"../../images/synth_img/{i}.png"}}
        grid = f"synth_img/{i}"
        imgs.append(rec(f"synth_img/raw/{i}a", state, {"q": choice_q(4)},
                        {"q": {"A": 0.25, "B": 0.5, "C": 0.25, "D": 0.0}}, source="synth_img", grid_id=grid))
        imgs.append(rec(f"synth_img/raw/{i}b", state, {"q": score_q(3)},
                        {"q": {"0": 0.2, "1": 0.3, "2": 0.5}}, source="synth_img", grid_id=grid,
                        meta={"coco_id": 1000 + i}))
    draw_image(tmp / "images" / "synth_img" / "held.png", seed=999)
    imgs.append(rec("synth_img/raw/held", {"image": {"path": "../../images/synth_img/held.png"}}, {"n": noul_q()},
                    {"n": 0.0}, source="synth_img"))
    imgs.append(rec("synth_img/raw/val", {"image": {"path": "../../images/synth_img/0.png"}}, {"n": noul_q()},
                    {"n": 1.0}, source="synth_img", meta={"coco_id": HELD_COCO}))
    write_jsonl(raw / "image" / "synth_img.jsonl", imgs)

    tiers = {"synth_text": "commercial-ok", "synth_img": "commercial-ok", "synth_nc": "non-commercial"}
    (tmp / "licenses.json").write_text(json.dumps({k: {"tier": v} for k, v in tiers.items()}), encoding="utf-8")
    # COCO per-image licenses: odd ids NC (2), even ids CC BY (4); the held-out val id is CC BY
    coco = {str(1000 + i): (2 if i % 2 else 4) for i in range(40)}
    coco[str(HELD_COCO)] = 4
    (tmp / "coco_licenses.json").write_text(json.dumps(coco), encoding="utf-8")
    return {"raw": raw, "held": held}


def run(tmp: Path, *extra) -> dict:
    w = make_world(tmp)
    argv = ["--version", "vt", "--seed", "0", "--raw-dir", str(w["raw"]), "--out-dir", str(tmp / "build"),
            "--manifest-dir", str(tmp / "manifest"), "--licenses", str(tmp / "licenses.json"),
            "--heldout-records", f"{w['held'] / 'image'},{w['held'] / 'text'}",
            "--heldout-images", str(w["held"] / "images"), *extra]
    assert build.main(argv) == 0
    return json.loads((tmp / "manifest" / "vt.json").read_text(encoding="utf-8"))


def test_build_end_to_end(tmp_path):
    m = run(tmp_path, "--preview", "12")
    out = tmp_path / "build" / "vt"
    for name in ("train.jsonl", "calibration.jsonl", "validation.jsonl", "augment_config.json",
                 "preview.jsonl", "preview.md"):
        assert (out / name).exists(), name

    s = m["sources"]
    assert set(s) == {"synth_text", "synth_img", "synth_nc"}
    assert s["synth_nc"]["license"] == "non-commercial" and s["synth_nc"]["license_dropped"] == 10
    assert s["synth_text"]["dedup_dropped"] == {"image": 0, "text": 1, "coco": 0}
    assert s["synth_img"]["dedup_dropped"] == {"image": 1, "text": 0, "coco": 1}
    assert s["synth_img"]["coco_license_dropped"] == 0
    assert s["synth_text"]["raw_records"] == 121 and len(s["synth_text"]["raw_sha256"]) == 64
    assert s["synth_text"]["converter_sha256"] is None
    assert all(v > 0 for v in s["synth_text"]["splits"].values())
    assert m["totals"]["by_lang_records"]["zh"] > 0 and set(m["totals"]["by_bucket_records"]) == {"text", "image"}
    assert m["augment_config"]["p"]["shuffle"] == 1.0

    records = []
    for sp in ("train", "calibration", "validation"):
        for r in iter_records(out / f"{sp}.jsonl"):
            assert validate_record(r) == [] and r["split"] == sp
            records.append(r)
    assert len(records) == m["totals"]["records"]
    ids = {mid for r in records for mid in r["meta"]["members"]}
    assert "synth_text/raw/dup" not in ids and "synth_img/raw/held" not in ids and "synth_img/raw/val" not in ids
    # image paths resolve from the build directory
    for r in records:
        if isinstance(r["state"], dict) and "image" in r["state"]:
            assert (out / r["state"]["image"]["path"]).exists()

    preview = list(iter_records(out / "preview.jsonl"))
    assert len(preview) == 12 and all("aug" in r["meta"] and validate_record(r) == [] for r in preview)
    assert "aug:" in (out / "preview.md").read_text(encoding="utf-8")


def test_build_coco_license_filter(tmp_path):
    m = run(tmp_path, "--coco-licenses", str(tmp_path / "coco_licenses.json"))
    # 20 odd ids are NC; the COCO-val record is CC BY here, so it still reaches (and is dropped by) dedup
    assert m["sources"]["synth_img"]["coco_license_dropped"] == 20
    assert m["sources"]["synth_img"]["dedup_dropped"]["coco"] == 1
    members = set()
    for sp in ("train", "calibration", "validation"):
        members |= {mid for r in iter_records(tmp_path / "build" / "vt" / f"{sp}.jsonl") for mid in r["meta"]["members"]}
    assert "synth_img/raw/1b" not in members and "synth_img/raw/2b" in members
    assert "synth_img/raw/1a" in members  # no coco_id -> not subject to the filter


def test_coco_licenses_map(tmp_path):
    ann = {"images": [{"id": 9, "file_name": "COCO_train2014_000000000009.jpg", "license": 3},
                      {"id": 25, "file_name": "000000000025.jpg", "license": 4}]}
    (tmp_path / "instances_train2014.json").write_text(json.dumps(ann), encoding="utf-8")
    mapping = license_map([tmp_path / "instances_train2014.json"])
    assert mapping == {"9": 3, "COCO_train2014_000000000009.jpg": 3, "25": 4, "000000000025.jpg": 4}
    assert lookup(mapping, 25) == 4 and lookup(mapping, "000000000009") == 3
    assert lookup(mapping, "../images/x/000000000025.jpg") == 4 and lookup(mapping, 7) is None
    assert coco_main(["--annotations", str(tmp_path / "instances_train2014.json"),
                      "--out", str(tmp_path / "m.json")]) == 0
    assert json.loads((tmp_path / "m.json").read_text(encoding="utf-8")) == mapping
