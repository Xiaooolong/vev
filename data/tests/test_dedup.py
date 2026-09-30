import json

from PIL import Image

from data.dedup import Dedup
from data.tests.synth import draw_image, mixed_record, noul_q, rec

LONG_TEXT = ("The committee reviewed the quarterly budget and decided to postpone the bridge repair until the "
             "spring, citing a shortage of certified welders and rising steel prices across the region.")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def img_record(path: str, **kw):
    return rec("x", {"image": {"path": path}}, {"n": noul_q("Is there a dog in the image?")}, {"n": 1.0}, **kw)


def test_image_pass(tmp_path):
    held = tmp_path / "held"
    draw_image(held / "images" / "set" / "a.png", seed=1)
    d = Dedup([], held / "images")
    # same picture, re-encoded as JPEG at another size -> hit
    train = tmp_path / "train"
    train.mkdir()
    Image.open(held / "images" / "set" / "a.png").resize((128, 128)).save(train / "a.jpg", quality=85)
    draw_image(train / "b.png", seed=2)
    assert d.check(img_record("a.jpg"), train) == "image"
    assert d.check(img_record("b.png"), train) is None
    # multi-image record: one matching image is enough
    multi = rec("m", {"left": {"image": {"path": "b.png"}}, "right": {"image": {"path": "a.jpg"}}},
                {"n": noul_q()}, {"n": 0.0})
    assert d.check(multi, train) == "image"


def test_text_pass(tmp_path):
    write_jsonl(tmp_path / "text" / "h.jsonl", [rec("h", LONG_TEXT, {"n": noul_q("Was the repair postponed?")}, {"n": 1.0})])
    d = Dedup([tmp_path / "text"], None)
    near = LONG_TEXT.replace("postpone", "delay")
    assert d.check(rec("t", near, {"n": noul_q("Was the repair postponed?")}, {"n": 1.0})) == "text"
    other = "A cat slept on the warm windowsill all afternoon while the rain kept falling outside the old house."
    assert d.check(rec("u", other, {"n": noul_q("Was the repair postponed?")}, {"n": 1.0})) is None


def test_text_pass_ignores_short_template_on_image_records(tmp_path):
    write_jsonl(tmp_path / "img" / "h.jsonl", [img_record("../images/h/1.jpg")])
    d = Dedup([tmp_path / "img"], None)
    assert d.check(img_record("other.jpg")) is None
    assert d.check(rec("t", "", {"n": noul_q("Is there a dog in the image?")}, {"n": 1.0})) == "text"


def test_coco_pass(tmp_path):
    write_jsonl(tmp_path / "img" / "pope.jsonl", [img_record("../images/pope/COCO_val2014_000000310196.jpg")])
    d = Dedup([tmp_path / "img"], None)
    assert 310196 in d.coco
    assert d.check(mixed_record("a", meta={"coco_id": 310196})) == "coco"
    assert d.check(img_record("../../images/vqav2/COCO_val2014_000000310196.jpg")) == "coco"
    assert d.check(img_record("../../images/vqav2/COCO_train2014_000000000009.jpg")) is None
    assert d.check(mixed_record("b", meta={"coco_id": 9})) is None


def test_coco_pass_external_ids(tmp_path):
    (tmp_path / "ids.txt").write_text("139\n285\n", encoding="utf-8")
    d = Dedup([], None, tmp_path / "ids.txt")
    assert d.check(mixed_record("a", meta={"coco_id": "000000000285"})) == "coco"
    assert d.check(mixed_record("b", meta={"coco_id": 286})) is None
