from data.sources import _localhf


def test_parse_hf_resolve_url():
    repo, rev, path = _localhf.parse_url(
        "https://huggingface.co/datasets/hflqf88888/GUIOdyssey/resolve/61632d0f/screenshots/screenshots.z01")
    assert (repo, rev, path) == ("hflqf88888/GUIOdyssey", "61632d0f", "screenshots/screenshots.z01")
    assert _localhf.parse_url("https://www.dropbox.com/s/x/foil.json?dl=1") is None
    assert _localhf.parse_url("http://images.cocodataset.org/train2014/COCO_train2014_000000000009.jpg") is None


def test_parse_hf_fs_path():
    assert _localhf.parse_hf_path("datasets/Kaining/MMT-Bench@4980433/MMT-Bench_VAL.tsv") == \
        ("Kaining/MMT-Bench", "4980433", "MMT-Bench_VAL.tsv")
    assert _localhf.parse_hf_path("datasets/THUDM/ImageRewardDB/images/train/train_1.zip") == \
        ("THUDM/ImageRewardDB", None, "images/train/train_1.zip")
    assert _localhf.parse_hf_path("datasets/a/b@rev%2Fx/dir/f.parquet") == ("a/b", "rev/x", "dir/f.parquet")
    assert _localhf.parse_hf_path("/tmp/local.parquet") is None


def test_disabled_by_default_returns_none():
    assert _localhf.ENABLED is False or _localhf.local_for_url("https://example.com/x") is None
    assert _localhf.local_for_hf_path("/tmp/x") is None
