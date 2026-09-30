"""MobileViews (Android screenshots + view hierarchies, MIT) -> two rule-labelled UI-state sets:

  ui_toggle   one screen, two records (meta.flip_group): "is there a switch/checkbox turned ON" and "... turned OFF";
              only screens whose visible checkables are all in a switch/checkbox class whitelist and all ON or all OFF,
              so the two targets are opposite (criteria-flip pairs)
  ui_input    "is there a text input field": positives have a visible EditText/editable node; negatives have none, a
              covered hierarchy, no WebView and no search/hint text

A loading-spinner set was tried and dropped: a visible, spinner-shaped ProgressBar node is drawn on screen only about
half the time (video players keep a hidden buffering spinner; open drawers cover it), so the rule label is not trusted.

Labels come from the hierarchy (visible, on-screen, non-empty bounds), never from a model. Screens with an open
dialog, or with WebView content when used as negatives, are skipped because the hierarchy does not show what is on
screen there. Row groups are read one at a time over HTTP range requests (500 screens, ~140 MB each); the sampled
row groups, seed and rules are recorded in the manifest notes. meta.cluster = app package.

python -m evals.datasets.image_ui_state --out-dir evals/data/image --row-groups 60 --per-set 400 --limit 20"""

from __future__ import annotations

import argparse
import io
import json
import random

from evals.datasets._common import IMAGE_MANIFEST, ImageSetWriter

HF_ID = "mllmTeam/MobileViews"
REVISION = "main"
PARQUETS = ["MobileViews_0-150000.parquet", "MobileViews_150001-291197.parquet",
            "MobileViews_300000-400000.parquet", "MobileViews_400000-522301.parquet"]
PREFIX = "MobileViews_Screenshots_ViewHierarchies/Parquets"
# ToggleButton is left out on purpose: apps use it for image toggles (a flag, a star) that no one would call a switch
TOGGLE_CLASSES = {"Switch", "SwitchCompat", "SwitchMaterial", "MaterialSwitch", "CheckBox", "MaterialCheckBox",
                  "AppCompatCheckBox"}
DIALOG_IDS = {"android:id/parentPanel", "android:id/alertTitle", "android:id/buttonPanel", "android:id/custom",
              "android:id/contentPanel", "android:id/topPanel"}
MIN_NODES = 15  # a hierarchy with fewer visible nodes is not trusted to describe the screen (Flutter/Compose/games)
Q = {
    "on": "Is there a switch or checkbox on this screen that is turned ON (checked)?",
    "off": "Is there a switch or checkbox on this screen that is turned OFF (unchecked)?",
    "input": "Is there a text input field (a box where the user can type) on this screen?",
}


def basename(cls: str) -> str:
    return (cls or "").rsplit(".", 1)[-1].rsplit("$", 1)[-1]


def on_screen(v: dict, w: int, h: int) -> bool:
    b = v.get("bounds")
    if not b or len(b) != 2:
        return False
    (x1, y1), (x2, y2) = b
    lim = max(w, h)
    return x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= lim and y2 <= lim


def dhash(im) -> int:
    g = im.convert("L").resize((9, 8))
    px = list(g.getdata())
    bits = 0
    for r in range(8):
        for c in range(8):
            bits = (bits << 1) | (px[r * 9 + c] > px[r * 9 + c + 1])
    return bits


def occluded(node: dict, views: list, w: int, h: int) -> bool:
    """True when a visible leaf listed after the node (drawn on top: later sibling subtree, popup window, overlay)
    covers at least half of the node's box. The dump is in draw order, ancestors come before their subtree."""
    (x1, y1), (x2, y2) = node["bounds"]
    area = (x2 - x1) * (y2 - y1)
    idx = node.get("temp_id")
    for v in views:
        if v.get("temp_id", -1) <= idx or not v.get("visible") or not on_screen(v, w, h) or v.get("child_count"):
            continue
        (a1, b1), (a2, b2) = v["bounds"]
        inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
        if inter >= 0.5 * area:
            return True
    return False


def analyse(j: dict) -> dict | None:
    views = j.get("views") or []
    w, h = int(j.get("width") or 0), int(j.get("height") or 0)
    vis = [v for v in views if v.get("visible") and on_screen(v, w, h)]
    if len(vis) < MIN_NODES:
        return None
    if any((v.get("resource_id") in DIALOG_IDS) or ("Dialog" in (v.get("class") or "")) or ("Popup" in (v.get("class") or "")) for v in vis):
        return None
    cls = [basename(v.get("class")) for v in vis]
    checkable = [v for v in vis if v.get("checkable")]
    toggles = [v for v, c in zip(vis, cls) if v.get("checkable") and c in TOGGLE_CLASSES]
    # a control under an overlay (dropdown list, text panel, popup) is in the hierarchy but not on the screen
    edits = [v for v, c in zip(vis, cls) if v.get("editable") or c.endswith("EditText")]
    if any(occluded(v, views, w, h) for v in toggles + edits):
        return None
    other_checkable = len(checkable) - len(toggles)
    webview = any(c == "WebView" or "WebView" in c for c in cls)
    editable = any(v.get("editable") or c.endswith("EditText") for v, c in zip(vis, cls))
    hinty = any(any(s in str(v.get(k) or "").lower() for k in ("text", "resource_id", "content_description") for s in ("search", "hint"))
                for v in vis)
    return {
        "toggle_state": (None if not toggles or other_checkable else
                         "on" if all(v.get("checked") for v in toggles) else
                         "off" if not any(v.get("checked") for v in toggles) else "mixed"),
        "input": True if editable else (False if not webview and not hinty else None),
        "package": (j.get("foreground_activity") or "/").split("/")[0],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="evals/data/image")
    ap.add_argument("--row-groups", type=int, default=40, help="row groups sampled across the 4 parquets")
    ap.add_argument("--per-set", type=int, default=400, help="target records per set (ui_toggle counts screens x 2)")
    ap.add_argument("--per-package", type=int, default=3, help="screens per app package per set")
    ap.add_argument("--limit", type=int, default=0, help="records per set cap; 0 = --per-set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=str(IMAGE_MANIFEST))
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi, HfFileSystem
    from PIL import Image
    import pyarrow.parquet as pq
    rev = HfApi().dataset_info(HF_ID, revision=REVISION).sha
    fs = HfFileSystem()
    rng = random.Random(a.seed)
    targets = {"toggle": a.per_set // 2, "input": a.per_set}
    if a.limit:
        targets = {k: min(v, a.limit if k != "toggle" else a.limit // 2) for k, v in targets.items()}
    ns = argparse.Namespace
    writers = {n: ImageSetWriter(n, ns(out=f"{a.out_dir}/{n}.jsonl", limit=0, images_dir=f"{a.out_dir}/../images/ui_state", manifest=a.manifest))
               for n in ("ui_toggle", "ui_input")}
    # candidate pools: (set, label) -> list of screens; balanced draws at the end
    pools: dict[tuple, list] = {("toggle", "on"): [], ("toggle", "off"): [], ("input", True): [], ("input", False): []}
    used_rg = []
    hashes: dict[str, list[int]] = {}
    per_pkg: dict[tuple, int] = {}
    # enumerate row groups
    files = []
    for name in PARQUETS:
        path = f"datasets/{HF_ID}@{rev}/{PREFIX}/{name}"
        with fs.open(path, "rb") as f:
            files.append((name, path, pq.ParquetFile(f).metadata.num_row_groups))
    all_rg = [(name, path, i) for name, path, n in files for i in range(n)]
    rng.shuffle(all_rg)
    scanned = 0
    for name, path, rg in all_rg:
        if all(len(pools[k]) >= targets[k[0]] for k in pools):
            break
        if len(used_rg) >= a.row_groups:
            break
        used_rg.append(f"{name}#{rg}")
        with fs.open(path, "rb", block_size=8 * 1024 * 1024) as f:
            rows = pq.ParquetFile(f).read_row_group(rg).to_pylist()
        order = list(range(len(rows)))
        rng.shuffle(order)
        for ri in order:
            r = rows[ri]
            scanned += 1
            try:
                j = json.loads(r["json_content"])
            except Exception:
                continue
            info = analyse(j)
            if info is None:
                continue
            pkg = info["package"]
            try:
                im = Image.open(io.BytesIO(r["image_content"]))
                hsh = dhash(im)
            except Exception:
                continue
            if any(bin(hsh ^ x).count("1") <= 2 for x in hashes.get(pkg, [])):
                continue  # near-duplicate of a kept screen from the same app
            sid = f"{name.split('.')[0]}-{rg}-{ri}"
            screen = {"sid": sid, "pkg": pkg, "bytes": r["image_content"], "info": info, "rg": f"{name}#{rg}", "row": ri}
            placed = False
            for key in ((("toggle", info["toggle_state"]),) if info["toggle_state"] in ("on", "off") else ()) + \
                       ((("input", info["input"]),) if info["input"] is not None else ()):
                if len(pools[key]) >= targets[key[0]] or per_pkg.get((key[0], pkg), 0) >= a.per_package:
                    continue
                pools[key].append(screen)
                per_pkg[(key[0], pkg)] = per_pkg.get((key[0], pkg), 0) + 1
                placed = True
            if placed:
                hashes.setdefault(pkg, []).append(hsh)
    print(f"scanned {scanned} screens in {len(used_rg)} row groups; pools: " + ", ".join(f"{k}={len(v)}" for k, v in pools.items()))

    def rec(w, set_name, s, q_key, target, extra):
        st = w.image({"bytes": s["bytes"], "path": None}, s["sid"])
        return {"id": f"{set_name}/mv/{s['sid']}{extra.get('suffix', '')}", "source": set_name, "split": "mv", "license": "commercial-ok",
                "lang": "en", "state": st, "questions": {"q1": {"type": "noul", "instructions": Q[q_key]}},
                "targets": {"q1": 1.0 if target else 0.0},
                "meta": {"grid_id": f"ui_state/{s['sid']}", "cluster": s["pkg"], "package": s["pkg"], "row_group": s["rg"], "row": s["row"],
                         "orig_answer_type": "rule", **{k: v for k, v in extra.items() if k != "suffix"}}}

    # ui_toggle: equal numbers of all-on and all-off screens, two records each
    k = min(len(pools[("toggle", "on")]), len(pools[("toggle", "off")]))
    w = writers["ui_toggle"]
    for state in ("on", "off"):
        for s in pools[("toggle", state)][:k]:
            for q_key in ("on", "off"):
                w.add(rec(w, "ui_toggle", s, q_key, q_key == state,
                          {"suffix": f"-{q_key}", "flip_group": s["sid"], "flip_kind": "toggle", "toggle_state": state}))
    for set_name, pool_key in (("ui_input", "input"),):
        w = writers[set_name]
        k = min(len(pools[(pool_key, True)]), len(pools[(pool_key, False)]))
        for label in (True, False):
            for s in pools[(pool_key, label)][:k]:
                w.add(rec(w, set_name, s, pool_key, label, {}))
    notes = (f"Rule labels from the view hierarchy (visible + on-screen + non-empty bounds; >= {MIN_NODES} visible nodes; "
             f"screens with AlertDialog ids or Dialog/Popup classes skipped; per-app near-duplicates removed by dHash<=2; "
             f"<= {a.per_package} screens per app per set; classes balanced 50/50). Toggle whitelist {sorted(TOGGLE_CLASSES)}; "
             f"screens with other checkables (tabs, radio, CheckedTextView) skipped; screens where a toggle or input box is "
             f"covered >=50% by a later-drawn leaf (dropdown, overlay) skipped. Input negatives exclude WebView screens "
             f"and search/hint text. seed={a.seed}; row groups: {', '.join(used_rg)}. "
             f"MIT; screenshots belong to the apps.")
    for w in writers.values():
        w.finish({"hf_id": HF_ID, "hf_config": PREFIX, "hf_split": "parquet", "hf_revision": rev, "n_records_full": None,
                  "license": "commercial-ok", "lang": "en", "notes": notes})


if __name__ == "__main__":
    main()
