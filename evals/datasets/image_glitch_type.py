"""GlitchBench (607 glitch screenshots from Reddit / Unity, MIT) -> glitch_type: a 4-way choice on the kind of glitch.
Diagnostic set: every image is a glitch, so "is there a glitch" cannot be asked; classes are
433 / 83 / 70 / 21, so read macro-F1 and the confusion matrix, not accuracy; the class descriptions are ours.

python -m evals.datasets.image_glitch_type --out evals/data/image/glitch_type.jsonl --limit 20

Ids derive from the upstream id; meta.cluster = game."""

from __future__ import annotations

from evals.datasets._common import ImageSetWriter, hf_stream, image_args, one_hot

NAME = "glitch_type"
HF_ID, SPLIT = "glitchbench/GlitchBench", "validation"
REVISION = "main"
N_FULL = 607
INSTRUCTIONS = "This video game screenshot contains a glitch. Which kind of glitch is it?"
CLASSES = {
    "physics": "Physics, collision or spawn issue: objects or characters clipping through geometry, floating, stuck, "
               "launched, or spawned in the wrong place",
    "animation": "Animation or pose error: a body or face in an unnatural pose (T-pose, twisted limbs), frozen or broken animation",
    "rendering": "Rendering or texture issue: missing or stretched textures, flickering, z-fighting, wrong shading, broken models",
    "camera_ui": "Camera, user interface or lighting issue: camera inside geometry or at a wrong angle, broken HUD or menus, wrong lighting",
}
UPSTREAM = {"Physics, Collision, and Spawn": "physics", "Animation and Pose": "animation",
            "Rendering and Texture": "rendering", "Camera, User Interface and Lighting": "camera_ui"}


def norm_type(s: str) -> str | None:
    s = s.strip()
    for suffix in (" Issues", " Errors"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return UPSTREAM.get(s)


def main():
    from huggingface_hub import HfApi
    args = image_args()
    rev = HfApi().dataset_info(HF_ID, revision=REVISION).sha
    w = ImageSetWriter(NAME, args)
    for row in hf_stream(HF_ID, None, SPLIT, rev, image_cols=["image"]):
        if w.full:
            break
        cls = norm_type(row["glitch-type"] or "")
        if cls is None:
            w.skip("unknown_type")
            continue
        if row.get("image") is None:
            w.skip("no_image")
            continue
        uid = str(row["id"])
        w.add({"id": f"{NAME}/{SPLIT}/{uid}", "source": NAME, "split": SPLIT, "license": "commercial-ok", "lang": "en",
               "state": w.image(row["image"], uid),
               "questions": {"q1": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": dict(CLASSES)}},
               "targets": {"q1": one_hot(CLASSES, cls)},
               "meta": {"grid_id": f"{NAME}/{uid}", "cluster": str(row.get("game") or "N/A"), "game": row.get("game"),
                        "upstream_type": row["glitch-type"], "image_source": row.get("source"), "orig_answer_type": "class"}})
    w.finish({"hf_id": HF_ID, "hf_config": None, "hf_split": SPLIT, "hf_revision": rev, "n_records_full": N_FULL,
              "license": "commercial-ok", "lang": "en",
              "notes": "4-way glitch-type choice; upstream type strings with/without 'Issues'/'Errors' merged. Class "
                       "descriptions written by us (not upstream). Imbalanced 433/83/70/21: diagnostic only, read macro-F1. "
                       "Every image is a glitch (no clean frames). MIT; screenshots belong to the games' publishers."})


if __name__ == "__main__":
    main()
