"""MULTI-Benchmark (OpenDFM): single-answer Chinese exam MCQs (单选) -> choice A/B/C/..., lang=zh.
Questions with and without images are both converted (no image = text-only state; meta.has_image).

Source: HF OpenDFM/MULTI-Benchmark, one 504 MB zip. The run range-reads the central directory, the full problem
file problem_v1.3.1_20241210.json (4.9 MB compressed; the *_release.json variant has no answers) and only the
images of the chosen questions. Skipped: every other question type (多选 / 填空 / 解答 / 其他), multi-part problems (problem_type_list > 1),
options not parseable as "A. ..." lines, answers not a single option letter.
[IMAGE_n] tags become "<image n>" and the images sit in state under those keys (single image: state["image"]).

python -m data.sources.image_multi_benchmark --out data/raw/image/multi_benchmark.jsonl --limit 30
"""
import json
import random
import re

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import CACHE, RemoteZip, hf_url

NAME, REPO = "multi_benchmark", "OpenDFM/MULTI-Benchmark"
REV = "45286aa9c0ed9cac80d0435d6a371f58c460cf83"
ZIP, PROBLEMS = "MULTI_v1.3.1_20251016_release.zip", "problem_v1.3.1_20241210.json"
INSTR = "回答选择题：选出唯一正确的选项。"
OPT = re.compile(r"^\s*([A-H])[\.．、:：]\s*(.*)$")
IMG = re.compile(r"\[IMAGE_(\d+)\]")


def split_options(text: str):
    """Trailing "A. ..." lines -> (stem, {letter: text}); None when the letters are not A, B, C, ... in order."""
    lines = text.split("\n")
    opts, i = {}, len(lines)
    while i > 0:
        m = OPT.match(lines[i - 1])
        if not m:
            break
        opts[m.group(1)] = m.group(2).strip()
        i -= 1
    letters = sorted(opts)
    if len(letters) < 2 or letters != [chr(ord("A") + k) for k in range(len(letters))]:
        return None
    return "\n".join(lines[:i]).strip(), {k: opts[k] for k in letters}


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    z = RemoteZip([hf_url(REPO, ZIP, REV)])
    cached = CACHE / f"multi_{PROBLEMS}"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(z.read(PROBLEMS))
    problems = list(json.loads(cached.read_text(encoding="utf-8")).values())
    random.Random(args.seed).shuffle(problems)
    for p in problems:
        if args.limit and w.n >= args.limit:
            break
        types = p["problem_type_list"]
        if len(types) != 1 or types[0] != "单选":
            w.skip(f"type={'+'.join(types)}")
            continue
        answer = (p["problem_answer_list"] or [""])[0].strip()
        parsed = split_options(p["problem_content_list"][0])
        if parsed is None:
            w.skip("options_not_parsed")
            continue
        stem, crit = parsed
        if answer not in crit:
            w.skip("answer_not_single_option")
            continue
        imgs = {int(n): f"images/{p['img_paths'][int(n) - 1].split('/images/', 1)[1]}"
                for n in IMG.findall(p["problem_content_list"][0]) if 0 < int(n) <= len(p["img_paths"])}
        if len(imgs) != len(set(IMG.findall(p["problem_content_list"][0]))) or len(imgs) != len(p["img_paths"]):
            w.skip("image_tag_mismatch")
            continue
        if any(m not in z.entries for m in imgs.values()):
            w.skip("image_missing_in_zip")
            continue
        stem = stem.replace("[MASK]", "（　）")
        stem = IMG.sub(lambda m: "" if len(imgs) == 1 else f"<image {m.group(1)}>", stem).strip()
        crit = {k: IMG.sub(lambda m: f"<image {m.group(1)}>", v) for k, v in crit.items()}
        state = {"question_context": stem}
        for n, member in sorted(imgs.items()):
            path = save_image(image_from_bytes(z.read(member)), NAME, member[len("images/"):].rsplit(".", 1)[0])
            if len(imgs) == 1:
                state["image"] = {"path": path}
            else:
                state[f"<image {n}>"] = {"image": {"path": path}}
        w.write(record(NAME, w.n, state, {"q1": {"type": "choice", "instructions": INSTR, "criteria": crit}},
                       {"q1": one_hot(crit, answer)}, lang="zh", grid_id=f"{NAME}/{p['problem_id']}",
                       source_split="all",
                       meta={"problem_id": p["problem_id"], "has_image": bool(imgs), "education": p["education"],
                             "subject": p["subject"], "difficulty": p["difficulty"]}))
    w.close()


if __name__ == "__main__":
    main()
