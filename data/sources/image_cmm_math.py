"""CMM-Math (ecnu-icalk/cmm-math) train split: single-answer multiple-choice -> choice A/B/C/D, lang=zh.
Fill-in / open answers are skipped. Text-only questions are kept (state without image; meta.has_image).

Source: train_data.jsonl (20 MB, ~22k rows) streamed line by line in file order, stopping at --limit; images range-read one by one from
images.zip (278 MB, central directory cached). Options come from the "options" field ("A. ...\\nB. ..."); images
named in "image" go into state (one image: state["image"]; several: "<image n>" keys in listed order).

python -m data.sources.image_cmm_math --out data/raw/image/cmm_math.jsonl --limit 30
"""
import json

from data.sources._common import RawWriter, image_from_bytes, one_hot, record, save_image, source_args
from data.sources._ranged import RemoteZip, hf_url, stream
from data.sources.image_multi_benchmark import split_options

NAME, REPO = "cmm_math", "ecnu-icalk/cmm-math"
REV = "5b227e818a47ebfd0090b4fdcaac38cbf8cb7aa3"
INSTR = "回答数学选择题：选出唯一正确的选项。"


def lines(url):
    buf = b""
    for chunk in stream(url):
        buf += chunk
        *done, buf = buf.split(b"\n")
        for ln in done:
            if ln.strip():
                yield json.loads(ln)
    if buf.strip():
        yield json.loads(buf)


def main():
    args = source_args(NAME)
    w = RawWriter(NAME, args.out)
    z = None
    for row in lines(hf_url(REPO, "train_data.jsonl", REV)):
        if args.limit and w.n >= args.limit:
            break
        parsed = split_options(row.get("options") or "")
        answer = str(row.get("answer") or "").strip()
        if parsed is None:
            w.skip("no_options")
            continue
        _, crit = parsed
        if answer not in crit:
            w.skip("answer_not_single_option")
            continue
        state = {"question_context": row["question"].replace("<ImageHere>", "").strip()}  # inline placeholder of the source
        images = row.get("image") or []
        if images:
            z = z or RemoteZip([hf_url(REPO, "images.zip", REV)])
            by_base = getattr(z, "_by_base", None) or {n.rsplit("/", 1)[-1]: n for n in z.names()}
            z._by_base = by_base
            members = [by_base.get(im.rsplit("/", 1)[-1]) for im in images]
            if None in members:
                w.skip("image_missing_in_zip")
                continue
            for n, m in enumerate(members, 1):
                path = save_image(image_from_bytes(z.read(m)), NAME, m.rsplit("/", 1)[-1].rsplit(".", 1)[0])
                if len(members) == 1:
                    state["image"] = {"path": path}
                else:
                    state[f"<image {n}>"] = {"image": {"path": path}}
        w.write(record(NAME, w.n, state, {"q1": {"type": "choice", "instructions": INSTR, "criteria": crit}},
                       {"q1": one_hot(crit, answer)}, lang="zh", grid_id=f"{NAME}/{row['id']}", source_split="train",
                       meta={"orig_id": row["id"], "has_image": bool(images), "level": row.get("level"),
                             "subject": row.get("subject")}))
    w.close()


if __name__ == "__main__":
    main()
