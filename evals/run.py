import random
import argparse
import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

from evals.client import SystemOneClient, SystemOneError
from evals.metrics import Row, compute_metrics
from evals.schema import iter_records, validate_record

PERTURBATIONS = ("order", "separate", "repeat", "negation", "jitter")


class BadAnswer(ValueError):
    pass


def reverse_question(q: dict) -> dict:
    q = dict(q)
    if q["type"] == "choice":
        q["criteria"] = {k: q["criteria"][k] for k in reversed(list(q["criteria"]))}
    elif q["type"] == "score":
        q["criteria"] = list(reversed(q["criteria"]))
    return q


def parse_answer(q: dict, answer: Any, reversed_order: bool = False) -> tuple[Any, float]:
    qtype = q["type"]
    if not isinstance(answer, dict) or answer.get("type") != qtype:
        raise BadAnswer(f"expected a {qtype} answer, got {str(answer)[:200]}")
    if qtype == "noul":
        p = answer.get("noul")
        if not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            raise BadAnswer(f"noul={p!r}")
        return float(p), max(p, 1.0 - p)
    probs, conf = answer.get("probabilities"), answer.get("confidence")
    if not isinstance(probs, dict) or not isinstance(conf, (int, float)):
        raise BadAnswer("missing probabilities/confidence")
    if qtype == "choice":
        if set(probs) != set(q["criteria"]):
            raise BadAnswer(f"probability keys {sorted(probs)} != criteria {sorted(q['criteria'])}")
        return {k: float(probs[k]) for k in q["criteria"]}, float(conf)
    n = len(q["criteria"])
    if set(probs) != {str(i) for i in range(n)}:
        raise BadAnswer(f"score probability keys {sorted(probs)} != 0..{n - 1}")
    if reversed_order:
        # level i of the reversed request is level n-1-i of the original
        return {str(j): float(probs[str(n - 1 - j)]) for j in range(n)}, float(conf)
    return {str(i): float(probs[str(i)]) for i in range(n)}, float(conf)


def labels_of(q: dict) -> list[str]:
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["true", "false"]


def strip_images(state: Any) -> Any:
    """Replace every image object in a state tree with a placeholder string (blind baseline)."""
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if k == "image" and isinstance(v, dict) and ("path" in v or "url" in v):
                out[k] = "[image omitted]"
            else:
                out[k] = strip_images(v)
        return out
    if isinstance(state, list):
        return [strip_images(v) for v in state]
    return state


def _is_image_obj(k: str, v: Any) -> bool:
    return k == "image" and isinstance(v, dict) and ("path" in v or "url" in v)


def _collect_images(state: Any, acc: list) -> list:
    if isinstance(state, dict):
        for k, v in state.items():
            if _is_image_obj(k, v):
                acc.append(v)
            else:
                _collect_images(v, acc)
    elif isinstance(state, list):
        for v in state:
            _collect_images(v, acc)
    return acc


def _replace_images(state: Any, it) -> Any:
    if isinstance(state, dict):
        return {k: (next(it) if _is_image_obj(k, v) else _replace_images(v, it)) for k, v in state.items()}
    if isinstance(state, list):
        return [_replace_images(v, it) for v in state]
    return state


def shuffle_images(records: list[dict], seed: int, group_field: str | None = None) -> tuple[list[dict], int]:
    """Mismatch control: every record gets the images of a different record that has the same number of images
    (a seeded derangement over distinct image sets, so records sharing an image never get it back). With
    group_field, donors are drawn only among records with the same meta[group_field] (e.g. the same prompt).
    Text and questions stay; ids stay, so results pair with the unshuffled run; meta.shuffle_donor names the
    record whose images were used. Returns (records, n_shuffled)."""
    rng = random.Random(seed)
    imgs = [_collect_images(r["state"], []) for r in records]
    grp = [((r.get("meta") or {}).get(group_field) if group_field else None) for r in records]
    keys = [tuple([json.dumps(v, sort_keys=True) for v in im] + ([json.dumps(g)] if group_field else [])) if im else ()
            for im, g in zip(imgs, grp)]
    groups: dict[tuple, list[tuple]] = {}
    for k in dict.fromkeys(keys):
        if k:
            groups.setdefault((len(k), k[-1] if group_field else None), []).append(k)
    donor: dict[tuple, tuple] = {}
    for ks in groups.values():
        if len(ks) < 2:
            continue
        order = ks[:]
        rng.shuffle(order)
        shift = rng.randrange(1, len(order))
        donor.update(zip(order, order[shift:] + order[:shift]))
    by_key = {k: (im, r["id"]) for k, im, r in zip(keys, imgs, records)}
    out, n = [], 0
    for r, k in zip(records, keys):
        if k in donor:
            im, did = by_key[donor[k]]
            out.append({**r, "state": _replace_images(r["state"], iter(im)),
                        "meta": {**(r.get("meta") or {}), "shuffle_donor": did}})
            n += 1
        else:
            out.append(r)
    return out, n


def has_image(state: Any) -> bool:
    if isinstance(state, dict):
        return any(_is_image_obj(k, v) or has_image(v) for k, v in state.items())
    if isinstance(state, list):
        return any(has_image(v) for v in state)
    return False


def _jitter_bytes(raw: bytes, rng: random.Random) -> str:
    """A near-duplicate frame: crop up to 3% off the border, resize back, brightness +-5%, faint noise."""
    import base64
    import io

    from PIL import Image, ImageEnhance

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    dx, dy = int(w * rng.uniform(0.0, 0.03)), int(h * rng.uniform(0.0, 0.03))
    ox, oy = rng.randint(0, dx) if dx else 0, rng.randint(0, dy) if dy else 0
    img = img.crop((ox, oy, w - (dx - ox), h - (dy - oy))).resize((w, h), Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.95, 1.05))
    if rng.random() < 0.5:
        from PIL import ImageFilter

        img = img.filter(ImageFilter.GaussianBlur(0.5))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def jitter_state(state: Any, base_dir: Path, rng: random.Random) -> Any:
    """Replace every image in the state with a slightly perturbed copy (inlined as a data URL)."""
    if isinstance(state, dict):
        out = {}
        for k, v in state.items():
            if _is_image_obj(k, v):
                if "path" in v:
                    p = Path(v["path"])
                    raw = (p if p.is_absolute() else base_dir / p).read_bytes()
                else:
                    import base64

                    raw = base64.b64decode(v["url"].split(",", 1)[1])
                out[k] = {"url": _jitter_bytes(raw, rng)}
            else:
                out[k] = jitter_state(v, base_dir, rng)
        return out
    if isinstance(state, list):
        return [jitter_state(v, base_dir, rng) for v in state]
    return state


def build_jobs(records: list[dict], perturb: set[str], base_dir: Path | None = None, seed: int = 0,
               one_per_request: bool = False) -> list[tuple[str, int, str | None, dict, Any]]:
    """Each job is (kind, record index, question name, questions, state override or None). With one_per_request the
    main answers come from one request per question ("main_part" jobs, merged per record)."""
    jobs = []
    rng = random.Random(seed)
    for ri, rec in enumerate(records):
        qs = rec["questions"]
        if one_per_request and len(qs) > 1:
            for name, q in qs.items():
                jobs.append(("main_part", ri, name, {name: q}, None))
        else:
            jobs.append(("main", ri, None, qs, None))
        if "order" in perturb and any(q["type"] in ("choice", "score") for q in qs.values()):
            jobs.append(("order", ri, None, {k: reverse_question(q) for k, q in qs.items()}, None))
        if "separate" in perturb and len(qs) > 1:
            for name, q in qs.items():
                jobs.append(("separate", ri, name, {name: q}, None))
        if "repeat" in perturb:
            jobs.append(("repeat", ri, None, qs, None))
        negated = (rec.get("meta") or {}).get("negated_questions")
        if "negation" in perturb and negated:
            jobs.append(("negation", ri, None, negated, None))
        if "jitter" in perturb and has_image(rec["state"]):
            jobs.append(("jitter", ri, None, qs, jitter_state(rec["state"], base_dir or Path("."), rng)))
    return jobs


def run(args: argparse.Namespace) -> dict:
    records_path = Path(args.records)
    records = list(iter_records(records_path))
    if args.limit > 0:
        records = records[: args.limit]
    if args.sample > 0 and len(records) > args.sample:
        # Seeded random subset, original order kept so runs on the same file are comparable.
        keep = sorted(random.Random(args.seed).sample(range(len(records)), args.sample))
        records = [records[i] for i in keep]
    if args.strip_images:
        records = [{**rec, "state": strip_images(rec["state"])} for rec in records]
    if args.shuffle_images:
        records, n_sh = shuffle_images(records, args.seed, args.shuffle_group or None)
        print(f"[run] shuffle-images: {n_sh}/{len(records)} records got another record's images", file=sys.stderr)
    invalid = [(rec.get("id", f"#{i}"), errs) for i, rec in enumerate(records) if (errs := validate_record(rec))]
    if invalid:
        for rid, errs in invalid[:10]:
            print(f"[run] invalid record {rid}: {errs}", file=sys.stderr)
        raise SystemExit(f"[run] {len(invalid)} invalid records, aborting")

    perturb = {p.strip() for p in args.perturb.split(",") if p.strip()}
    unknown = perturb - set(PERTURBATIONS)
    if unknown:
        raise SystemExit(f"[run] unknown --perturb values: {sorted(unknown)}")

    jobs = build_jobs(records, perturb, base_dir=records_path.parent, seed=args.seed,
                      one_per_request=args.one_question_per_request)
    total, done, lock = len(jobs), [0], threading.Lock()
    step = max(1, total // 20)

    def on_done(_i: int, _res: Any) -> None:
        with lock:
            done[0] += 1
            if done[0] % step == 0 or done[0] == total:
                print(f"[run] {done[0]}/{total} requests", file=sys.stderr, flush=True)

    print(f"[run] {len(records)} records, {total} requests, perturb={sorted(perturb)}", file=sys.stderr)
    with SystemOneClient(args.base_url, args.api_key, args.model, timeout=args.timeout,
                         max_retries=args.max_retries, base_dir=records_path.parent,
                         pool_size=args.concurrency) as client:
        results = client.ask_many([(st if st is not None else records[ri]["state"], qs) for _, ri, _, qs, st in jobs],
                                  args.concurrency, on_done)

    # group responses by record
    by_record: list[dict[str, Any]] = [{"separate": {}} for _ in records]
    for (kind, ri, name, qs, _st), res in zip(jobs, results):
        if kind == "separate":
            by_record[ri]["separate"][name] = (qs, res)
        elif kind == "main_part":
            parts = by_record[ri].setdefault("main_parts", [])
            parts.append(res)
            if len(parts) == len(records[ri]["questions"]):
                bad = next((r for r in parts if isinstance(r, Exception)), None)
                merged = bad or {**parts[0], "answers": {k: v for r in parts for k, v in r.get("answers", {}).items()},
                                 "latency_ms": sum(r["latency_ms"] for r in parts)}
                by_record[ri]["main"] = (records[ri]["questions"], merged)
        else:
            by_record[ri][kind] = (qs, res)

    rows: list[Row] = []
    raw: list[dict] = []
    errors: list[dict] = []
    main_latencies: list[float] = []

    def note_error(rid: str, kind: str, qname: str | None, err: Exception) -> dict:
        if isinstance(err, SystemOneError):
            e = {"id": rid, "kind": kind, "question": qname, **err.to_dict()}
        else:
            e = {"id": rid, "kind": kind, "question": qname, "status": "bad_answer", "body": str(err)}
        errors.append(e)
        return {"status": e["status"], "body": e["body"]}

    for ri, rec in enumerate(records):
        rid, qs, got = rec["id"], rec["questions"], by_record[ri]
        entry: dict[str, Any] = {"id": rid, "questions": {}}
        if (rec.get("meta") or {}).get("shuffle_donor"):
            entry["shuffle_donor"] = rec["meta"]["shuffle_donor"]
        raw.append(entry)
        _, main = got["main"]
        if isinstance(main, Exception):
            entry["error"] = note_error(rid, "main", None, main)
            continue
        entry["latency_ms"] = main["latency_ms"]
        entry["model"] = main.get("model")
        main_latencies.append(main["latency_ms"])

        # perturbation responses at the request level
        perturbed: dict[str, dict] = {}
        for kind in ("order", "repeat", "negation", "jitter"):
            if kind not in got:
                continue
            pqs, res = got[kind]
            if isinstance(res, Exception):
                entry.setdefault("perturb_errors", {})[kind] = note_error(rid, kind, None, res)
            else:
                entry.setdefault("perturb_latency_ms", {})[kind] = res["latency_ms"]
                perturbed[kind] = res
        for name, (pqs, res) in got["separate"].items():
            if isinstance(res, Exception):
                entry.setdefault("perturb_errors", {}).setdefault("separate", {})[name] = note_error(rid, "separate", name, res)
            else:
                entry.setdefault("perturb_latency_ms", {}).setdefault("separate", {})[name] = res["latency_ms"]

        for name, q in qs.items():
            qraw: dict[str, Any] = {"type": q["type"]}
            entry["questions"][name] = qraw
            try:
                pred, conf = parse_answer(q, main.get("answers", {}).get(name))
            except BadAnswer as e:
                qraw["error"] = note_error(rid, "main", name, e)
                continue
            qraw.update({"pred": pred, "confidence": conf})
            row = Row(record_id=rid, qname=name, qtype=q["type"], pred=pred, target=rec["targets"][name],
                      confidence=conf, latency_ms=main["latency_ms"], labels=labels_of(q))

            def try_parse(kind: str, answer: Any, pq: dict, reversed_order: bool = False) -> Any:
                try:
                    return parse_answer(pq, answer, reversed_order)[0]
                except BadAnswer as e:
                    qraw.setdefault("perturb_errors", {})[kind] = note_error(rid, kind, name, e)
                    return None

            if "order" in perturbed and q["type"] in ("choice", "score"):
                row.pred_reversed = try_parse("order", perturbed["order"].get("answers", {}).get(name),
                                              got["order"][0][name], reversed_order=True)
                qraw["order"] = row.pred_reversed
            if "repeat" in perturbed:
                row.pred_repeat = try_parse("repeat", perturbed["repeat"].get("answers", {}).get(name), q)
                qraw["repeat"] = row.pred_repeat
            if "jitter" in perturbed:
                row.pred_jitter = try_parse("jitter", perturbed["jitter"].get("answers", {}).get(name), q)
                qraw["jitter"] = row.pred_jitter
            if name in got["separate"] and not isinstance(got["separate"][name][1], Exception):
                row.pred_separate = try_parse("separate", got["separate"][name][1].get("answers", {}).get(name), q)
                qraw["separate"] = row.pred_separate
            if "negation" in perturbed and name in got["negation"][0]:
                row.pred_negated = try_parse("negation", perturbed["negation"].get("answers", {}).get(name),
                                             got["negation"][0][name])
                qraw["negation"] = row.pred_negated
            rows.append(row)

    return {
        "target": args.target,
        "base_url": args.base_url,
        "model": args.model,
        "records_file": str(records_path),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "n": len(records),
        "perturb": sorted(perturb),
        "metrics": compute_metrics(rows, request_latencies=main_latencies, errors=errors),
        "raw": raw,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m evals.run")
    ap.add_argument("--records", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", default="jev-latest")
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perturb", default="", help="comma-separated subset of " + ",".join(PERTURBATIONS))
    ap.add_argument("--one-question-per-request", action="store_true",
                    help="ask each question of a record in its own request (reproduces results of servers that did)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    ap.add_argument("--sample", type=int, default=0, help="seeded random subset of this size after --limit; 0 = off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strip-images", action="store_true", help="blind baseline: replace image objects with a placeholder")
    ap.add_argument("--shuffle-images", action="store_true",
                    help="mismatch control: give every record the images of another record with the same image count")
    ap.add_argument("--shuffle-group", default="", help="with --shuffle-images: only swap within records sharing this meta field")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=4)
    args = ap.parse_args(argv)

    result = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    m = result["metrics"]
    print(f"[run] wrote {out}: n={result['n']} accuracy={m['accuracy']} ece={m['ece']} "
          f"errors={m['errors']['count']}", file=sys.stderr)


if __name__ == "__main__":
    main()
