import json
import math
from pathlib import Path
from typing import Any, Iterator

QTYPES = ("choice", "noul", "score")
LICENSES = ("commercial-ok", "non-commercial", "unknown")
SUM_TOL = 1e-6


def iter_records(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e


def _is_prob(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0


def _is_image_value(v: Any) -> bool:
    return isinstance(v, dict) and bool({"path", "url"} & set(v))


def _check_images(node: Any, where: str, errors: list[str]) -> None:
    # an "image" key whose value is an object with path/url is an image object, whether it is
    # the reserved single-key shape (spec 4) or a key next to others (README example)
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "image" and (_is_image_value(v) or (set(node) == {"image"} and not isinstance(v, str))):
                if not isinstance(v, dict) or len(v) != 1 or not set(v) <= {"path", "url"}:
                    errors.append(f"{where}.image: must be an object with exactly one of 'path' or 'url'")
                elif not isinstance(next(iter(v.values())), str) or not next(iter(v.values())):
                    errors.append(f"{where}.image: value must be a non-empty string")
            else:
                _check_images(v, f"{where}.{k}", errors)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _check_images(v, f"{where}[{i}]", errors)


def _check_question(name: str, q: Any, errors: list[str]) -> None:
    where = f"questions.{name}"
    if not isinstance(q, dict):
        errors.append(f"{where}: must be an object")
        return
    qtype = q.get("type")
    if qtype not in QTYPES:
        errors.append(f"{where}.type: {qtype!r} not in {QTYPES}")
        return
    if "instructions" in q and not isinstance(q["instructions"], str):
        errors.append(f"{where}.instructions: must be a string")
    crit = q.get("criteria")
    if qtype == "choice":
        if not isinstance(crit, dict) or not 1 <= len(crit) <= 255:
            errors.append(f"{where}.criteria: choice needs an object with 1..255 labels")
    elif qtype == "score":
        if not isinstance(crit, list) or not 1 <= len(crit) <= 255:
            errors.append(f"{where}.criteria: score needs a list with 1..255 levels")
    elif crit is not None and (not isinstance(crit, dict) or not set(crit) <= {"true", "false"}):
        errors.append(f"{where}.criteria: noul criteria may only have 'true'/'false'")


def _check_dist(where: str, dist: Any, keys: set[str], errors: list[str]) -> None:
    if not isinstance(dist, dict):
        errors.append(f"{where}: must be an object of probabilities")
        return
    if set(dist) != keys:
        errors.append(f"{where}: keys {sorted(dist)} != expected {sorted(keys)}")
    bad = [k for k, v in dist.items() if not _is_prob(v)]
    if bad:
        errors.append(f"{where}: values out of [0,1] for {bad}")
        return
    total = sum(dist.values())
    if abs(total - 1.0) > SUM_TOL:
        errors.append(f"{where}: probabilities sum to {total}, not 1")


def validate_record(rec: dict) -> list[str]:
    errors: list[str] = []
    if not isinstance(rec, dict):
        return ["record must be a JSON object"]
    for key in ("id", "source", "split", "lang"):
        if not isinstance(rec.get(key), str) or not rec.get(key):
            errors.append(f"{key}: required non-empty string")
    if rec.get("license") not in LICENSES:
        errors.append(f"license: {rec.get('license')!r} not in {LICENSES}")
    if "state" not in rec or rec["state"] is None:
        errors.append("state: required and must not be null")
    else:
        _check_images(rec["state"], "state", errors)
    if "meta" in rec and not isinstance(rec["meta"], dict):
        errors.append("meta: must be an object")

    questions = rec.get("questions")
    if not isinstance(questions, dict) or not questions:
        errors.append("questions: required non-empty object")
        return errors
    for name, q in questions.items():
        _check_question(name, q, errors)

    targets = rec.get("targets")
    if not isinstance(targets, dict):
        errors.append("targets: required object")
        return errors
    if set(targets) != set(questions):
        errors.append(f"targets: keys {sorted(targets)} != questions keys {sorted(questions)}")
    for name, q in questions.items():
        if name not in targets or not isinstance(q, dict):
            continue
        t, where = targets[name], f"targets.{name}"
        qtype = q.get("type")
        if qtype == "noul":
            if not _is_prob(t):
                errors.append(f"{where}: noul target must be a float in [0,1], got {t!r}")
        elif qtype == "choice" and isinstance(q.get("criteria"), dict):
            _check_dist(where, t, set(q["criteria"]), errors)
        elif qtype == "score" and isinstance(q.get("criteria"), list):
            _check_dist(where, t, {str(i) for i in range(len(q["criteria"]))}, errors)

    negated = rec["meta"].get("negated_questions") if isinstance(rec.get("meta"), dict) else None
    if negated is not None:
        if not isinstance(negated, dict):
            errors.append("meta.negated_questions: must be an object")
        else:
            for name, q in negated.items():
                if name not in questions:
                    errors.append(f"meta.negated_questions.{name}: no such question")
                elif not isinstance(q, dict) or q.get("type") != "noul" or not isinstance(questions[name], dict) or questions[name].get("type") != "noul":
                    errors.append(f"meta.negated_questions.{name}: original and negation must both be noul")
                else:
                    _check_question(name, q, errors)
    return errors
