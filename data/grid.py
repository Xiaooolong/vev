"""Gridding: group raw records by meta.grid_id, derive noul questions, sample question sets. See data/README.md."""
import json
import random
import re

P_DERIVE_CHOICE = 0.5
P_DERIVE_SCORE = 0.3
GEOM_P = 0.35
MAX_QUESTIONS = 16

TEMPLATES = {
    "choice": {"en": "Is the answer: {desc}?", "zh": "答案是否是：{desc}？"},
    "score": {"en": "Is it at least: {desc}?", "zh": "是否至少达到{desc}？"},
}


_CJK = re.compile(r"[一-鿿]")


def _lang_of(q: dict, fallback: str) -> str:
    """Template language follows the question text itself (a zh record can carry English questions and vice versa)."""
    text = (q.get("instructions") or "").strip()
    if _CJK.search(text):
        return "zh"
    return "en" if text else fallback


def _template(kind: str, lang: str) -> str:
    return TEMPLATES[kind]["zh" if lang == "zh" else "en"]


def _desc(value, label: str) -> str:
    """Option text for a derived question: the description, or the label when there is none (None / empty)."""
    if value is None:
        return label
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text.strip() or label


_TERMINAL = "?？.。!！:："


def _join(instructions: str, suffix: str, lang: str) -> str:
    instructions = (instructions or "").strip()
    if not instructions:
        return suffix
    if instructions[-1] not in _TERMINAL:
        instructions += "。" if lang == "zh" else "."
    return instructions + ("" if lang == "zh" else " ") + suffix


def _argmax_labels(dist: dict) -> list[str]:
    best = max(dist.values())
    return [k for k, v in dist.items() if v == best]


def derive_choice(q: dict, target: dict, lang: str, rng: random.Random) -> tuple[dict, float]:
    """noul 'is the answer X?' — half from the argmax label(s), half from the others; target = P(X)."""
    correct = _argmax_labels(target)
    wrong = [k for k in q["criteria"] if k not in correct]
    pick = rng.choice(correct if (rng.random() < 0.5 or not wrong) else wrong)
    desc = _desc(q["criteria"][pick], pick)
    lang = _lang_of(q, lang)
    suffix = _template("choice", lang).format(desc=desc)
    return {"type": "noul", "instructions": _join(q.get("instructions", ""), suffix, lang)}, float(target[pick])


def derive_score(q: dict, target: dict, lang: str, rng: random.Random) -> tuple[dict, float]:
    """noul 'at least level k?' with k in 1..n-1 (k=0 is always true); target = tail sum P(level >= k)."""
    n = len(q["criteria"])
    k = rng.randint(1, n - 1)
    desc = _desc(q["criteria"][k], str(k))
    lang = _lang_of(q, lang)
    suffix = _template("score", lang).format(desc=desc)
    p = min(1.0, max(0.0, sum(float(target[str(i)]) for i in range(k, n))))
    return {"type": "noul", "instructions": _join(q.get("instructions", ""), suffix, lang)}, p


def sample_size(rng: random.Random) -> int:
    """Geometric on {1,2,...} with success prob GEOM_P, clipped at MAX_QUESTIONS."""
    k = 1
    while k < MAX_QUESTIONS and rng.random() >= GEOM_P:
        k += 1
    return k


def _id_prefix(rec: dict) -> str:
    """Record ids must be unique across sources; two sources may share grid ids (vqav2_mc and vqav2_yesno both use
    'vqav2/<image>'), so the source name is prefixed unless the grid id already starts with it."""
    src = rec["source"]
    return "" if str(rec["meta"]["grid_id"]).startswith(src + "/") else src + "/"


def _group(records: list[dict]) -> list[list[dict]]:
    """Records sharing (source, grid_id, state). Same grid_id with a different state becomes its own group."""
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["source"], r["meta"]["grid_id"], json.dumps(r["state"], sort_keys=True, ensure_ascii=False))
        groups.setdefault(key, []).append(r)
    return list(groups.values())


def _shared_meta(members: list[dict]) -> dict:
    first = members[0]["meta"]
    skip = {"negated_questions", "derived", "aug"}
    return {k: v for k, v in first.items() if k not in skip and all(m["meta"].get(k) == v for m in members[1:])}


def build_grids(records: list[dict], rng: random.Random) -> list[dict]:
    out: list[dict] = []
    variants: dict[tuple, int] = {}
    for members in _group(records):
        head = members[0]
        lang = head["lang"]
        # pool of (question, target, negated-or-None, derived?, origin record id)
        pool = []
        for r in members:
            negs = r["meta"].get("negated_questions") or {}
            for name, q in r["questions"].items():
                t = r["targets"][name]
                pool.append((q, t, negs.get(name), False, r["id"]))
                if q["type"] == "choice" and len(q["criteria"]) >= 3 and rng.random() < P_DERIVE_CHOICE:
                    dq, dt = derive_choice(q, t, lang, rng)
                    pool.append((dq, dt, None, True, r["id"]))
                elif q["type"] == "score" and len(q["criteria"]) >= 2 and rng.random() < P_DERIVE_SCORE:
                    dq, dt = derive_score(q, t, lang, rng)
                    pool.append((dq, dt, None, True, r["id"]))
        rng.shuffle(pool)

        vkey = (head["source"], head["meta"]["grid_id"])
        v = variants.get(vkey, 0)
        variants[vkey] = v + 1
        meta = _shared_meta(members)
        meta["members"] = sorted({r["id"] for r in members})

        # partition the whole pool into chunks of geometric size, so every question is used once
        i = chunk = 0
        while i < len(pool):
            part = pool[i:i + sample_size(rng)]
            i += len(part)
            questions, targets, negated, derived = {}, {}, {}, []
            for j, (q, t, neg, is_derived, _) in enumerate(part, 1):
                name = f"q{j}"
                questions[name], targets[name] = q, t
                if neg is not None:
                    negated[name] = neg
                if is_derived:
                    derived.append(name)
            m = dict(meta, derived=derived)
            if negated:
                m["negated_questions"] = negated
            out.append({
                "id": f"{_id_prefix(head)}{head['meta']['grid_id']}#{v}.{chunk}", "source": head["source"], "split": head["split"],
                "license": head["license"], "lang": lang, "state": head["state"],
                "questions": questions, "targets": targets, "meta": m,
            })
            chunk += 1
    return out
