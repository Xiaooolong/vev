"""Open-answer VQA -> choice: answer cleanup, typed answer pools, distractors (data/README.md: "Open answers to choice")."""
import collections
import re

YES_NO = {"yes": 1.0, "no": 0.0}
KEYS = "ABCD"
NUM = re.compile(r"^(-?)(\d{1,3}(?:,\d{3})+|\d+)?(\.\d+)?(e[+-]\d+)?(%?)$", re.I)
COLOR_WORDS = {"white", "black", "red", "blue", "green", "yellow", "brown", "gray", "grey", "orange", "pink", "purple",
               "silver", "tan", "beige", "gold", "maroon", "navy", "teal", "cream"}


def question_line(user: str) -> str:
    """Cauldron appends a one-line answer-format hint ("Short answer required."); the question is the first line."""
    return user.strip().split("\n", 1)[0].strip()


def clean_answer(a: str) -> str:
    """Cauldron capitalises and appends a period: 'Net.' -> 'Net'."""
    a = a.strip()
    return a[:-1].strip() if a.endswith(".") else a


def yes_no(a: str):
    return YES_NO.get(clean_answer(a).lower())


def parse_number(s: str):
    m = NUM.match(s.strip())
    if not m or (m.group(2) is None and m.group(3) is None):
        return None
    try:
        return float(s.replace(",", "").rstrip("%"))
    except ValueError:
        return None


def is_number(s: str) -> bool:
    return parse_number(s) is not None


def notation(s: str) -> tuple:
    neg, intpart, frac, exp, pct = NUM.match(s.strip()).groups()
    return bool(neg), bool(frac), bool(exp), bool(pct)


def format_like(v: float, like: str) -> str:
    """Render v in the notation of `like` (int / fixed decimals / sci / thousands commas / %, leading zeros)."""
    m = NUM.match(like.strip())
    neg, intpart, frac, exp, pct = m.groups()
    if exp:
        mant = len(frac) - 1 if frac else 0
        out = f"{v:.{mant}e}"
    elif frac:
        out = f"{v:.{len(frac) - 1}f}"
    else:
        out = str(int(round(v)))
        if intpart and "," in intpart:
            out = f"{int(round(v)):,}"
        elif intpart and len(intpart) > 1 and intpart.startswith("0") and v >= 0:
            out = out.zfill(len(intpart))
    return out + pct


def number_distractors(answer: str, n: int, rng, question: str = "") -> list[str]:
    """±10–50% relative perturbations in the answer's own notation; years shift by 1–10; 0 gets small offsets."""
    v = parse_number(answer)
    m = NUM.match(answer.strip())
    is_int = not (m.group(3) or m.group(4))
    year = is_int and 1800 <= v <= 2100 and "year" in question.lower()
    out, seen = [], {answer.strip().lower()}
    for attempt in range(200):
        if year:
            cand = v + rng.choice([-1, 1]) * rng.randint(1, 10)
        elif v == 0:
            cand = rng.randint(1, 5) if is_int else round(rng.uniform(0.1, 5), 2)
        else:
            cand = v * (1 + rng.choice([-1, 1]) * rng.uniform(0.10, 0.50))
            if is_int and round(cand) == v:
                cand = v + rng.choice([-1, 1]) * (attempt // 10 + 1)
        if is_int and v >= 0 and cand < 0:
            continue
        s = format_like(cand, answer)
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
            if len(out) == n:
                break
    return out


def answer_kind(question: str, answer: str) -> str:
    if is_number(answer):
        return "number"
    q = question.lower()
    if q.startswith(("what color", "what colour", "what is the color", "what is the colour")) or answer.lower() in COLOR_WORDS:
        return "color"
    return "text"


def pool_keys(question: str, answer: str) -> list[str]:
    """Most specific first: 'what sport is' -> 'what sport' -> kind."""
    kind = answer_kind(question, answer)
    if kind != "text":
        return [kind]
    w = re.findall(r"[a-z']+", question.lower())
    return [" ".join(w[:3]), " ".join(w[:2]), "text"]


class AnswerPool:
    """Answers seen so far in the stream, keyed by question prefix / kind, frequency-weighted when drawn."""

    def __init__(self):
        self.by_key: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    def add(self, question: str, answer: str) -> None:
        for k in pool_keys(question, answer):
            self.by_key[k][answer] += 1

    def draw(self, key: str, n: int, exclude: set, rng) -> list[str]:
        c = {a: w for a, w in self.by_key.get(key, {}).items() if a.lower() not in exclude}
        out = []
        while c and len(out) < n:
            a = rng.choices(list(c), weights=list(c.values()))[0]
            out.append(a)
            del c[a]
        return out


def pick_distractors(question: str, answer: str, same_image: list[tuple[str, str]], pool: AnswerPool, rng,
                     n: int = 3, magnitude: float = 5.0):
    """same_image: (question, answer) of the other questions on this image. Returns (distractors, provenance) or None.
    Order: same-image answers of the same kind (numbers: within `magnitude`x of the answer), then numeric
    perturbation / pool by question prefix (3 words -> 2 words -> all text answers)."""
    exclude = {answer.lower()}
    out, prov = [], []

    def add(cands, tag):
        for c in cands:
            if len(out) == n:
                return
            if c.lower() not in exclude:
                exclude.add(c.lower())
                out.append(c)
                prov.append(tag)

    kind = answer_kind(question, answer)
    others = [(q, a) for q, a in same_image if yes_no(a) is None]
    same_kind = [a for q, a in others if answer_kind(q, a) == kind]
    if kind == "number":  # same sign, same notation and within `magnitude`x, or the odd one out gives itself away
        v = abs(parse_number(answer)) or 1.0
        same_kind = [a for a in same_kind if notation(a) == notation(answer)
                     and 1 / magnitude <= (abs(parse_number(a)) or 1.0) / v <= magnitude]
    rng.shuffle(same_kind)
    add(same_kind, "same_image")
    if kind == "number":
        add(number_distractors(answer, n, rng, question), "perturb")
    else:
        for k in pool_keys(question, answer):
            add(pool.draw(k, n, exclude, rng), f"pool:{k}")
        if kind == "color":  # early in a stream the pool may know too few colours
            fill = sorted(COLOR_WORDS - {"grey"})
            rng.shuffle(fill)
            add([c.capitalize() if answer[:1].isupper() else c for c in fill], "color_list")
    if len(out) < n:
        return None
    return out, prov


def to_choice(answer: str, distractors: list[str], rng):
    """Random position for the correct option. Returns (criteria, target, correct_key)."""
    opts = distractors[:]
    pos = rng.randrange(len(opts) + 1)
    opts.insert(pos, answer)
    keys = KEYS[:len(opts)]
    crit = dict(zip(keys, opts))
    return crit, {k: 1.0 if i == pos else 0.0 for i, k in enumerate(keys)}, keys[pos]


LETTERED = re.compile(r"^Question: (?P<q>.*?)\nChoices:\n(?P<opts>(?:[A-Z]\. .*(?:\n|$))+)\s*(?:Answer with the letter\.?)?\s*$", re.S)


def parse_lettered(user: str, assistant: str):
    """Cauldron ai2d / visual7w: 'Question: q\nChoices:\nA. x\nB. y\n…Answer with the letter.' + 'Answer: B'.
    Returns (question, {label: text}, label) or a skip reason (str)."""
    m = LETTERED.match(user.strip())
    if not m:
        return "prompt_unparsed"
    opts = dict(re.findall(r"^([A-Z])\. (.*)$", m.group("opts"), re.M))
    if list(opts) != [chr(65 + i) for i in range(len(opts))] or len(opts) < 2:
        return "labels_not_consecutive"
    if len({v.strip().lower() for v in opts.values()}) != len(opts):
        return "duplicate_options"
    a = re.fullmatch(r"(?:Answer: )?([A-Z])\.?", assistant.strip())
    if not a or a.group(1) not in opts:
        return "answer_not_in_options"
    return m.group("q").strip(), {k: v.strip() for k, v in opts.items()}, a.group(1)
