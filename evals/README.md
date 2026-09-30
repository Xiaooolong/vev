# evals: evaluation suite

Runs the same evaluations against any `/v1/systemone` implementation. The model under test is never loaded in-process; everything goes over HTTP, so the official Jev API, kev, Vev itself and a zero-shot baseline server are all evaluated with the same command.

## Layout

```
evals/
├─ README.md            this file: record format and directory conventions
├─ schema.py            record format validation
├─ metrics.py           metrics
├─ client.py            HTTP client: concurrency, retries, inlines image.path as a data URL
├─ run.py               CLI: records + target -> raw outputs + metrics JSON
├─ report.py            CLI: a set of result JSONs -> markdown summary table
├─ gate.py              CLI: paired comparison of a trained model against the zero-shot reference
├─ criteria_flip.py     CLI: criteria-flip probe on paired records
├─ datasets/            converters, one file per source, each writes records JSONL
│   ├─ text_*.py
│   └─ image_*.py
├─ manifest/            per-set record counts, hashes, source revisions, licenses (in git)
├─ data/                converted JSONL and images (not in git)
│   ├─ text/<set>.jsonl
│   ├─ image/<set>.jsonl
│   └─ images/<set>/<id>.<ext>
└─ tests/               unit tests on synthetic data
```

Some converted sets (ccbench, cmmmu, hallusionbench, mjbench, mmbench_cn, mmmu, vl_rewardbench) have no results in
the README. They are converted so that `data/dedup.py` can keep their images and texts out of the training data.

## Record format

One JSON object per line; one record = one state with N questions.

```json
{
  "id": "mmstar/val/000123",
  "source": "mmstar",
  "split": "val",
  "license": "unknown",
  "lang": "en",
  "state": {"context": "question stem or dialogue text, optional", "image": {"path": "../images/mmstar/000123.jpg"}},
  "questions": {
    "q1": {"type": "choice", "instructions": "…", "criteria": {"A": "…", "B": "…", "C": "…", "D": "…"}},
    "q2": {"type": "noul", "instructions": "…"}
  },
  "targets": {
    "q1": {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0},
    "q2": 0.87
  },
  "meta": {"grid_id": "mmstar/img/000123", "orig_answer_type": "mc"}
}
```

Rules:

- `state` is any JSON value (string, object, array). Images use the reserved object `{"image": {"path": "…"}}` or `{"image": {"url": "data:…"}}`. `path` is relative to the directory of the JSONL file; the client reads the file and inlines it as a data URL when sending the request. Image files all live under `evals/data/images/<set>/`, so `evals/data/image/<set>.jsonl` refers to them as `../images/<set>/<id>.jpg`. A state may hold several images and mix images with text; when the state is a single image, the whole state is `{"image": {"path": …}}`.
- Each value in `questions` is a `/v1/systemone` Question as is (see `spec/systemone-api.md` §3.1). For choice, the criteria keys are the labels (multiple-choice questions use `"A"`, `"B"`, …, with the option text as the description); for score, criteria is the list of levels from low to high.
- `targets` are distributions, not hard labels: for choice, an object label -> probability whose key set must equal the criteria key set; for noul, P(true) as a float; for score, an object `"0"`…`"n-1"` -> probability. Multi-annotator sets keep the vote shares; single-annotator sets use one-hot.
- `license` is one of `commercial-ok` / `non-commercial` / `unknown`. Eval sets are not filtered by license, but the value must be accurate.
- `lang`: `en` / `zh` / another ISO 639-1 code; the main language of the question and options.
- `meta` is free-form, but `grid_id` should be set: records on the same image or the same text share it.
- Optional `meta.negated_questions`: `{"q2": {…noul question…}}`, the negation of the original question, used to test P(q)+P(¬q)=1. Converters add it when they can.

## Metrics

One JSON per set, with fixed field names:

| Key | Meaning |
|---|---|
| `n_records`, `n_questions`, `by_type` | counts |
| `accuracy`, `macro_f1` | argmax vs. argmax (noul threshold 0.5; score by argmax level) |
| `brier`, `nll` | against the target distributions |
| `ece`, `ece_ci95`, `ece_noise_floor` | 10 bins; 1000 bootstrap resamples; noise floor = expected ECE of "perfectly calibrated" answers sampled from the target distributions |
| `ece_maxprob` | the same ECE, but with max p as the confidence instead of the response's `confidence`. For choice/score, `confidence` is the spec's rescaling formula, not the probability of being correct, so this is the one to read for whether the distribution itself is calibrated |
| `reliability` | `[conf_mean, acc, n]` per bin |
| `coverage_accuracy` | thresholds 0.50…0.99 in steps of 0.01: `[threshold, coverage, accuracy]` |
| `auroc_confidence` | AUROC of the confidence as a score for being correct |
| `confidence_distinct` | number of distinct confidence values |
| `confidence_one_bucket` | `{"n": questions with confidence == 1.0, "errors": how many of them are wrong}` |
| `order_flip_rate` | argmax flip rate after reversing the options (choice/score; needs `--perturb order`) |
| `batched_vs_separate_max_dp` | max \|Δp\| between batched and single-question requests (`--perturb separate`) |
| `repeat_max_dp` | max \|Δp\| when resending the same request (`--perturb repeat`) |
| `negation_gap` | mean \|P(q)+P(¬q)−1\| (for records with `meta.negated_questions`) |
| `jitter_max_dp`, `jitter_flip_rate` | near-frame stability: each image in the state is cropped by ≤3% at the edges and resized back, brightness ±5%, half of them lightly blurred, then asked again; max \|Δp\| and argmax flip rate (`--perturb jitter`, image records only) |
| `latency_ms` | `{"p50","p95","mean"}` per request |
| `errors` | HTTP error counts and examples |

Confidence: choice/score use the response's `confidence`; noul uses `max(p, 1−p)`. Correct: for choice, the argmax equals the target argmax; for noul, `p≥0.5` agrees with `target≥0.5`; for score, the argmax level is the same.

## Commands

```bash
pip install -e ".[dev]"

# run one set
python -m evals.run --records evals/data/image/mmstar.jsonl \
  --base-url http://127.0.0.1:8009 --api-key local --model jev-latest \
  --target zeroshot-qwen3.5-4b --out results/zeroshot-qwen3.5-4b/mmstar.json \
  --perturb order,separate,repeat --concurrency 8 --limit 0 --sample 0 --seed 0

# summarize all sets of one target
python -m evals.report results/zeroshot-qwen3.5-4b/ --out results/zeroshot-qwen3.5-4b/summary.md
```

Besides the metrics, the output JSON of `run.py` carries `raw`: the response probabilities, latency and perturbed responses for every question of every record, for inspection.
