# Vev

Vev is a judgment model with open weights. You send it a state (text, JSON, images, or a mix) and a set of typed questions
(yes/no, multiple choice, or a graded score), and it returns a probability for every answer in one forward pass,
without generating text. The server speaks the same request and response format as TypeSafe's `/v1/systemone`
endpoint ([TypeSafe docs](https://docs.typesafe.ai/concepts/system-one.md)), so clients written for that API work
against Vev after changing the base URL.

This is v0.1, a research preview. The weights are licensed for non-commercial use only; see [License](#license).

[README in Chinese](README_zh.md)

## Quickstart

```bash
pip install git+https://github.com/Xiaooolong/vev
vev serve --model CountingSheep/vev-4b          # downloads the weights on first start, listens on 127.0.0.1:8009
```

Vev needs Python 3.11 or newer and an NVIDIA GPU; CPU and Apple Silicon are untested. In bf16, `vev-4b` takes about
10 GB of GPU memory once loaded and `vev-9b` about 19 GB; long states and images need more on top. Docker:

```bash
git clone https://github.com/Xiaooolong/vev && cd vev
docker build -t vev .
docker run --gpus all -p 8009:8009 -v ~/.cache/huggingface:/root/.cache/huggingface vev \
  --model CountingSheep/vev-4b
```

With the official Python SDK (`pip install typesafe-sdk`):

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009")
resp = client.system_one(
    state="Order #4411 still shows 'label created' after 9 days. I need it before Friday.",
    questions={
        "urgent": Noul(instructions="Is the customer asking for something time-sensitive?"),
        "team": Choice(instructions="Which team should handle this?",
                       criteria={"shipping": "Delivery and tracking", "billing": "Charges and refunds"}),
    },
)
print(resp.answers["urgent"].noul)          # probability of "yes": 0.98 with vev-4b
print(resp.answers["team"].probabilities)   # {"shipping": 0.97, "billing": 0.03}
```

Images are a Vev extension; the official API takes text only. Put them in the state as an `image` object with a
data URL:

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "vev-latest",
  "state": {"screen": {"image": {"url": "data:image/png;base64,iVBORw0..."}}},
  "questions": {"error": {"type": "noul", "instructions": "Does the screen show an error message?"}}
}'
```

The full request format, limits and error codes are in [spec/systemone-api.md](spec/systemone-api.md).

## Models

| Model | Base | Hugging Face |
|---|---|---|
| `vev-4b` | Qwen3.5-4B | [CountingSheep/vev-4b](https://huggingface.co/CountingSheep/vev-4b) (merged), [CountingSheep/vev-4b-lora](https://huggingface.co/CountingSheep/vev-4b-lora) (adapter) |
| `vev-9b` | Qwen3.5-9B | [CountingSheep/vev-9b](https://huggingface.co/CountingSheep/vev-9b) (merged), [CountingSheep/vev-9b-lora](https://huggingface.co/CountingSheep/vev-9b-lora) (adapter) |

Both are LoRA fine-tunes. The answer probabilities are the model's own next-token probabilities over the answer
tokens (Yes/No, option letters, level digits), renormalised over the allowed answers. Training adjusts those
probabilities; it does not add a separate classifier head.

## Results

All numbers below are accuracy on the full sets, measured with the harness in `evals/`. Where two systems are
compared, the difference is tested with a paired bootstrap over the same questions (95% interval); "n.s." marks
differences whose interval includes zero; for the image sets, questions that share an image are resampled together.
Per-set metrics, including Brier score and calibration error, are in [results/](results/). Sets in `results/` that
are not in the tables below were used for diagnostics during development.

The systems compared: [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is TypeSafe's hosted
judgment model; [kev-4B](https://github.com/jaredpalmer/kev) is an open judgment model built on a 4B Qwen base.
The text sets are [judgekit](https://github.com/lexingtonhibiki/judgekit) (Chinese),
[JevBench](https://github.com/fstandhartinger/jevbench) (the public subset),
[nimble](https://github.com/bespokelabsai/nimble) and kev's own held-out test set, transfer-v4.

### Text judgment

| Set | n | kev-4B | Vev-4B | Vev-9B | Jev |
|---|---|---|---|---|---|
| judgekit (Chinese) | 130 | 0.846 | 0.962 | 0.938 | 0.962 |
| JevBench (public subset) | 231 | 0.719 | 0.766 | 0.823 | 0.861 |
| nimble | 324 | 0.735 | 0.707 | 0.747 | 0.923 |
| kev transfer-v4 | 1528 | 0.802 | 0.776 | 0.781 | 0.854 |

kev-4B was run locally with the same harness. Jev was queried through TypeSafe's API (`jev-latest`, 23–24 September
2026); no Jev outputs were used for training.
Against kev-4B, Vev-4B is better on judgekit and Vev-9B on judgekit and JevBench; both Vev models are
2–3 points behind on kev's own test set, and nimble is n.s. for both. Jev is significantly ahead of both Vev models
on nimble (18–22 points) and kev transfer-v4 (7–8 points), and ahead of Vev-4B on JevBench; Vev-9B's 3.9-point gap
on JevBench is n.s. On judgekit Vev-4B ties Jev and Vev-9B is 2.3 points lower (n.s.).

### Before and after training

The table compares Vev with its base model, read the same way without fine-tuning (zero-shot).

| Set | n | Qwen3.5-4B | Vev-4B | Qwen3.5-9B | Vev-9B |
|---|---|---|---|---|---|
| judgekit | 130 | 0.962 | 0.962 | 0.938 | 0.938 |
| JevBench | 231 | 0.758 | 0.766 | 0.805 | 0.823 |
| nimble | 324 | 0.710 | 0.707 | 0.710 | **0.747** |
| kev transfer-v4 | 1528 | 0.758 | **0.776** | 0.764 | **0.781** |
| MMBench-EN | 1164 | 0.872 | 0.881 | 0.887 | **0.902** |
| POPE | 9000 | 0.894 | *0.889* | 0.894 | 0.897 |
| MMStar | 1498 | 0.544 | **0.627** | 0.608 | **0.674** |

Bold marks a significant improvement, italics a significant drop. The one regression is Vev-4B on POPE: accuracy is
0.5 points lower (95% interval −0.9 to −0.1) and the Brier score 0.007 worse. Vev-9B has no significant drop.

### Image judgment

Public vision benchmarks mostly ask questions about an image. Vev is meant for judging an image against a stated
rule, so we built judgment sets from human-labelled public data. The converters are in `evals/datasets/`; the
converted data is not redistributed.

| Set | What is judged | n | Qwen3.5-4B | Vev-4B | Qwen3.5-9B | Vev-9B |
|---|---|---|---|---|---|---|
| policy_mod | Does the image violate this safety policy? (LlavaGuard) | 659 | 0.686 | **0.742** | 0.666 | **0.724** |
| game_glitch | Does the game screenshot contain a glitch? (VideoGameQA-Bench) | 1000 | 0.646 | 0.613 | 0.577 | **0.626** |
| game_clip | Is the object clipping into the character? (VideoGameQA-Bench) | 686 | 0.561 | **0.671** | 0.736 | 0.729 |
| ui_toggle | Is there a switch turned on / off? (MobileViews) | 800 | 0.901 | **0.928** | 0.921 | 0.923 |
| ui_input | Is there a text input field? (MobileViews) | 800 | 0.949 | 0.951 | 0.948 | 0.955 |
| t2i_elem | Does the generated image show this prompt element? (EvalMuse) | 1027 | 0.690 | **0.715** | 0.733 | 0.703 |

Vev-4B on game_glitch and Vev-9B on t2i_elem are about 3 points lower than their base models; neither difference
is significant.

## Limitations

- Jev is more accurate than Vev on nimble and kev transfer-v4 (by 18–22 and 7–8 points), and than Vev-4B on
  JevBench.
- Answers depend on the order in which options are listed. Reversing the options changes the top answer on 13%
  (Vev-9B) and 17% (Vev-4B) of JevBench and kev transfer-v4 questions, against 10–12% for kev-4B and under 4% for
  Jev. Fine-tuning lowered the rate on those two sets (the base models: 19–25%). On nimble it went down for 9B
  (33% → 9%) and up for 4B (19% → 21%).
- Judgments that need several steps of reasoning are weaker than the base model's own answer when it may think
  first. On one internal scenario set (not released), the base model in thinking mode was 4–8 points more accurate
  than Vev's single pass. On most of the public sets we tried, the single pass was as accurate or better.
- On binary "is something wrong here?" questions both the base models and Vev lean towards "no": they flag
  problems less often than the labels say.
- The image judgment sets are our own conversions of public data, not established benchmarks.
- "Compatible" means the request and response shapes match `/v1/systemone`. It does not mean Vev behaves like Jev.
- Vev is not affiliated with or endorsed by TypeSafe.

## Reproducing

`train/` holds the training code; `data/` builds the training mixture from 42 public sources; `evals/` converts
the evaluation sets and runs them against any `/v1/systemone` server. [TRAINING.md](TRAINING.md) lists the exact
recipe, library versions and what cannot be rebuilt from public sources.

`conformance/` checks an implementation against the API spec. It runs against any server, including the official
one:

```bash
SO_BASE_URL=http://127.0.0.1:8009 SO_SUPPORTS_IMAGES=1 pytest conformance -q
```

## License

The code in this repository is released under the Apache License 2.0. The model weights are released under
CC BY-NC 4.0: six of the training sources allow research or non-commercial use only, and eleven have no clear
license for their data. This license does not replace the terms of those sources. The base models, Qwen3.5-4B and
Qwen3.5-9B, are Apache 2.0. [TRAINING.md](TRAINING.md) lists every training source and its terms.

## Citation

```bibtex
@software{vev2026,
  title  = {Vev: an open judgment model for text and images},
  author = {Wang, Xiaolong},
  year   = {2026},
  url    = {https://github.com/Xiaooolong/vev},
  version = {0.1.0}
}
```
