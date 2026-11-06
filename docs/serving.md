# Running the server

How to run `vev serve` the way the Decision Index numbers were measured. All numbers are for `vev-4b-v2` on the
public vision suite of the Decision Index (12,210 rows, one choice question per row, up to two images).

## Install

`requirements-serve.txt` pins the packages of the two environments the latencies below were measured in.

```bash
git clone https://github.com/Xiaooolong/vev && cd vev
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps -r requirements-serve.txt
pip install --no-deps .
```

`--no-deps` is needed because torch 2.8.0 declares `triton==3.4.0` while flash-linear-attention 0.5.2 runs on triton
3.8.0; `pip check` reports the mismatch, which is expected. Triton compiles kernels at run time and needs `gcc`. A
plain `pip install .` also gives a working server, but possibly different kernels, so probabilities can differ in the
last digits.

| | H800 | RTX PRO 6000 Blackwell |
|---|---|---|
| Where | cluster node, Python 3.13 | Hugging Face Jobs (`rtx-pro-6000`), Python 3.11, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` |
| torch / triton | 2.8.0+cu128 / 3.8.0 | 2.8.0+cu128 / 3.8.0 |
| transformers / flash-linear-attention | 5.17.0 / 0.5.2 | 5.17.0 / 0.5.2 |

Neither environment has `causal-conv1d`, so transformers logs that `causal_conv1d_fn` falls back to PyTorch. All
numbers here were measured with that fallback.

## Start

```bash
scripts/serve_board.sh CountingSheep/vev-4b-v2 --port 8009
```

The script sets the reading layout explicitly and starts `python -m vev serve`; arguments after the model go to
`vev serve`. The variables override the checkpoint's `vev.json`, which for `vev-4b-v2` has the same values.

| Variable | Value | Effect |
|---|---|---|
| `VEV_BIDIR` | `state` | in the 8 full-attention layers, the state tokens attend to each other in both directions |
| `VEV_READ_TWICE` | `full` | a text state is written twice: state, question, state again, question with options |
| `VEV_READ_TWICE_MAX` | `3000` | only for text states up to 3,000 tokens |
| `VEV_IMAGE_READ` | `twice` | the same for a request with images and one question |
| `VEV_MAX_IMAGE_PIXELS` | `1638400` | larger images are downscaled to 1.6 MP |

A request that is over the token limits (32,768 for the state plus one question, 65,536 for the whole request) only
when written twice is read once, without the repetition.

With Docker:

```bash
docker build -t vev .
docker run --gpus all -p 8009:8009 \
  -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.triton:/root/.triton \
  --entrypoint /app/scripts/serve_board.sh vev CountingSheep/vev-4b-v2 --host 0.0.0.0 --port 8009
```

The second volume keeps Triton's compiled kernels between runs.

## Startup and warm-up

Triton compiles and tunes a flash-linear-attention kernel the first time it sees a new input size, which can take
about 8 s. To keep that off real requests, `vev serve` sends itself 17 requests before `/healthz` answers
(`vev/warmup.py`): text, one to four images at 1.6 MP, rows of up to 16,384 tokens, then batches of 2 to 64
questions. `--no-warmup` skips it.

| Run | Warm-up | Start to healthy | Slowest of the first 10 rows | Kernels tuned after healthy |
|---|---|---|---|---|
| H800, empty Triton cache | 114.9 s | 130.0 s | 488 ms | 0 |
| H800, cache filled | 5.8 s | 23.2 s | 479 ms | 0 |
| RTX PRO 6000, empty Triton cache | 102.0 s | 111.1 s | 706 ms | 0 |

Without the warm-up, two of the first ten rows took about 8 s each on both GPUs. The warm-up does not change answers:
all 12,210 rows gave bit-identical probabilities with and without it on the H800. A single question longer than
16,384 tokens, or a batch size other than 1–4, 6, 8, 16 and 64, still tunes once on an empty cache.

## Latency

Measured as the board describes it: one process on one GPU, one request at a time, 10 untimed rows first, then the
end-to-end time of each row as the Decision Index vision runner records it (including encoding the images on the
client), with `evals/decision_index/vision_engine.py`. Times in ms.

| GPU | Rows | median | mean | p80 | p95 |
|---|---|---|---|---|---|
| H800 | all 12,210 | 101 | 151 | 225 | 395 |
| H800 | the 574 two-image rows | 270 | 237 | 320 | 373 |
| RTX PRO 6000 | 1,294-row sample, weighted to the suite | 120 | 182 | 283 | 458 |
| RTX PRO 6000 | the 574 two-image rows | 344 | 299 | 408 | 481 |

- The board's limit is 1,000 ms for each of median, mean and p80.
- The RTX PRO 6000 sample takes up to 60 rows per benchmark and image count, plus all 574 two-image rows, each
  weighted by its group's share of the suite.
- The two GPUs run different kernels: on the sample, 1,289 of 1,294 rows get the same answer, and probabilities
  differ by up to 0.060.

Sources: `results/decision-index/latency-h800/summary.json`, `results/decision-index/latency-rtx-pro-6000/summary.json`,
`results/decision-index/warmup/summary.json`.
