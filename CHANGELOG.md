# Changelog

## 0.2.1 (2026-11-06)

- `/v1/models` reports the release date from the checkpoint's `vev.json` (`release_date`) when it has one.
- The package reports its own version (0.2.0 reported 0.1.1).
- The Docker image and `scripts/serve_board.sh` default to `CountingSheep/vev-4b-v2`.
- Docs: models are named `vev-4b-v2` (second generation) and `vev-4b` / `vev-9b` (first generation); the README image
  example downloads the picture instead of reading it from a clone; the spec gives the image limit and the reading
  layout of `vev-4b-v2`.

## 0.2.0 (2026-11-06)

New model [`vev-4b-v2`](https://huggingface.co/CountingSheep/vev-4b-v2), the second generation of `vev-4b`. `vev-4b`,
`vev-9b` and the adapter repositories are unchanged.

- `vev-4b-v2` is trained from Qwen3.5-4B in four stages with a new data mix, including data for the task families of
  the Jev Decision Index (DI) built from their training splits. Sources and the overlaps with evaluation data:
  [TRAINING_DATA.md](TRAINING_DATA.md). CharXiv, whose card forbids training, is no longer used.
- DI public suites: vision 72.90, text 60.32 (`vev-4b`: 38.91). On the 18 release sets, `vev-4b-v2` is significantly
  more accurate than `vev-4b` on 8 and less accurate on 4 (game_vr, policy_cat, t2i_elem, ui_toggle);
  `results/comparisons/vev-4b-v2.json`.
- `vev serve` reads the reading layout from the checkpoint's `vev.json`: `bidir` (state tokens attend to each other
  in both directions), `read_twice` (short text states are written twice), `image_read` (images are passed twice in
  one-question requests) and `max_image_pixels` (1.6 MP for `vev-4b-v2`). Checkpoints without these fields are read
  as before. A request that is over the token limits only because of the repetition is read once.
- The server warms up its GPU kernels before `/healthz` answers (`--no-warmup` skips it).
- `requirements-serve.txt`, `scripts/serve_board.sh`, a `Dockerfile` and [docs/serving.md](docs/serving.md): the
  environment and start command the DI latencies were measured with (H800: 101 / 151 / 225 ms median / mean / p80).
- `evals/decision_index/`: engines that run the DI tool's text and vision suites against `vev serve`.
- `results/vev-4b-v2/` holds the new results.

## 0.1.1 (2026-10-01)

- The questions of a request run as one batch instead of one forward pass each. On one H800, 100 questions about a
  short text take 0.23 s instead of 3.75 s with `vev-4b`; images go through the image processor once per request.
- Requests with one question give bit-identical probabilities to 0.1.0. With several questions, probabilities move by
  up to a few hundredths (bf16 rounding); spec §6 and the conformance tolerance are updated with the measured values.
- `flash-linear-attention` 0.5.2 is a dependency on Linux, as in the evaluation environment; the server logs which
  DeltaNet kernels it uses. `torchvision` is a dependency.
- `vev serve --revision`; startup fails fast without CUDA; 500 responses no longer include the exception text.
- `evals.run --one-question-per-request`.

## 0.1.0 (2026-10-01)

First public release, a research preview.

- `vev serve`: HTTP server with the `/v1/systemone` request and response shapes. Text, JSON and images in `state`;
  `noul`, `choice` and `score` questions.
- Models: `vev-4b` and `vev-9b`, LoRA fine-tunes of Qwen3.5-4B and Qwen3.5-9B, released as merged weights and as
  adapters. Weights are CC BY-NC 4.0.
- Conformance suite that runs against any `/v1/systemone` implementation.
- Evaluation harness, dataset converters and the training code used for the release.
