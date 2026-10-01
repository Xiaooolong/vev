# Changelog

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
