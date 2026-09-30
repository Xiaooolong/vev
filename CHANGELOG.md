# Changelog

## 0.1.0 (2026-10-01)

First public release, a research preview.

- `vev serve`: HTTP server with the `/v1/systemone` request and response shapes. Text, JSON and images in `state`;
  `noul`, `choice` and `score` questions.
- Models: `vev-4b` and `vev-9b`, LoRA fine-tunes of Qwen3.5-4B and Qwen3.5-9B, released as merged weights and as
  adapters. Weights are CC BY-NC 4.0.
- Conformance suite that runs against any `/v1/systemone` implementation.
- Evaluation harness, dataset converters and the training code used for the release.
