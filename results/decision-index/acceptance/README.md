# Acceptance comparisons

Reports from the release test of `vev-4b-v2`. `s4c` is `vev-4b-v2`; `s4m-avg` is an unreleased research checkpoint
trained on the earlier data, used as the reference. The rules were fixed before training; the vev-4b-v2 model card
explains them ("Release acceptance").

- `s4c-vs-s4m-avg.*`, `kit-*`: the acceptance run. Its Decision Index text numbers (60.13 / 63.41) are from before the
  token-limit fallback, when both checkpoints refused the same 21 scored requests.
- `text-s4c-v023-vs-s4m-avg-v023.*`: the same text comparison after the fallback (60.32 / 63.60); the judged result is
  unchanged.

References to internal plan sections and files (`docs/74`, `data/stage4/...`) point to the research code, which is not
in this repository.
