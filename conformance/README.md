# Conformance suite

HTTP only, so it runs against any `/v1/systemone` implementation. It tests `spec/systemone-api.md`: response shape, errors, real clients, question isolation and determinism. Latency curves are measured separately by `curves.py`, which only writes JSON and asserts nothing.

Each run writes everything it observed to `results/<SO_TARGET>/<timestamp>.json`, including the response bodies seen by failing tests.

## Running

```bash
pip install -e ".[dev]"
pip install typesafe-sdk langchain-typesafe   # optional; the client tests are skipped without them

# a self-hosted server, local or remote
SO_TARGET=vev-4b SO_BASE_URL=http://127.0.0.1:8009 SO_API_KEY=local \
  pytest conformance -q

# the official Jev API (shape reference, costs a few cents)
SO_TARGET=jev-official SO_BASE_URL=https://api.typesafe.ai SO_API_KEY=$TYPESAFE_API_KEY \
  SO_EXPECT_AUTH=1 SO_ISOLATION_TOL=0.05 SO_DETERMINISM_TOL=0.02 \
  pytest conformance -q

# latency curves
SO_TARGET=vev-4b SO_BASE_URL=http://127.0.0.1:8009 python conformance/curves.py
```

`SO_ISOLATION_TOL` and `SO_DETERMINISM_TOL` default to 1e-4 and 0, which is what this project's server must meet. The official Jev API quantizes probabilities to 0.01, returns four different outputs for the same request sent five times (max difference 0.02 in our runs), and differs by up to 0.03 between batched and separate requests. Against it, loosen the tolerances to 0.05 / 0.02; otherwise the isolation and determinism groups always fail. Failed runs still record their observations.

## Environment variables

See the comment at the top of `conftest.py`.

## Observed behaviour of the official API

2026-09-23, jev-1.13.0:

- `instructions` omitted: 200. noul without `criteria`: 200. choice with 1 option: 200, probability 1.0, confidence 1.0.
- `x-typesafe-request-id` looks like `req_01a0cc8b…`. `/healthz` does not exist (404).
- `GET /v1/models` returns `{"models":[{"name":"jev-latest",…},{"name":"jev-preview",…}]}`, with `release_date` as an ISO timestamp.
- `model` is filled with `jev-1.13.0`.
- Probabilities are quantized to 0.01; `score` is computed from the unquantized values and can differ by up to 0.03 (10 levels) from the expectation over the quantized probabilities.
- On the same state, 5 questions batched vs. one at a time differ by 0.01 for noul and up to 0.04 for score; the order of questions in the request also moves the probabilities (above the quantization step). The official isolation is not strict.
- Content in sibling questions is not visible to another question (probe 0.06 -> 0.06); put in the state, it is (0.81).
- The same request sent 5 times gives 4 different outputs, max difference 0.02. The official API is not deterministic.
- Client-side latency to api.typesafe.ai: 3 questions 305 ms, 50 options 278 ms, 23 questions 325 ms; essentially independent of the number of questions and options.
- 429/529 not observed; their shapes in the spec are inferred.
