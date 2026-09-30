# Conformance suite

HTTP only, so it runs against any `/v1/systemone` implementation. It tests `spec/systemone-api.md`: response shape, errors, real clients, question isolation and determinism. Latency curves are measured separately by `curves.py`, which only writes JSON and asserts nothing.

Each run writes everything it observed to `results/<SO_TARGET>/<timestamp>.json`, including the response bodies seen by failing tests.

## Running

```bash
pip install -e ".[dev]"
pip install typesafe-sdk langchain-typesafe   # optional; the client tests are skipped without them

# a self-hosted server, local or remote
SO_TARGET=kev-4b SO_BASE_URL=http://127.0.0.1:8009 SO_API_KEY=local SO_MODEL=jev-latest \
  pytest conformance -q

# the official Jev API (shape reference, costs a few cents)
SO_TARGET=jev-official SO_BASE_URL=https://api.typesafe.ai SO_API_KEY=$TYPESAFE_API_KEY \
  SO_EXPECT_AUTH=1 SO_ISOLATION_TOL=0.05 SO_DETERMINISM_TOL=0.02 \
  pytest conformance -q

# latency curves
SO_TARGET=kev-4b SO_BASE_URL=http://127.0.0.1:8009 python conformance/curves.py
```

`SO_ISOLATION_TOL` and `SO_DETERMINISM_TOL` default to 1e-4 and 0, which is what this project's server must meet. The official Jev API quantizes probabilities to 0.01, returns four different outputs for the same request sent five times (max difference 0.02 in our runs), and differs by up to 0.03 between batched and separate requests. Against it, loosen the tolerances to 0.05 / 0.02; otherwise the isolation and determinism groups always fail. Failed runs still record their observations.

## Environment variables

See the comment at the top of `conftest.py`.
