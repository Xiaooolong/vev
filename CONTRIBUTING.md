# Contributing

Issues and pull requests are welcome. For anything larger than a bug fix, open an issue first so we can agree on
the approach before you spend time on it.

## Development setup

```bash
git clone https://github.com/Xiaooolong/vev
cd vev
pip install -e ".[dev,train]"
pytest -q
```

The unit tests need no GPU. To check the HTTP contract without a model:

```bash
python conformance/stub_server.py --port 8009 &
SO_STUB=1 SO_TARGET=stub SO_SUPPORTS_IMAGES=1 pytest conformance -q
```

To run the same suite against a real server, start `vev serve` and drop `SO_STUB=1`.

## Pull requests

- Keep changes focused; one topic per pull request.
- Add or update tests for behaviour changes.
- Run `ruff check vev conformance` and `pytest -q` before pushing.
- If a change affects model outputs, include before/after numbers from `python -m evals.run` on the sets it touches.
