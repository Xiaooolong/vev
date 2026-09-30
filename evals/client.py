import base64
import mimetypes
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import httpx

RETRY_STATUS = {429, 529}
LOOPBACK = {"127.0.0.1", "localhost", "::1"}
MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class SystemOneError(Exception):
    def __init__(self, status: int | None, body: Any, attempts: int):
        self.status = status
        self.body = body
        self.attempts = attempts
        super().__init__(f"status={status} attempts={attempts} body={str(body)[:300]}")

    def to_dict(self) -> dict:
        return {"status": self.status, "body": self.body, "attempts": self.attempts}


@lru_cache(maxsize=256)
def _data_url(path: str) -> str:
    p = Path(path)
    mime = MIME_BY_EXT.get(p.suffix.lower()) or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def inline_images(node: Any, base_dir: Path) -> Any:
    # any {"image": {"path": ...}} value becomes {"image": {"url": "data:..."}}; returns a new tree
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "image" and isinstance(v, dict) and set(v) == {"path"} and isinstance(v["path"], str):
                path = Path(v["path"])
                if not path.is_absolute():
                    path = base_dir / path
                out[k] = {"url": _data_url(str(path.resolve()))}
            else:
                out[k] = inline_images(v, base_dir)
        return out
    if isinstance(node, list):
        return [inline_images(v, base_dir) for v in node]
    return node


class SystemOneClient:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 max_retries: int = 4, base_dir: str | Path = ".", backoff: float = 0.5,
                 pool_size: int = 32):
        self.model = model
        self.max_retries = max_retries
        self.base_dir = Path(base_dir)
        self.backoff = backoff
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=timeout,
            limits=httpx.Limits(max_connections=pool_size, max_keepalive_connections=pool_size),
            # env proxies (HTTP(S)_PROXY) are honoured for remote targets but never for loopback
            trust_env=httpx.URL(base_url).host not in LOOPBACK,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _sleep_before_retry(self, attempt: int, resp: httpx.Response | None) -> None:
        delay = self.backoff * (2 ** attempt) * (1 + 0.25 * random.random())
        if resp is not None and resp.headers.get("retry-after-ms"):
            try:
                delay = max(delay, float(resp.headers["retry-after-ms"]) / 1000)
            except ValueError:
                pass
        time.sleep(delay)

    def ask(self, state: Any, questions: dict) -> dict:
        body = {"state": inline_images(state, self.base_dir), "model": self.model, "questions": questions}
        attempt = 0
        while True:
            resp = None
            t0 = time.perf_counter()
            try:
                resp = self._http.post("/v1/systemone", json=body)
            except httpx.TransportError as e:
                if attempt >= self.max_retries:
                    raise SystemOneError(None, f"{type(e).__name__}: {e}", attempt + 1) from e
            else:
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    out = resp.json()
                    out["latency_ms"] = latency_ms
                    return out
                retryable = resp.status_code in RETRY_STATUS or resp.status_code >= 500
                if not retryable or attempt >= self.max_retries:
                    try:
                        detail = resp.json()
                    except ValueError:
                        detail = resp.text[:500]
                    raise SystemOneError(resp.status_code, detail, attempt + 1)
            self._sleep_before_retry(attempt, resp)
            attempt += 1

    def ask_many(self, items: list[tuple[Any, dict]], concurrency: int = 8,
                 on_done: Callable[[int, Any], None] | None = None) -> list[dict | SystemOneError]:
        results: list[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = {pool.submit(self.ask, state, qs): i for i, (state, qs) in enumerate(items)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except SystemOneError as e:
                    results[i] = e
                except Exception as e:
                    results[i] = SystemOneError(None, f"{type(e).__name__}: {e}", 1)
                if on_done is not None:
                    on_done(i, results[i])
        return results
