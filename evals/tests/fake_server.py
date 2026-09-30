import hashlib
import json
import math
import random
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse


@dataclass
class FakeConfig:
    mode: str = "deterministic"  # "deterministic" | "random" | "oracle"
    oracle: dict[str, Any] = field(default_factory=dict)  # instructions -> dist (noul: float)
    order_bias: float = 0.0  # extra logit for the first label/level in request order
    fail_first: int = 0  # first N requests get fail_status
    fail_status: int = 529
    fail_instructions: set[str] = field(default_factory=set)  # always 500 if any question has these
    delay_ms: float = 0.0
    requests: list[dict] = field(default_factory=list)


def _hash_unit(*parts: Any) -> float:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def _confidence(probs: list[float]) -> float:
    k = len(probs)
    return 1.0 if k == 1 else (max(probs) - 1 / k) / (1 - 1 / k)


class FakeServer:
    def __init__(self):
        self.config = FakeConfig()
        self._lock = threading.Lock()
        self._rng = random.Random()
        self.app = self._build_app()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.url = ""

    def reset(self, **kwargs: Any) -> FakeConfig:
        self.config = FakeConfig(**kwargs)
        return self.config

    def _logits(self, state: Any, q: dict, keys: list[Any]) -> list[float]:
        cfg = self.config
        if cfg.mode == "random":
            with self._lock:
                xs = [self._rng.gauss(0, 2) for _ in keys]
        else:
            # order-invariant: each option's logit depends on its own content, not its position
            xs = [4 * _hash_unit(state, q.get("instructions", ""), k) - 2 for k in keys]
        if xs:
            xs[0] += cfg.order_bias
        return xs

    def _answer(self, state: Any, q: dict) -> dict:
        cfg, qtype = self.config, q["type"]
        oracle = cfg.oracle.get(q.get("instructions", "")) if cfg.mode == "oracle" else None
        if qtype == "noul":
            if oracle is not None:
                p = float(oracle)
            else:
                p = 1 / (1 + math.exp(-self._logits(state, q, ["true"])[0]))
            return {"type": "noul", "noul": p}
        if qtype == "choice":
            labels = list(q["criteria"])
            if oracle is not None:
                probs = [float(oracle[k]) for k in labels]
            else:
                probs = _softmax(self._logits(state, q, [[k, q["criteria"][k]] for k in labels]))
            best = labels[max(range(len(labels)), key=lambda i: (probs[i], -i))]
            # emit keys in reverse to exercise lookup-by-key on the client
            out = {labels[i]: probs[i] for i in reversed(range(len(labels)))}
            return {"type": "choice", "choice": best, "confidence": _confidence(probs), "probabilities": out}
        levels = q["criteria"]
        n = len(levels)
        if oracle is not None:
            probs = [float(oracle[str(i)]) for i in range(n)]
        else:
            probs = _softmax(self._logits(state, q, levels))
        return {
            "type": "score",
            "score": sum(i * p for i, p in enumerate(probs)),
            "confidence": _confidence(probs),
            "probabilities": {str(i): probs[i] for i in reversed(range(n))},
            "legend": {str(i): levels[i] for i in range(n)},
        }

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/healthz")
        def healthz():
            return {"status": "ok"}

        @app.post("/v1/systemone")
        def systemone(body: dict[str, Any] = Body(...)):
            cfg = self.config
            with self._lock:
                cfg.requests.append(body)
                idx = len(cfg.requests)
            if cfg.delay_ms:
                time.sleep(cfg.delay_ms / 1000)
            if idx <= cfg.fail_first:
                return JSONResponse({"detail": {"error_type": "overloaded_error", "message": "busy"}},
                                    status_code=cfg.fail_status, headers={"retry-after-ms": "10"})
            qs = body.get("questions") or {}
            if any(q.get("instructions") in cfg.fail_instructions for q in qs.values()):
                return JSONResponse({"detail": {"error_type": "internal_error", "message": "boom"}}, status_code=500)
            answers = {name: self._answer(body.get("state"), q) for name, q in qs.items()}
            return {"model": "fake-1", "answers": answers, "usage": {"input_tokens": 1, "output_tokens": 1}}

        return app

    def start(self) -> "FakeServer":
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self.app, log_level="warning", lifespan="off"))
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, daemon=True)
        self._thread.start()
        deadline = time.time() + 10
        while not self._server.started:
            if time.time() > deadline:
                raise RuntimeError("fake server did not start")
            time.sleep(0.01)
        self.url = f"http://127.0.0.1:{port}"
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
            self._thread.join(timeout=5)
