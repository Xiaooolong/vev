"""FastAPI server for /v1/systemone (spec/systemone-api.md)."""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import uuid
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from vev.readout import MAX_CHOICE_OPTIONS, MAX_SCORE_LEVELS, Engine
from vev.state import InvalidRequest

log = logging.getLogger("vev")

ALIASES = ["vev-latest", "jev-latest", "latest"]
MAX_IMAGES = 8
MAX_IMAGE_PIXELS = 1024 * 1024
MAX_STATE_TOKENS = 32768
MAX_REQUEST_TOKENS = 65536
KNOWN_TYPES = ("noul", "choice", "score")
EXTENSIONS = ("logits", "timing", "tokens")


class Question(BaseModel):
    type: str
    instructions: str | None = ""
    criteria: Any = None

    @model_validator(mode="after")
    def _criteria_shape(self) -> Question:
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("choice questions require `criteria` as an object of label -> description")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("score questions require `criteria` as an array of level descriptions")
        if self.type == "noul" and self.criteria is not None and not isinstance(self.criteria, dict):
            raise ValueError("noul `criteria` must be an object with optional `true`/`false` keys")
        return self


class SystemOneRequest(BaseModel):
    state: Any
    model: str
    questions: dict[str, Question] = Field(min_length=1)

    @field_validator("state")
    @classmethod
    def _state_not_null(cls, v: Any) -> Any:
        if v is None:
            raise ValueError("state must not be null")
        return v


def bad(message: str, error_type: str = "invalid_request_error") -> InvalidRequest:
    return InvalidRequest(400, error_type, message)


def validate_semantics(req: SystemOneRequest, model_name: str) -> None:
    if req.model not in ALIASES and req.model != model_name:
        raise bad(f"Unknown model: {req.model}", "api_usage_error")
    for name, q in req.questions.items():
        where = f"questions.{name}"
        if q.type not in KNOWN_TYPES:
            raise bad(f"{where}.type: unknown question type {q.type!r}; expected one of {list(KNOWN_TYPES)}")
        if q.type == "choice":
            k = len(q.criteria)
            if k == 0:
                raise bad(f"{where}.criteria: choice needs at least 1 option")
            if k > MAX_CHOICE_OPTIONS:
                raise bad(f"{where}.criteria: too many choices ({k}); this model supports at most {MAX_CHOICE_OPTIONS}")
            if any(label == "" for label in q.criteria):
                raise bad(f"{where}.criteria: labels must be non-empty strings")
        elif q.type == "score":
            n = len(q.criteria)
            if n == 0:
                raise bad(f"{where}.criteria: score needs at least 1 level")
            if n > MAX_SCORE_LEVELS:
                raise bad(f"{where}.criteria: too many score levels ({n}); this model supports at most {MAX_SCORE_LEVELS}")


def model_card(name: str, hf_id: str) -> dict[str, Any]:
    return {
        "name": name,
        "description": (
            f"{name} (base {hf_id}): answers are read from next-token logits over constrained answer tokens "
            "(choice letters, Yes/No, level digits). One forward pass per question, no decoding. "
            "confidence = (p_max - 1/K) / (1 - 1/K). output_tokens is informational only."
        ),
        "release_date": "2026-10-01T00:00:00Z",
        "limits": {
            "max_images": MAX_IMAGES,
            "max_image_pixels": MAX_IMAGE_PIXELS,
            "max_state_tokens": MAX_STATE_TOKENS,
            "max_request_tokens": MAX_REQUEST_TOKENS,
            "max_score_levels": MAX_SCORE_LEVELS,
            "max_choice_options": MAX_CHOICE_OPTIONS,
        },
        "aliases": ALIASES,
    }


def err(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"error_type": error_type, "message": message}})


def create_app(engine: Engine, model_name: str, hf_id: str, allow_remote_images: bool) -> FastAPI:
    app = FastAPI(title="vev", docs_url=None, redoc_url=None)
    lock = threading.Lock()
    keys = {k.strip() for k in os.environ.get("VEV_API_KEYS", "").split(",") if k.strip()}

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = "req_" + uuid.uuid4().hex
        if keys and request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
            if token not in keys:
                resp = err(401, "authentication_error", "Invalid API key")
                resp.headers["x-typesafe-request-id"] = rid
                return resp
        try:
            resp = await call_next(request)
        except Exception:
            log.exception("unhandled error in %s", rid)
            resp = err(500, "internal_error", f"Internal server error; see the server log for request {rid}")
        resp.headers["x-typesafe-request-id"] = rid
        return resp

    @app.exception_handler(InvalidRequest)
    async def invalid_request(_: Request, e: InvalidRequest):
        return err(e.status, e.error_type, e.message)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/v1/models")
    def models():
        return {"models": [model_card(model_name, hf_id)]}

    @app.post("/v1/systemone")
    def systemone(req: SystemOneRequest, request: Request):
        validate_semantics(req, model_name)
        wanted = {x.strip().lower() for x in request.headers.get("x-vev-extensions", "").split(",")} & set(EXTENSIONS)
        questions = {name: q.model_dump() for name, q in req.questions.items()}
        with lock:
            result = engine.run(
                req.state,
                questions,
                max_images=MAX_IMAGES,
                max_pixels=MAX_IMAGE_PIXELS,
                allow_remote=allow_remote_images,
                max_state_tokens=MAX_STATE_TOKENS,
                max_request_tokens=MAX_REQUEST_TOKENS,
            )
        body: dict[str, Any] = {
            "model": model_name,
            "answers": result.answers,
            "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        }
        if wanted:
            body["extensions"] = {k: result.extensions[k] for k in EXTENSIONS if k in wanted}
        return body

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vev serve", description="Serve a Vev model (or a plain Qwen3.5 model, zero-shot) "
                                 "behind a /v1/systemone-compatible HTTP API.")
    ap.add_argument("--model", required=True,
                    help="Vev checkpoint (local directory or Hugging Face repo id), or a Qwen3.5 model id for zero-shot use")
    ap.add_argument("--revision", default=None, help="Hugging Face revision of --model (branch, tag or commit), e.g. v0.1.0")
    ap.add_argument("--name", default=None, help="served model id; default derived from --model")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-remote-images", action="store_true")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--prefix-min-tokens", type=int, default=4096,
                    help="Vev checkpoints: prefill the state once and share it across questions for states of at least "
                         "this many tokens")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import torch

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"--device {a.device} but this torch build has no CUDA ({torch.__version__}). Install a CUDA "
                         "build of PyTorch (see https://pytorch.org/get-started/locally/) or pass --device cpu.")
    from vev.model import CheckpointEngine, is_checkpoint

    if is_checkpoint(a.model, a.revision):
        engine = CheckpointEngine(a.model, dtype=a.dtype, device=a.device, prefix_min_tokens=a.prefix_min_tokens,
                                  revision=a.revision)
        name = a.name or "vev-" + os.path.basename(os.path.normpath(a.model)).lower().removeprefix("vev-")
        hf_id = engine.base_id
    else:
        engine = Engine(a.model, dtype=a.dtype, device=a.device, revision=a.revision)
        name = a.name or "zeroshot-" + a.model.split("/")[-1].lower()
        hf_id = a.model
    if not a.no_warmup:
        t0 = time.perf_counter()
        engine.warmup()
        log.info("warmup done in %.0f ms", (time.perf_counter() - t0) * 1000)
    app = create_app(engine, name, hf_id, a.allow_remote_images)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
