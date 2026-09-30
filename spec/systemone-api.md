# System One API specification

Version 0.1. The HTTP API served by `vev serve`. Request and response shapes match TypeSafe's `/v1/systemone`; where the official API leaves something unspecified, this document says what Vev does. Images are an extension.

## 1. Compatibility

- Any text-only client written with the official `typesafe-sdk` (Python), `@typesafe-ai/sdk` (JS), `langchain-typesafe`, jeview or judgekit works after pointing its base URL at this server.
- The default response has exactly the official field set. Extension fields appear only when the client explicitly asks for them (§7).
- Images go inside `state` as a reserved object (§4), so to a client a request with images is just "one more object in the state".

## 2. Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/systemone` | judgment |
| GET | `/v1/models` | list of model cards, including this project's limit fields |
| GET | `/healthz` | liveness; returns `{"status":"ok"}` |

Authentication: `Authorization: Bearer <key>`. By default the server does not check the key (any non-empty string is accepted; the official SDK requires a non-empty printable-ASCII key). With the environment variable `VEV_API_KEYS` set (comma-separated), keys are checked and a mismatch returns 401.

Every response carries the header `x-typesafe-request-id` (`req_` followed by 32 hex digits).

## 3. Request

```json
{
  "state": <string | object | array | null>,
  "model": "jev-latest",
  "questions": {
    "<name>": <Question>,
    ...
  }
}
```

- `state`: a string, JSON object or array. Objects and arrays may nest arbitrarily. Strings, numbers and booleans are all treated as text. `null` returns 422, as on the official API.
- `model`: required. Accepted values: `vev-latest`, `jev-latest` (the official SDK default, mapped to this server's default model), `latest`, and the concrete id listed by `GET /v1/models` (for example `vev-4b`). Unknown values return 400.
- `questions`: a non-empty object. Keys are caller-defined names, used only to key the answers and never sent to the model.

### 3.1 Question

Three types; `type` is required. `instructions` may be omitted and then defaults to the empty string.

**noul**

```json
{"type": "noul", "instructions": "…", "criteria": {"true": <desc>, "false": <desc>}}
```

`criteria` may be omitted; `true` and `false` may each be omitted.

**choice**

```json
{"type": "choice", "instructions": "…", "criteria": {"<label>": <desc>, ...}}
```

`criteria` is required, 1 to 255 labels; a single option gets probability 1.0 and confidence 1.0. A label is any non-empty string; order follows the key order of the JSON object.

**score**

```json
{"type": "score", "instructions": "…", "criteria": [<desc0>, <desc1>, ...]}
```

`criteria` is required, ordered from low to high, 1 to 10 levels, the same bounds the official API accepts; a single level gets probability 1.0.

`<desc>` may be a string, object, array or null; the server serializes it as is and passes it to the model.

### 3.2 Limits

| Item | Value | Source |
|---|---|---|
| total tokens per request | 65,536 | same as official |
| state + longest single question | 32,768 | same as official |
| questions per request | no hard limit | |
| images per request | 8 (`limits.max_images`) | Vev |
| pixels per image | 1,048,576 (`limits.max_image_pixels`); larger images are downscaled to the limit | Vev |

Tokens are counted with the model's own tokenizer and image processor; image tokens count towards the state.

## 4. Image extension

Put this reserved object anywhere in `state` (top level, an object value, an array element):

```json
{"image": {"url": "data:image/png;base64,<...>"}}
```

- Detection rule: in any object of the state tree, a key named `image` whose value is an object with a string field `url` is an image. When rendering, that key is replaced by vision tokens at its position in the parent object's key order; the parent's other keys are rendered as text as usual. So `{"screenshot": {"image": {...}}}` and `{"context": "…", "image": {...}}` are both valid. An `image` key whose value is not an object, or is an object without a string `url`, is treated as plain JSON text without an error.
- `url` accepts only `data:` URIs with MIME type `image/png`, `image/jpeg` or `image/webp`. `http(s)` URLs are rejected by default (400); the server flag `--allow-remote-images` enables them, with a 10 s fetch timeout.
- Serialization order: the server walks the state in order (objects by key order, arrays by index), turning text into tokens and image objects into vision tokens, preserving their relative positions. Multiple images are allowed.
- The official Jev API reads such an object as plain JSON without an error; it just does not see the image.

Bad images (decode failure, unsupported MIME type, over `max_images`) return 400, with `detail.message` stating which image and what is wrong.

## 5. Response

```json
{
  "model": "<concrete model id>",
  "answers": {
    "<name>": <Answer>,
    ...
  },
  "usage": {"input_tokens": <int>, "output_tokens": <int>}
}
```

`model` is the concrete id actually served, never an alias.

### 5.1 Answer

**noul**

```json
{"type": "noul", "noul": 0.87}
```

`noul` is P(true), a float in [0, 1]. No confidence (same as official).

**choice**

```json
{"type": "choice", "choice": "technical", "confidence": 0.82,
 "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07}}
```

- The key set of `probabilities` equals the key set of the request's `criteria`, and the values sum to 1 (error ≤ 1e-6). Key order is not guaranteed (on the official API, two identical requests can return different key orders); clients must look up by key, not by position. This server returns keys in request order but does not promise it.
- `choice` is the key with the highest probability; ties go to the key that comes first in the request.
- `confidence = (p_max − 1/K) / (1 − 1/K)`, where K is the number of options; 1.0 when K = 1. The same formula is used by TypeSafe's reference adapter.

**score**

```json
{"type": "score", "score": 1.43, "confidence": 0.35,
 "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
 "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"}}
```

- The keys of `probabilities` and `legend` are level indices as strings, `"0"`…`"n-1"` (langchain-typesafe converts them to int, so no other key form is allowed).
- `score = Σ i × p_i`, a float.
- `confidence` uses the choice formula with K = number of levels.

Probabilities are not quantized (the official API quantizes to 0.01; this server keeps float precision to avoid ties in confidence).

### 5.2 usage

- `input_tokens`: tokens of the state (including image tokens) and all questions, counted on the sequence actually fed to the model.
- `output_tokens`: tokens of the serialized `answers` JSON. Informational only and not comparable with official billing; the model card says so.

## 6. Semantic guarantees

| Guarantee | Content | Verified by |
|---|---|---|
| question isolation | on the same state, N questions sent in one request vs. one at a time differ by ≤ 1e-4 in every probability (fp32 inference); the `instructions`/`criteria` of one question have no effect on the answers to the others | conformance "isolation" group |
| determinism | the same request sent again gives bit-identical probabilities | conformance "determinism" group |
| no generation | the server does no autoregressive decoding; latency is roughly independent of the number of questions (100 questions ≤ 1.5 × 1 question) | conformance "curves" group, written to JSON |
| option order | not guaranteed to be invariant; the model card reports the flip rate after reversing the options | eval suite |
| cold start | at startup the server warms up with one request carrying an image and three questions, so the first external request is not slow (the official Python SDK times out after 10 s) | conformance "clients" group, run as the first test |
| isolation and precision | question isolation ≤ 1e-4 holds at the server's default precision (bf16), not only fp32 | conformance "isolation" group, run at default precision |

## 7. Extensions

The default response contains no extra fields. The request header `X-Vev-Extensions` lists the wanted extensions (comma-separated), and the response gains a top-level `extensions` object:

| Extension | Content |
|---|---|
| `logits` | answer-token logits per question, before the softmax |
| `timing` | `{"prefill_ms", "questions_ms", "total_ms"}` |
| `tokens` | token counts of the three parts `{"state_text", "state_image", "questions"}` |

Unknown extension names are ignored without an error.

## 8. Errors

Status codes and body shapes follow the official API as observed on 2026-09-23 (jev-1.13.0). The official server is FastAPI, and error bodies are always under the `detail` key; the official SDK extracts the message from `detail` (a string, the `message` of an object, or a list), and so does langchain-typesafe.

| Status | When | `detail` shape | Official example |
|---|---|---|---|
| 422 | body does not match the schema: missing `questions`, missing `model`, empty `questions` object, `state` is null, choice without `criteria`, JSON parse failure | FastAPI list `[{"type","loc","msg","input"}]` | `{"detail":[{"type":"missing","loc":["body","questions"],"msg":"Field required",…}]}` |
| 400 | semantic errors or limits: unknown `type`, unknown `model`, choice with 0 or more than 255 options, score with more than 10 levels, bad image, image limits exceeded | string, or `{"error_type","message"}` | `{"detail":"Too many choices. Must have at most 255 choices."}`; `{"detail":{"error_type":"api_usage_error","message":"Unknown model: no-such-model-xyz"}}` |
| 401 | key checking enabled and the key does not match | `{"error_type":"authentication_error","message"}` | same wording as official |
| 429 | queue full; carries the header `retry-after-ms` | `{"error_type":"rate_limit_error","message"}` | not observed |
| 529 | server overloaded | `{"error_type":"overloaded_error","message"}` | not observed |
| 500 | anything else | `{"error_type":"internal_error","message"}` | — |

Vev always uses the object shape `{"error_type","message"}` for 400, with `message` pointing at the exact path (`questions.<name>.<field>`, or which image in `state.<path>`).

## 9. GET /v1/models

```json
{"models": [
  {"name": "vev-4b", "description": "…", "release_date": "2026-..",
   "limits": {"max_images": 8, "max_image_pixels": 1048576, "max_state_tokens": 32768,
              "max_request_tokens": 65536, "max_score_levels": 10, "max_choice_options": 255},
   "aliases": ["vev-latest", "jev-latest", "latest"]}
]}
```

`name`, `description` and `release_date` have the same shape as the official ModelCard; `limits` and `aliases` are Vev additions, and the official SDK's type definitions ignore unknown fields (verified by the conformance suite).

Behaviour of the official API observed with the conformance suite is recorded in `conformance/README.md`.
