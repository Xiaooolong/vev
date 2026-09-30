# System One API specification

Version 0.1, 2026-09-23. The authoritative definition of this project's server API. It has the same shape as TypeSafe's `/v1/systemone`; where the official API leaves something unspecified, this document pins it down; images are an extension of this project.

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

Every response carries the header `x-typesafe-request-id` (UUID).

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

- `state`: a string, JSON object or array. Objects and arrays may nest arbitrarily. Strings, numbers and booleans are all treated as text. `null` returns 422 (observed on the official API, although the SDK type allows null).
- `model`: required. Accepted values: `vev-latest`, `jev-latest` (the official SDK default, mapped to this server's default model), `latest`, and the concrete ids listed by `GET /v1/models` (for example `vev-4b` or `vev-9b`). Unknown values return 422.
- `questions`: a non-empty object. Keys are caller-defined names, used only to key the answers and never sent to the model.

### 3.1 Question

Three types; `type` is required. `instructions` may be omitted for all three and then defaults to the empty string (the official api.md marks it required, the SDK marks it optional; this server treats it as optional).

**noul**

```json
{"type": "noul", "instructions": "…", "criteria": {"true": <desc>, "false": <desc>}}
```

`criteria` may be omitted; `true` and `false` may each be omitted.

**choice**

```json
{"type": "choice", "instructions": "…", "criteria": {"<label>": <desc>, ...}}
```

`criteria` is required, 1 to 255 labels (the official docs only state the upper bound of 255; with a single option the response gives it probability 1.0 and confidence 1.0, as kev does). A label is any non-empty string; order follows the key order of the JSON object.

**score**

```json
{"type": "score", "instructions": "…", "criteria": [<desc0>, <desc1>, ...]}
```

`criteria` is required, ordered from low to high, 1 to 255 levels (the official docs say "at least two and at most 10 levels"; observed, 1 level returns 200 with probability 1.0 and 11 levels return 400. This server matches the observed lower bound and raises the upper bound to 255, as kev does; the model card states the largest number of levels seen in training).

`<desc>` may be a string, object, array or null; the server serializes it as is and passes it to the model.

### 3.2 Limits

| Item | Value | Source |
|---|---|---|
| total tokens per request | 65,536 | same as official |
| state + longest single question | 32,768 | same as official |
| questions per request | no hard limit; the model card states the largest number tested | not found in official docs |
| images per request | model card `limits.max_images` | this project |
| pixels per image | model card `limits.max_image_pixels`; larger images are downscaled proportionally to the limit | this project |

Tokens are counted with the model's own tokenizer and image processor; image tokens count towards the state.

## 4. Image extension

Put this reserved object anywhere in `state` (top level, an object value, an array element):

```json
{"image": {"url": "data:image/png;base64,<...>"}}
```

- Detection rule: in any object of the state tree, a key named `image` whose value is an object with a string field `url` is an image. When rendering, that key is replaced by vision tokens at its position in the parent object's key order; the parent's other keys are rendered as text as usual. So `{"screenshot": {"image": {...}}}` and `{"context": "…", "image": {...}}` are both valid. An `image` key whose value is not an object, or is an object without a string `url`, is treated as plain JSON text without an error.
- `url` accepts only `data:` URIs with MIME type `image/png`, `image/jpeg` or `image/webp`. `http(s)` URLs are rejected by default (422); the server flag `--allow-remote-images` enables them, with a 10 s fetch timeout.
- Serialization order: the server walks the state in order (objects by key order, arrays by index), turning text into tokens and image objects into vision tokens, preserving their relative positions. Multiple images are allowed.
- The official Jev API reads such an object as plain JSON without an error; it just does not see the image.

Bad images (decode failure, unsupported MIME type, over `max_images`) return 422, with `error.message` stating which image and what is wrong.

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
- `confidence = (p_max − 1/K) / (1 − 1/K)`, where K is the number of options; 1.0 when K = 1. The official formula is not public; this one matches kev and is stated in the model card.

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
| cold start | at startup the server warms up with one request carrying an image and three questions; afterwards every request (including the first external one) takes ≤ 5 s on the stated hardware. The official Python SDK defaults to a 10 s timeout with 2 retries, and kev-4b on an H800 times out on the first question from a cold start | conformance "clients" group, run as the first test |
| isolation and precision | question isolation ≤ 1e-4 holds at the server's default precision (bf16), not only fp32. kev-4b in bf16 differs by 0.0068 between batched and single-question requests; the 4e-6 in its README holds only in fp32 | conformance "isolation" group, run at default precision |

## 7. Extensions

The default response contains no extra fields. The request header `X-Vev-Extensions` lists the wanted extensions (comma-separated), and the response gains a top-level `extensions` object:

| Extension | Content |
|---|---|
| `logits` | raw pre-calibration scores per question |
| `timing` | `{"prefill_ms", "questions_ms", "total_ms"}` |
| `tokens` | token counts of the three parts `{"state_text", "state_image", "questions"}` |

Unknown extension names are ignored without an error.

## 8. Errors

Status codes and body shapes follow the official API as observed (2026-09-23, jev-1.13.0). The official server is FastAPI, and error bodies are always under the `detail` key; the official SDK extracts the message from `detail` (a string, the `message` of an object, or a list), and so does langchain-typesafe.

| Status | When | `detail` shape | Official example |
|---|---|---|---|
| 422 | body does not match the schema: missing `questions`, missing `model`, empty `questions` object, `state` is null, choice without `criteria`, JSON parse failure | FastAPI list `[{"type","loc","msg","input"}]` | `{"detail":[{"type":"missing","loc":["body","questions"],"msg":"Field required",…}]}` |
| 400 | semantic errors or limits: unknown `type`, unknown `model`, choice with 0 or more than 255 options, score with more than 10 levels (this server: more than 255), bad image, image limits exceeded | string, or `{"error_type","message"}` | `{"detail":"Too many choices. Must have at most 255 choices."}`; `{"detail":{"error_type":"api_usage_error","message":"Unknown model: no-such-model-xyz"}}` |
| 401 | key checking enabled and the key does not match | `{"error_type":"authentication_error","message"}` | same wording as official |
| 429 | queue full; carries the header `retry-after-ms` | `{"error_type":"rate_limit_error","message"}` | not observed |
| 529 | server overloaded | `{"error_type":"overloaded_error","message"}` | not observed |
| 500 | anything else | `{"error_type":"internal_error","message"}` | — |

This server always uses the object shape `{"error_type","message"}` for 400, with `message` pointing at the exact path (`questions.<name>.<field>`, or which image in `state.<path>`).

## 9. GET /v1/models

```json
{"models": [
  {"name": "vev-4b", "description": "…", "release_date": "2026-..",
   "limits": {"max_images": 8, "max_image_pixels": 4194304, "max_state_tokens": 32768,
              "max_request_tokens": 65536, "max_score_levels": 255, "max_choice_options": 255},
   "aliases": ["vev-latest", "jev-latest", "latest"]}
]}
```

`name`, `description` and `release_date` have the same shape as the official ModelCard; `limits` and `aliases` are fields added by this project, and the official SDK's type definitions ignore unknown fields (verified by the conformance suite).

## 10. Observations on the official API (2026-09-23, jev-1.13.0, conformance suite)

- `instructions` omitted: 200. noul without `criteria`: 200. choice with 1 option: 200, probability 1.0, confidence 1.0.
- `x-typesafe-request-id` looks like `req_01a0cc8b…`. `/healthz` does not exist (404).
- `GET /v1/models` returns `{"models":[{"name":"jev-latest",…},{"name":"jev-preview",…}]}`, with `release_date` as an ISO timestamp.
- `model` is filled with `jev-1.13.0`.
- Probabilities are quantized to 0.01; `score` is computed from the unquantized values and can differ by up to 0.03 (10 levels) from the expectation over the quantized probabilities. This server does not quantize.
- On the same state, 5 questions batched vs. one at a time differ by 0.01 for noul and up to 0.04 for score; the order of questions in the request also moves the probabilities (above the quantization step). The official isolation is not strict. This server requires ≤ 1e-4.
- Content in sibling questions is not visible to another question (probe 0.06 -> 0.06); put in the state, it is (0.81).
- The same request sent 5 times gives 4 different outputs, max difference 0.02. The official API is not deterministic. This server requires bit-identical outputs.
- Latency (client through a proxy to api.typesafe.ai): 3 questions 305 ms, 50 options 278 ms, 23 questions 325 ms; essentially independent of the number of questions and options.
- 429/529 not observed; their shapes in §8 are inferred.

## 11. Observations on kev-4b (2026-09-23, H800, bf16)

- 31 passed, 4 failed. Failures: the official SDK's first request hit ReadTimeout (cold start > 10 s); batched vs. single-question max |Δp| 0.0068; unknown `model` returns 200 and is echoed back; a one-level score returns 200 (same as official; an older version of the test misjudged it).
- Determinism 0.0, question order 0.0, sibling questions not visible (0.043 / 0.0435 / 0.9908).
- Shape differences from the official API: `model` is filled with the alias `jev-latest` instead of a concrete id; an extra top-level `latency_ms` in the response; `x-typesafe-request-id` is a bare UUID; probabilities have 4 decimals; the open server does not check keys; unknown type and limit violations go through pydantic 422 instead of the official 400. This server follows the official behaviour (§8).
- Latency (H800): 1 question 51 ms, 10 questions 54, 50 questions 70, 100 questions 127 (2068 tokens); state of 641 tokens 54 ms, 2356 tokens 57, 6927 tokens 116. Occasional outliers of 1–10 s (the max column), cause not investigated.
