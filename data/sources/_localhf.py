"""Local-first access to Hugging Face dataset files.

Where HTTP range reads are unreliable but whole-file downloads work, set VEV_HF_LOCAL=1: every helper that would
range-read an HF file first downloads it into the HF cache (HF_HOME) and serves bytes from disk. Off by default,
so the converters stream only the parts they need.
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

ENABLED = os.environ.get("VEV_HF_LOCAL", "0") == "1"
_URL = re.compile(r"^https://huggingface\.co/datasets/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")
_HFPATH = re.compile(r"^datasets/([^/@]+/[^/@]+)(?:@([^/]+))?/(.+)$")


def parse_url(url: str) -> tuple[str, str, str] | None:
    m = _URL.match(url)
    return (m.group(1), unquote(m.group(2)), m.group(3)) if m else None


def parse_hf_path(path: str) -> tuple[str, str | None, str] | None:
    m = _HFPATH.match(path)
    return (m.group(1), unquote(m.group(2)) if m.group(2) else None, m.group(3)) if m else None


def ensure_local(repo: str, path: str, revision: str | None, tries: int = 6) -> Path:
    from huggingface_hub import hf_hub_download

    for k in range(tries):
        try:
            t0 = time.time()
            p = Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision))
            if time.time() - t0 > 5:
                print(f"  downloaded {repo}/{path} ({p.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)",
                      file=sys.stderr, flush=True)
            return p
        except Exception as e:  # noqa: BLE001 - transient hub/network errors
            if k == tries - 1:
                raise
            print(f"  download retry {repo}/{path}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(min(60, 10 * (k + 1)))
    raise RuntimeError("unreachable")


def local_for_url(url: str) -> Path | None:
    if not ENABLED:
        return None
    parsed = parse_url(url)
    if not parsed:
        return None
    repo, rev, path = parsed
    return ensure_local(repo, path, rev)


def local_for_hf_path(path: str) -> Path | None:
    if not ENABLED:
        return None
    parsed = parse_hf_path(path)
    return ensure_local(parsed[0], parsed[2], parsed[1]) if parsed else None
