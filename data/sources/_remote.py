"""Partial-fetch helpers for the image converters: read only the parquet row groups, zip members, tar members
and single files a run needs, with retries (the HF CDN drops TLS connections under load)."""
import os
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import quote

from data.sources._common import DATA_DIR

CACHE = Path(os.environ.get("VEV_SOURCE_CACHE", DATA_DIR / "raw" / "_cache"))  # small metadata files only


def retry(fn, what: str, tries: int = 8, fatal=(KeyError, ValueError, FileNotFoundError)):
    for i in range(tries):
        try:
            return fn()
        except fatal:
            raise
        except Exception as e:  # network errors surface as many different types (httpx, ssl, OSError, fsspec)
            if i == tries - 1:
                raise
            wait = min(60, 5 * 2 ** i)
            print(f"  retry {i + 1}/{tries - 1} {what}: {type(e).__name__}: {str(e)[:120]} (sleep {wait}s)",
                  file=sys.stderr)
            time.sleep(wait)


_FS = None


def hf_fs():
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem()
    return _FS


def hf_path(repo: str, path: str, revision: str | None = None) -> str:
    rev = f"@{quote(revision, safe='')}" if revision else ""
    return f"datasets/{repo}{rev}/{path}"


def hf_open(repo: str, path: str, revision: str | None = None, block_size: int = 1 << 20):
    from data.sources._localhf import ENABLED, ensure_local
    if ENABLED:  # whole-file download once, then plain local reads (cluster: fast and never cut off)
        return open(ensure_local(repo, path, revision), "rb")
    return retry(lambda: hf_fs().open(hf_path(repo, path, revision), "rb", block_size=block_size), f"open {path}")


def hf_download(repo: str, filename: str, revision: str | None = None) -> str:
    """Whole-file download into the HF cache: for metadata files only (csv / json / small parquet)."""
    from huggingface_hub import hf_hub_download
    return retry(lambda: hf_hub_download(repo, filename, repo_type="dataset", revision=revision), f"download {filename}")


def http_bytes(url: str, timeout: int = 120) -> bytes:
    """GET with the env proxy (urllib reads HTTPS_PROXY / HTTP_PROXY)."""
    from data.sources._localhf import local_for_url
    local = local_for_url(url)
    if local is not None:
        return local.read_bytes()

    def get():
        req = urllib.request.Request(url, headers={"User-Agent": "vev-data/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    return retry(get, url)


def http_cached(url: str, source: str, name: str) -> Path:
    """Download a single metadata file once into CACHE/<source>/<name>."""
    dest = CACHE / source / name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        part.write_bytes(http_bytes(url, timeout=3600))
        part.replace(dest)
    return dest


# ---------------------------------------------------------------- parquet: row groups as sampling units

def parquet_units(repo: str, files: list[str], revision: str | None):
    """[(file, row_group, n_rows, metadata)] from the footers only."""
    import pyarrow.parquet as pq
    units = []
    for f in files:
        md = retry(lambda: pq.ParquetFile(hf_open(repo, f, revision, block_size=1 << 16)).metadata, f"footer {f}")
        units += [(f, g, md.row_group(g).num_rows, md) for g in range(md.num_row_groups)]
    return units


def iter_parquet(repo: str, files: list[str], revision: str | None, columns: list[str], *, seed: int, limit: int,
                 per_unit: int | None = None):
    """Rows of the given columns, one row group at a time. With a limit the row groups are visited in a seeded
    random order, rows inside a group are shuffled, and at most `per_unit` rows are taken per group (so a small
    --limit touches few row groups but still more than one). Each row gets `_file`, `_rg`, `_row`."""
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    units = parquet_units(repo, files, revision)
    if limit:
        rng.shuffle(units)
    for f, g, _, md in units:
        def read():
            with hf_open(repo, f, revision, block_size=1 << 20) as fh:
                return pq.ParquetFile(fh, metadata=md).read_row_group(g, columns=columns).to_pylist()
        rows = retry(read, f"{f} row group {g}")
        order = list(range(len(rows)))
        if limit:
            rng.shuffle(order)
            if per_unit:
                order = order[:per_unit]
        for i in order:
            row = rows[i]
            row.update(_file=f, _rg=g, _row=i)
            yield row


# ---------------------------------------------------------------- zip: central directory + single members

class RemoteZip:
    """zipfile over a range-read file object: listing reads the central directory, read() fetches one member."""

    def __init__(self, opener, name: str):
        self.opener, self.name, self._z = opener, name, None

    def _zip(self):
        if self._z is None:
            self._z = retry(lambda: zipfile.ZipFile(self.opener()), f"zip directory {self.name}")
        return self._z

    def namelist(self) -> list[str]:
        return self._zip().namelist()

    def read(self, member: str) -> bytes:
        def get():
            try:
                return self._zip().read(member)
            except KeyError:
                raise
            except Exception:
                self._z = None  # reopen the connection on the next attempt
                raise
        return retry(get, f"{self.name}:{member}")


def image_source_args(name: str, extra=None):
    """source_args() with the default --out under data/raw/image/ (source_args picks the bucket by looking for
    'image' in the source name, which sends e.g. 'sugarcrepe' to raw/text)."""
    from data.sources._common import source_args
    args = source_args(name, extra)
    if not any(a == "--out" or a.startswith("--out=") for a in sys.argv[1:]):
        args.out = str(DATA_DIR / "raw" / "image" / f"{name}.jsonl")
    return args
