"""Byte-range access for the group-3 converters (GUI / game / zh): read only the archive members a run needs.

RemoteZip handles plain and split (`zip -s`: .z01 ... .zNN + .zip) archives, which zipfile cannot; the central
directory is fetched once and cached under data/raw/_cache/. Network goes through requests (honours HTTPS_PROXY).
"""
import hashlib
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import requests

from data.sources._common import DATA_DIR
from data.sources._localhf import local_for_url

CACHE = DATA_DIR / "raw" / "_cache"
_S = requests.Session()
# raw.githubusercontent.com gzips responses: Content-Range/Content-Length then describe the compressed body while
# requests hands back the decompressed bytes, so sizes and slices disagree. Ask for identity encoding everywhere.
_S.headers["Accept-Encoding"] = "identity"


def hf_url(repo: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def gh_raw(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def retry(fn, tries: int = 8):
    for k in range(tries):
        try:
            return fn()
        except (requests.RequestException, OSError) as e:  # the proxy drops TLS connections now and then
            if k == tries - 1:
                raise
            print(f"  retry after {type(e).__name__}: {str(e)[:120]}", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * (k + 1)))


def get_range(url: str, start: int, end: int, chunk: int = 4 << 20) -> bytes:
    """Bytes [start, end) of url, in chunks (long single responses get cut off by the proxy)."""
    local = local_for_url(url)
    if local is not None:
        with local.open("rb") as fh:
            fh.seek(start)
            return fh.read(end - start)
    if end - start > chunk:
        return b"".join(get_range(url, s, min(end, s + chunk), chunk) for s in range(start, end, chunk))
    if end <= start:
        return b""

    def go():
        r = _S.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
        r.raise_for_status()
        if len(r.content) == end - start:
            return r.content
        if len(r.content) >= end:
            # raw.githubusercontent.com (even with a 206) and some CDNs ignore Range and send the whole file
            return r.content[start:end]
        raise OSError(f"short read {len(r.content)} != {end - start} from {url} (status {r.status_code})")
    return retry(go)


def remote_size(url: str) -> int:
    local = local_for_url(url)
    if local is not None:
        return local.stat().st_size

    def go():
        r = _S.get(url, headers={"Range": "bytes=0-0"}, timeout=120)
        r.raise_for_status()
        return int(r.headers["Content-Range"].rsplit("/", 1)[1])
    return retry(go)


def fetch(url: str, name: str | None = None) -> Path:
    """Whole (small) file, cached under data/raw/_cache/ keyed by url."""
    key = hashlib.sha1(url.encode()).hexdigest()[:12]
    dest = CACHE / f"{key}_{name or url.rsplit('/', 1)[-1]}"
    size = remote_size(url)
    if dest.exists() and dest.stat().st_size != size:  # a truncated copy from an interrupted earlier run
        print(f"  cache {dest.name} is {dest.stat().st_size} bytes, expected {size}; re-fetching", file=sys.stderr, flush=True)
        dest.unlink()
    if not dest.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        data = get_range(url, 0, size)  # chunked: one long response gets cut off by the proxy
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        print(f"  fetched {url} ({len(data) / 1e6:.1f} MB, {time.time() - t0:.0f}s)", file=sys.stderr, flush=True)
    return dest


def fetch_bytes(url: str) -> bytes:
    local = local_for_url(url)
    if local is not None:
        return local.read_bytes()

    def go():
        r = _S.get(url, timeout=600)
        r.raise_for_status()
        return r.content
    return retry(go)


def stream(url: str, chunk: int = 1 << 16, tries: int = 8):
    """Body of url in chunks, for sequential formats (gzip TFRecord); stop iterating to stop downloading.
    Resumes with a Range request from the last byte received when the connection drops mid-body."""
    local = local_for_url(url)
    if local is not None:
        with local.open("rb") as fh:
            while part := fh.read(chunk):
                yield part
        return
    offset = 0
    for k in range(tries):
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        r = retry(lambda: _S.get(url, stream=True, timeout=120, headers=headers))
        r.raise_for_status()
        if offset and r.status_code != 206:
            r.close()
            raise OSError(f"cannot resume {url} at {offset}: server ignored Range (status {r.status_code})")
        try:
            for part in r.iter_content(chunk):
                offset += len(part)
                yield part
            return
        except (requests.RequestException, OSError) as e:
            if k == tries - 1:
                raise
            print(f"  stream dropped at {offset} ({type(e).__name__}); resuming", file=sys.stderr, flush=True)
            time.sleep(min(60, 5 * (k + 1)))
        finally:
            r.close()


class RemoteZip:
    """Random access to members of a remote zip. parts = [url, ...] in disk order (.z01, ..., .zip) or one url."""

    def __init__(self, parts: list[str], sizes: list[int] | None = None):
        self.parts = parts
        cache = CACHE / f"zipcd_{hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16]}.json"
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            self.sizes, self.entries = cached["sizes"], cached["entries"]
            return
        t0 = time.time()
        self.sizes = sizes or [remote_size(u) for u in parts]
        self.entries = self._read_cd()
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"sizes": self.sizes, "entries": self.entries}), encoding="utf-8")
        print(f"  zip central directory: {len(self.entries)} entries, {time.time() - t0:.0f}s ({parts[-1]})",
              file=sys.stderr, flush=True)

    def _abs(self, disk: int, off: int) -> int:
        return sum(self.sizes[:disk]) + off

    def _read(self, start: int, end: int) -> bytes:
        """Bytes [start, end) of the concatenated parts."""
        out, base = [], 0
        for url, size in zip(self.parts, self.sizes):
            lo, hi = max(start, base), min(end, base + size)
            if lo < hi:
                out.append(get_range(url, lo - base, hi - base))
            base += size
        return b"".join(out)

    def _read_cd(self) -> dict:
        lsize = self.sizes[-1]
        tail = get_range(self.parts[-1], lsize - min(lsize, 1 << 16), lsize)
        i = tail.rfind(b"PK\x05\x06")
        if i < 0:
            raise ValueError("no end-of-central-directory record")
        _, _, cd_disk, _, _, cd_size, cd_off, _ = struct.unpack("<IHHHHIIH", tail[i:i + 22])
        j = tail.rfind(b"PK\x06\x07", 0, i)  # zip64 locator -> zip64 end record
        if j >= 0:
            _, z_disk, z_off, _ = struct.unpack("<IIQI", tail[j:j + 20])
            z = self._read(self._abs(z_disk, z_off), self._abs(z_disk, z_off) + 56)
            _, _, _, _, _, cd_disk, _, _, cd_size, cd_off = struct.unpack("<IQHHIIQQQQ", z)
        start = self._abs(cd_disk, cd_off)
        cd = self._read(start, start + cd_size)
        entries, p = {}, 0
        while cd[p:p + 4] == b"PK\x01\x02":
            (_, _, _, flags, method, _, _, _, csize, usize, nlen, xlen, clen, disk, _, _, loff) = \
                struct.unpack("<IHHHHHHIIIHHHHHII", cd[p:p + 46])
            name = cd[p + 46:p + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
            extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
            q = 0
            while q + 4 <= len(extra):  # zip64 extra: only the overflowed fields are present, in this order
                hid, hlen = struct.unpack("<HH", extra[q:q + 4])
                if hid == 1:
                    r = q + 4
                    if usize == 0xFFFFFFFF:
                        usize, r = struct.unpack("<Q", extra[r:r + 8])[0], r + 8
                    if csize == 0xFFFFFFFF:
                        csize, r = struct.unpack("<Q", extra[r:r + 8])[0], r + 8
                    if loff == 0xFFFFFFFF:
                        loff, r = struct.unpack("<Q", extra[r:r + 8])[0], r + 8
                    if disk == 0xFFFF:
                        disk = struct.unpack("<I", extra[r:r + 4])[0]
                q += 4 + hlen
            if not name.endswith("/"):
                entries[name] = [method, csize, usize, disk, loff]
            p += 46 + nlen + xlen + clen
        return entries

    def names(self) -> list[str]:
        return list(self.entries)

    def read(self, name: str) -> bytes:
        method, csize, _, disk, loff = self.entries[name]
        start = self._abs(disk, loff)
        nlen, xlen = struct.unpack("<HH", self._read(start + 26, start + 30))
        data = self._read(start + 30 + nlen + xlen, start + 30 + nlen + xlen + csize)
        if method == 0:
            return data
        if method == 8:
            return zlib.decompress(data, -15)
        raise ValueError(f"{name}: unsupported zip method {method}")
