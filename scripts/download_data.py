"""Download only the labeled-disease subset of DENTEX from Hugging Face.

The full training zip is ~11 GB (mostly unlabeled x-rays). We read the zip
over HTTP with Range requests and extract only the entries we need.
"""
import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests
from huggingface_hub import hf_hub_url

REPO = "ibrahimhamamci/DENTEX"
CHUNK = 16 * 1024 * 1024  # read-ahead size per HTTP request


class RemoteFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a read-ahead cache."""

    def __init__(self, url: str):
        self.url = url
        self.sess = requests.Session()
        r = self.sess.head(url, allow_redirects=True)
        r.raise_for_status()
        self.url = r.url  # follow redirect to the CDN once
        self.size = int(r.headers["Content-Length"])
        self.pos = 0
        self.buf_start = 0
        self.buf = b""
        self.requests = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = off
        elif whence == io.SEEK_CUR:
            self.pos += off
        elif whence == io.SEEK_END:
            self.pos = self.size + off
        return self.pos

    def _fetch(self, start, length):
        end = min(start + length, self.size) - 1
        r = self.sess.get(self.url, headers={"Range": f"bytes={start}-{end}"})
        r.raise_for_status()
        self.requests += 1
        self.buf_start = start
        self.buf = r.content

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        inside = self.buf_start <= self.pos < self.buf_start + len(self.buf)
        if not inside:
            self._fetch(self.pos, max(n, CHUNK))
        off = self.pos - self.buf_start
        out = self.buf[off:off + n]
        self.pos += len(out)
        if len(out) < n:  # spans beyond cache; recurse for the rest
            out += self.read(n - len(out))
        return out


def extract(zip_name: str, prefix: str, out_dir: Path, list_only: bool):
    url = hf_hub_url(REPO, f"DENTEX/{zip_name}", repo_type="dataset")
    rf = RemoteFile(url)
    print(f"{zip_name}: {rf.size / 1e9:.2f} GB remote")
    zf = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=CHUNK))
    members = [m for m in zf.infolist() if m.filename.startswith(prefix)
               and not m.is_dir() and ".ipynb_checkpoints" not in m.filename]
    total = sum(m.file_size for m in members)
    print(f"{len(members)} files, {total / 1e9:.2f} GB under {prefix}")
    if list_only:
        dirs = sorted({m.filename.split("/")[1] for m in zf.infolist() if "/" in m.filename})
        print("top-level dirs:", dirs)
        return
    members.sort(key=lambda m: m.header_offset)  # sequential reads = fewer requests
    done = 0
    for i, m in enumerate(members, 1):
        dest = out_dir / m.filename
        if dest.exists() and dest.stat().st_size == m.file_size:
            done += m.file_size
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        done += m.file_size
        if i % 25 == 0 or i == len(members):
            print(f"  {i}/{len(members)}  {done / 1e9:.2f}/{total / 1e9:.2f} GB  "
                  f"({rf.requests} http requests)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--list", action="store_true", help="only list, do not download")
    args = ap.parse_args()
    out = Path(args.out)
    extract("training_data.zip", "training_data/quadrant-enumeration-disease/", out, args.list)
