"""Robustly download Qwen2.5-1.5B-Instruct using requests (no HF library).

The huggingface_hub library has been stalling on this network. We bypass it
and stream the files directly with retry logic.

Usage: python -m scripts.download_base_model
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


# Files we need from the Qwen2.5-1.5B-Instruct repo.
# Verified against the actual repo contents via the HF API.
FILES = [
    "config.json",
    "generation_config.json",
    "merges.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "model.safetensors",
    "LICENSE",
    "README.md",
]

REPO_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"


def download_one(session: requests.Session, filename: str, out_dir: Path) -> None:
    url = f"{REPO_URL}/{filename}"
    out_path = out_dir / filename
    if out_path.exists() and out_path.stat().st_size > 100:
        print(f"  [skip] {filename} (already exists)")
        return

    print(f"  [get]  {filename}")
    start = time.time()
    r = session.get(url, allow_redirects=True, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length") or 0)
    written = 0
    with out_path.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 16):  # 64 KB
            if not chunk:
                continue
            f.write(chunk)
            written += len(chunk)
    elapsed = time.time() - start
    speed = written / elapsed / (1024 * 1024) if elapsed > 0 else 0
    size_mb = written / (1024 * 1024)
    print(f"         {size_mb:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-1.5B-Instruct.")
    parser.add_argument("--out", type=str, default="models/Qwen2.5-1.5B-Instruct",
                        help="Output directory.")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Optional override of file list.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    files = args.files or FILES
    session = requests.Session()
    # retry adapter
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    print(f"Downloading to {out_dir}")
    for i, fn in enumerate(files, 1):
        try:
            download_one(session, fn, out_dir)
        except Exception as e:
            print(f"  [ERROR] {fn}: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"  [{i}/{len(files)}] complete")
    print(f"\nDone. Files in {out_dir}:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
