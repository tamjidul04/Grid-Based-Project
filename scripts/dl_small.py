"""Download Qwen2.5-0.5B-Instruct (smaller, CPU-faster alternative to 1.5B).

Same robust plain-HTTPS approach as dl.py but only the files needed.
We deliberately skip the safetensors index file (model.safetensors.index.json)
since 0.5B is a single-file model.

Usage:
    python scripts/dl_small.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


FILES = [
    ("config.json",            100),
    ("generation_config.json", 100),
    ("merges.txt",             100),
    ("tokenizer_config.json",  100),
    ("tokenizer.json",         100),
    ("vocab.json",             100),
    ("model.safetensors",      500_000_000),  # 0.5B fp16 ≈ 1 GB; reject <500MB
]

HF_BASE = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=10,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def download_file(session, url, out_path, *, min_bytes, max_attempts=50):
    existing = out_path.stat().st_size if out_path.exists() else 0
    if existing >= min_bytes:
        print(f"  [skip] {out_path.name} ({existing:,} bytes >= {min_bytes:,})")
        return
    if existing > 0:
        print(f"  [resume] {out_path.name}: {existing:,} bytes on disk")
    else:
        print(f"  [get]   {out_path.name}")

    resume_from = existing
    total_written = existing
    t0 = time.time()
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        headers = {}
        if resume_from > 1_000_000:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            r = session.get(url, allow_redirects=True, stream=True,
                            headers=headers, timeout=(20, 90))
            r.raise_for_status()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError) as e:
            print(f"         [retry {attempt}] connect/headers error: {type(e).__name__}: {e}")
            time.sleep(min(2 ** min(attempt, 6), 30))
            continue

        mode = "ab" if resume_from > 0 and r.status_code == 206 else "wb"
        if resume_from > 0 and r.status_code == 200:
            print(f"         server ignored Range; resetting local file")
            out_path.unlink(missing_ok=True)
            resume_from = 0
            total_written = 0
            mode = "wb"
            r.close()
            continue

        chunk_written = 0
        broken = False
        try:
            with out_path.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if not chunk:
                        continue
                    f.write(chunk)
                    chunk_written += len(chunk)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            broken = True
            print(f"         [retry {attempt}] stream broken after +{chunk_written/(1024*1024):.1f} MB: {type(e).__name__}")
        finally:
            r.close()

        total_written += chunk_written
        resume_from = total_written
        if broken:
            time.sleep(min(2 ** min(attempt, 6), 30))
            continue
        break
    else:
        raise RuntimeError(f"{out_path.name}: gave up after {max_attempts} attempts")

    elapsed = time.time() - t0
    final_size = out_path.stat().st_size
    print(f"         OK in {elapsed:.0f}s, total {final_size:,} bytes after {attempt} attempt(s)")


def main() -> int:
    here = Path(__file__).resolve().parent.parent
    out_dir = here / "models" / "Qwen2.5-0.5B-Instruct"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Qwen2.5-0.5B-Instruct to {out_dir}")
    print()

    session = make_session()
    for fn, min_bytes in FILES:
        url = f"{HF_BASE}/{quote(fn)}"
        try:
            download_file(session, url, out_dir / fn, min_bytes=min_bytes)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 1
        except Exception as e:
            print(f"  [ERROR] {fn}: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    print()
    print("Final state:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            print(f"  {p.name}: {p.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
