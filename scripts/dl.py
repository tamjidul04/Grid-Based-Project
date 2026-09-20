"""Robust one-shot model downloader — bypass hf-xet, use plain HTTPS.

hf-xet's chunked-transfer protocol is fragile over slow/intermittent
networks and tends to corrupt mid-download with "I/O error: error
decoding response body". This script downloads each file via a plain
HTTPS GET with HTTP Range resume — same approach as
scripts/download_base_model.py but exposed as a single `python dl.py`
command.

Resumes from any partial file already on disk (per-file size threshold).

Usage:
    python scripts/dl.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Files to fetch. (filename, min_bytes_to_accept_as_complete).
FILES = [
    ("config.json",          100),
    ("generation_config.json", 100),
    ("merges.txt",           100),
    ("tokenizer_config.json", 100),
    ("tokenizer.json",       100),
    ("vocab.json",           100),
    ("model.safetensors",    1_000_000_000),  # ~3.1 GB; reject <1 GB as truncated
]

# Each file is downloaded from this URL on HF's CDN.
HF_BASE = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"


def expected_sha256() -> dict[str, str]:
    """Optional integrity check. Fetch from HF's API at startup."""
    try:
        r = requests.get(
            "https://huggingface.co/api/models/Qwen/Qwen2.5-1.5B-Instruct",
            timeout=30,
        )
        r.raise_for_status()
        info = r.json()
        siblings = info.get("siblings", [])
        return {
            s["rfilename"]: s.get("lfs", {}).get("oid", "")
            for s in siblings
            if "rfilename" in s
        }
    except Exception as e:
        print(f"  [warn] could not fetch sha256 map: {e}", file=sys.stderr)
        return {}


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


def download_file(
    session: requests.Session,
    url: str,
    out_path: Path,
    *,
    min_bytes: int,
    max_attempts: int = 50,
) -> None:
    """Download with HTTP Range resume + auto-retry.

    Treats any stream-side error (ChunkedEncodingError, ConnectionError,
    Timeout) as recoverable: re-opens the stream at the byte after the
    last successfully written byte, appends more data.
    """
    existing = out_path.stat().st_size if out_path.exists() else 0
    if existing >= min_bytes:
        print(f"  [skip] {out_path.name} ({existing:,} bytes >= {min_bytes:,})")
        return
    if existing > 0:
        print(f"  [resume] {out_path.name}: {existing:,} bytes on disk "
              f"(need {min_bytes:,})")
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
            elapsed = time.time() - t0
            print(f"         [retry {attempt}] connect/headers error "
                  f"after {total_written/(1024*1024):.1f} MB "
                  f"({elapsed:.0f}s): {type(e).__name__}: {e}")
            time.sleep(min(2 ** min(attempt, 6), 30))
            continue

        mode = "ab" if resume_from > 0 and r.status_code == 206 else "wb"
        if r.status_code == 206:
            print(f"         server confirmed resume at byte {resume_from:,} "
                  f"(attempt {attempt})")
        elif resume_from > 0 and r.status_code == 200:
            # Server ignored our Range; throw away partial, start fresh.
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
                for chunk in r.iter_content(chunk_size=1 << 16):  # 64 KB
                    if not chunk:
                        continue
                    f.write(chunk)
                    chunk_written += len(chunk)
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            broken = True
            print(f"         [retry {attempt}] stream broken after "
                  f"+{chunk_written/(1024*1024):.1f} MB: {type(e).__name__}")
        finally:
            r.close()

        total_written += chunk_written
        resume_from = total_written
        if broken:
            time.sleep(min(2 ** min(attempt, 6), 30))
            continue

        # Clean EOF.
        break
    else:
        raise RuntimeError(
            f"{out_path.name}: gave up after {max_attempts} attempts "
            f"at {total_written:,} bytes"
        )

    elapsed = time.time() - t0
    speed = (total_written - existing) / elapsed / (1024 * 1024) if elapsed > 0 else 0
    final_size = out_path.stat().st_size
    print(f"         OK in {elapsed:.0f}s ({speed:.2f} MB/s), "
          f"total {final_size:,} bytes after {attempt} attempt(s)")


def main() -> int:
    here = Path(__file__).resolve().parent.parent  # gridwise-llm/
    out_dir = here / "models" / "Qwen2.5-1.5B-Instruct"
    out_dir.mkdir(parents=True, exist_ok=True)

    sha_map = expected_sha256()

    print(f"Downloading Qwen/Qwen2.5-1.5B-Instruct to {out_dir}")
    print(f"  plain HTTPS + Range resume (bypasses hf-xet)")
    print()

    session = make_session()
    for fn, min_bytes in FILES:
        url = f"{HF_BASE}/{quote(fn)}"
        try:
            download_file(session, url, out_dir / fn, min_bytes=min_bytes)
        except KeyboardInterrupt:
            print("\nInterrupted by user — partial files kept on disk.")
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
