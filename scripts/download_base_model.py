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
# Tuple of (filename, expected_min_bytes) — used to detect a truncated
# partial file and trigger a resume rather than skipping it.
FILES = [
    ("config.json",          100),
    ("generation_config.json", 100),
    ("merges.txt",           100),
    ("tokenizer_config.json", 100),
    ("tokenizer.json",       100),
    ("vocab.json",           100),
    ("model.safetensors",    1_000_000_000),  # ~3.1 GB; reject anything <1 GB
    ("LICENSE",              100),
    ("README.md",            100),
]

REPO_URL = "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main"


def download_one(session: requests.Session, filename: str, out_dir: Path,
                  expected_size: int | None = None, *, force: bool = False,
                  max_resume_attempts: int = 50) -> None:
    """Download a single file with HTTP Range resume + auto-retry.

    The HF→AWS CDN frequently resets long-lived streaming connections
    (ConnectionResetError 10054 on Windows). We treat those as recoverable:
    on any transport error we re-open the stream with a Range request for
    the bytes we still need, then keep writing in append mode.
    """
    url = f"{REPO_URL}/{filename}"
    out_path = out_dir / filename

    # Decide whether to skip, resume, or start fresh.
    # We use a per-file min_size threshold — for the ~3 GB safetensors
    # anything <1 GB is obviously a truncated partial, NOT a valid copy.
    threshold = expected_size if expected_size is not None else 100

    existing = out_path.stat().st_size if out_path.exists() else 0
    if not force and existing >= threshold:
        print(f"  [skip] {filename} (already exists, {existing:,} bytes ≥ {threshold:,})")
        return

    if force and existing > 0:
        print(f"  [force] {filename}: discarding local {existing:,} bytes")
        out_path.unlink()
        existing = 0
    elif existing > 0:
        print(f"  [resume] {filename}: local {existing:,} bytes is below "
              f"expected minimum {threshold:,} — resuming")

    resume_from = existing
    total_written = existing
    start = time.time()

    print(f"  [get]  {filename} (resuming from byte {resume_from:,})")
    attempt = 0
    while attempt < max_resume_attempts:
        attempt += 1
        headers = {}
        if resume_from > 1_000_000:
            headers["Range"] = f"bytes={resume_from}-"

        # Use a fresh connection per attempt — persistent connections seem
        # to be the ones that get killed by the CDN.
        try:
            r = session.get(
                url,
                allow_redirects=True,
                stream=True,
                headers=headers,
                timeout=(15, 60),
            )
            r.raise_for_status()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout) as e:
            elapsed = time.time() - start
            print(f"         [retry {attempt}] transport error after "
                  f"{total_written/(1024*1024):.1f} MB "
                  f"({elapsed:.0f}s): {type(e).__name__}")
            time.sleep(min(2 ** min(attempt, 6), 30))
            continue

        mode = "ab" if resume_from > 0 and r.status_code == 206 else "wb"
        if r.status_code == 206:
            print(f"         server confirmed resume at byte {resume_from:,}")
        elif resume_from > 0 and r.status_code == 200:
            # Server ignored our Range and sent from the start — bail; we'll
            # trash the partial and start over next attempt.
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

        # Done (clean exit from the iter loop = EOF).
        break

    else:
        raise RuntimeError(f"{filename}: gave up after {max_resume_attempts} attempts "
                           f"at {total_written:,} bytes")

    elapsed = time.time() - start
    speed = (total_written - existing) / elapsed / (1024 * 1024) if elapsed > 0 else 0
    size_mb = (total_written - existing) / (1024 * 1024)
    final_size = out_path.stat().st_size
    print(f"         +{size_mb:.1f} MB in {elapsed:.1f}s ({speed:.2f} MB/s); "
          f"total {final_size:,} bytes ({attempt} attempt(s))")
    if final_size < 100:
        raise RuntimeError(f"{filename}: downloaded file is suspiciously small "
                           f"({final_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Qwen2.5-1.5B-Instruct.")
    parser.add_argument("--out", type=str, default="models/Qwen2.5-1.5B-Instruct",
                        help="Output directory.")
    parser.add_argument("--files", nargs="*", default=None,
                        help="Optional override of file list.")
    parser.add_argument("--force", action="store_true",
                        help="Redownload all files even if they exist locally.")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only download these filenames (e.g. model.safetensors).")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    files = args.files or [f for f, _ in FILES]
    if args.only:
        files = [f for f in files if f in set(args.only)]
    min_sizes = {fn: ms for fn, ms in FILES}

    session = requests.Session()
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)

    print(f"Downloading {len(files)} file(s) to {out_dir} (force={args.force})")
    for i, fn in enumerate(files, 1):
        try:
            download_one(session, fn, out_dir, expected_size=min_sizes.get(fn, 100),
                          force=args.force)
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
