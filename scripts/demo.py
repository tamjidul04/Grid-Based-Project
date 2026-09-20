"""Live demo script — runs the 10 sample cases and prints a clean table.

Designed for the hackathon stage presentation: one command, instant output,
proves all 10 cases pass.

Usage:
    python scripts/demo.py
    python scripts/demo.py --use-llm   (Phase B with fine-tuned LLM)
    python scripts/demo.py --cache      (Phase C — instant cached responses)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Live demo — run all sample cases.")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use the fine-tuned LLM (Phase B).")
    parser.add_argument("--cache", action="store_true",
                        help="Use the pre-computed demo cache (instant).")
    parser.add_argument("--serve", action="store_true",
                        help="Also start the API server (foreground).")
    args = parser.parse_args()

    print("=" * 78)
    print("GridWise LLM — BUP CSE FEST 2026 (Track 02, P-08)")
    print("=" * 78)
    print()
    print(f"  Base model:        Qwen2.5-1.5B-Instruct (Apache 2.0)")
    print(f"  Parser:            {'Fine-tuned LLM' if args.use_llm else 'Rules fallback'}")
    print(f"  Cache:             {'ENABLED (instant)' if args.cache else 'disabled'}")
    print(f"  Optimizer:         PuLP LP (CBC solver)")
    print(f"  Validator:         11+ constraint checks")
    print()

    cases_path = HERE / "data" / "raw" / "sample_cases.json"
    with cases_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    print(f"Loaded {len(cases)} public sample cases from {cases_path.name}")
    print()
    print(f"{'CASE':<12}{'COST (BDT)':>14}{'REF':>14}{'DELTA':>10}  STATUS  LABEL")
    print("-" * 78)

    # Decide which evaluator to call.
    if args.cache:
        cmd = [sys.executable, "-c", _CACHE_RUNNER]
    elif args.use_llm:
        cmd = [sys.executable, "-m", "scripts.evaluate", "--use-llm"]
    else:
        cmd = [sys.executable, "-m", "scripts.evaluate"]

    if args.cache:
        env = os.environ.copy()
        env["GRIDWISE_USE_CACHE"] = "1"
        proc = subprocess.run(cmd, env=env, cwd=str(HERE),
                              capture_output=True, text=True)
    else:
        proc = subprocess.run(cmd, cwd=str(HERE),
                              capture_output=True, text=True)

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    print(proc.stdout)
    return 0


# Inlined runner that uses the cache for instant demo output.
_CACHE_RUNNER = """
import json, os
from pathlib import Path
from llm import InputSchema, OutputSchema

HERE = Path('.').resolve()
with (HERE / 'data' / 'raw' / 'sample_cases.json').open() as f:
    cases = json.load(f)['cases']
with (HERE / 'data' / 'demo_cache.json').open() as f:
    cache = json.load(f)

print(f"{'CASE':<12}{'COST (BDT)':>14}{'REF':>14}{'DELTA':>10}  STATUS  LABEL")
print('-' * 78)
n_ok = 0
for c in cases:
    inp = InputSchema.model_validate(c['input'])
    entry = cache.get(c['id'])
    if not entry or not entry.get('valid'):
        print(f"{c['id']:<12}{'CACHE MISS':>14}")
        continue
    out = OutputSchema.model_validate(entry['response'])
    delta = out.total_cost_bdt - c['expected_output']['total_cost_bdt']
    status = 'OK' if abs(delta) <= 0.01 else 'DELTA'
    if abs(delta) <= 0.01:
        n_ok += 1
    print(f"{c['id']:<12}{out.total_cost_bdt:>14.2f}"
          f"{c['expected_output']['total_cost_bdt']:>14.2f}"
          f"{delta:>+10.2f}  {status:<7} {c.get('label', '')}")
print('-' * 78)
print(f'Summary: {n_ok}/{len(cases)} cases matched reference cost exactly.')
"""


if __name__ == "__main__":
    sys.exit(main())
