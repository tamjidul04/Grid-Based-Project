"""Speed + accuracy test for the prompt-engineered LLM.

Loads the base Qwen2.5-1.5B-Instruct, runs all 10 public sample cases
through the LLM directive parser, and prints:
  - per-case latency
  - parsed-directive types vs. reference types (where we have them)
  - summary accuracy and total wall time

Usage:
    python -m scripts.test_llm_speed
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("llm_speed")

HERE = Path(__file__).resolve().parent.parent


def main() -> int:
    # Force LLM mode on for this script and pin the 0.5B model.
    # (3x faster on CPU than 1.5B; both produce valid JSON.)
    import os
    os.environ["GRIDWISE_USE_LLM"] = "1"
    os.environ["GRIDWISE_LLM_MODEL"] = "models/Qwen2.5-0.5B-Instruct"

    from llm import InputSchema, parse_directives_with_llm
    from llm.inference import _load_model, clear_cache

    cases_path = HERE / "data" / "raw" / "sample_cases.json"
    with cases_path.open(encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    print(f"Loaded {len(cases)} cases. Warming up model...")
    t0 = time.time()
    _load_model()  # force singleton load so warm-up happens before timer
    print(f"  warm-up: {time.time() - t0:.1f}s")
    clear_cache()

    print()
    print(f"{'CASE':<12}{'LATENCY(s)':>12}{'TYPES':<35}  STATUS")
    print("-" * 80)

    n_ok = 0
    total_t = 0.0
    for c in cases:
        inp = InputSchema.model_validate(c["input"])
        expected = c.get("expected_directives") or c.get("expected_output", {}).get("directive_interpretation", [])
        t0 = time.time()
        try:
            directives = parse_directives_with_llm(inp)
        except Exception as e:
            elapsed = time.time() - t0
            total_t += elapsed
            print(f"{c['id']:<12}{elapsed:>12.1f}{'(parse failed)':<35}  FAIL: {e}")
            continue
        elapsed = time.time() - t0
        total_t += elapsed

        got_types = [d.directive_type for d in directives]
        exp_types = [d.get("directive_type") for d in expected] if expected else []
        if exp_types:
            status = "OK" if got_types == exp_types else "MISMATCH"
            if got_types == exp_types:
                n_ok += 1
        else:
            status = "(no reference)"

        print(f"{c['id']:<12}{elapsed:>12.1f}{','.join(got_types):<35}  {status}")
        if status == "MISMATCH":
            print(f"   got: {got_types}")
            print(f"   exp: {exp_types}")

    print("-" * 80)
    print(f"Summary: {n_ok}/{len(cases)} directive-type matches in {total_t:.1f}s total")
    print(f"Avg per-case: {total_t / len(cases):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
