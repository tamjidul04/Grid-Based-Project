"""Pre-compute all 10 sample cases and cache the results.

For the live demo we don't want to wait for the LLM to generate each
response in real time. This script runs the full pipeline once and
saves the responses to disk so the API can serve them instantly.

The cache uses the same parsing path as the live API (rules fallback
by default; LLM when GRIDWISE_USE_LLM=1) so the cached output exactly
matches what live inference would produce for the same input.

Usage:
    python -m scripts.cache_demo_results            # rules fallback
    GRIDWISE_USE_LLM=1 python -m scripts.cache_demo_results  # LLM
    python -m scripts.cache_demo_results --force    # rebuild even if cache exists
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from llm import InputSchema, parse_directives_safely
from optimizer.model import InfeasibleScheduleError, optimize_schedule
from optimizer.validate import validate_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute demo cases.")
    parser.add_argument("--cases", type=str, default="data/raw/sample_cases.json",
                        help="Path to the sample-cases JSON.")
    parser.add_argument("--out", type=str, default="data/demo_cache.json",
                        help="Where to write the cache.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if the cache file already exists.")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent.parent
    cases_path = here / args.cases
    out_path = here / args.out

    if out_path.exists() and not args.force:
        print(f"Cache already exists at {out_path} — pass --force to rebuild.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    use_llm = os.environ.get("GRIDWISE_USE_LLM", "0") == "1"
    print(f"Caching {len(cases)} cases "
          f"(parser={'LLM' if use_llm else 'rules'})...")

    cache: dict[str, dict] = {}
    t_total = time.time()
    for c in cases:
        t0 = time.time()
        try:
            inp = InputSchema.model_validate(c["input"])
            directives = parse_directives_safely(inp)
            output = optimize_schedule(
                inp, directives,
                plan_summary=f"Pre-computed for {c['id']}.",
            )
            ok, _ = validate_output(inp, output)
            elapsed_ms = int((time.time() - t0) * 1000)
            cache[c["id"]] = {
                "input_hash": _hash_input(inp),
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_ms": elapsed_ms,
                "parser": "llm" if use_llm else "rules",
                "valid": ok,
                "response": output.model_dump(mode="json"),
            }
            status = "OK " if ok else "INV"
            print(f"  {status} {c['id']}: {output.total_cost_bdt:>10.2f} BDT  "
                  f"({elapsed_ms} ms)")
        except InfeasibleScheduleError as e:
            print(f"  ERR {c['id']}: Infeasible: {e}")
            cache[c["id"]] = {
                "input_hash": None,
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "parser": "llm" if use_llm else "rules",
                "valid": False,
                "error": f"Infeasible: {e}",
            }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    total_s = time.time() - t_total
    n_ok = sum(1 for v in cache.values() if v.get("valid"))
    print(f"\nWrote {len(cache)} entries ({n_ok} valid) to {out_path} "
          f"in {total_s:.1f}s")


def _hash_input(inp: InputSchema) -> str:
    """Stable hash of the input that affects optimization, for cache invalidation."""
    import hashlib
    payload = {
        "operator_notes": inp.operator_notes,
        "hours": [
            {"hour": h.hour, "demand_kwh": h.demand_kwh,
             "solar_kwh": h.solar_kwh, "tariff_bdt_per_kwh": h.tariff_bdt_per_kwh}
            for h in inp.hours
        ],
        "battery": inp.battery.model_dump(),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


if __name__ == "__main__":
    main()
