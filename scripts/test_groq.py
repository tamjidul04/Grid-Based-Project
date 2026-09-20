"""Test Groq on all 10 sample cases. Loads .env automatically."""
import os
import sys
import time
import json
import traceback
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

os.environ["GRIDWISE_USE_GROQ"] = "1"

from llm.schema import InputSchema
from llm.groq import parse_directives_with_groq, clear_cache, _GROQ_CACHE
_GROQ_CACHE.clear()
import llm.groq as groq_mod
# Monkey-patch to capture raw responses
_orig_call = groq_mod._call_groq
RAW_RESPONSES = {}
def _capturing_call(messages):
    key = str(len(messages))
    raw = _orig_call(messages)
    RAW_RESPONSES.setdefault(key, []).append(raw)
    return raw
groq_mod._call_groq = _capturing_call

data = json.load(open("data/raw/sample_cases.json", encoding="utf-8"))
n_ok = 0
total_t = 0
print(f"{'CASE':<12}{'LATENCY':>10}  EXPECTED -> GOT")
print("-" * 80)
for case in data["cases"]:
    inp = InputSchema.model_validate(case["input"])
    expected = [d["directive_type"] for d in case.get("expected_output", {}).get("directive_interpretation", [])]
    t0 = time.time()
    try:
        directives = parse_directives_with_groq(inp)
        dt = (time.time() - t0) * 1000
        total_t += dt
        got = [d.directive_type for d in directives]
        ok = got == expected
        n_ok += int(ok)
        status = "OK" if ok else "MISMATCH"
        print(f"{case['id']:<12}{dt:>8.0f}ms  {expected} -> {got}  {status}")
        if not ok:
            for d in directives:
                print(f"             -> {d.directive_type}: {d.structured_adjustment}")
    except Exception as e:
        print(f"{case['id']:<12}ERROR: {type(e).__name__}: {e}")
        # Show the raw text for debugging
        if hasattr(e, '__cause__') and '--- raw ---' in str(e):
            raw_section = str(e).split('--- raw ---')[1].split('--- end ---')[0].strip()
            print(f"             RAW: {raw_section[:400]}")

print("-" * 80)
print(f"Accuracy: {n_ok}/{len(data['cases'])} directive-type matches")
print(f"Total: {total_t:.0f}ms  (avg {total_t/len(data['cases']):.0f}ms/case)")
