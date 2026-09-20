"""Test Gemini 3.5 Flash on all 10 sample cases."""
import os, sys, time, json, traceback
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

os.environ["GRIDWISE_USE_GEMINI"] = "1"

from llm.schema import InputSchema
from llm.gemini import parse_directives_with_gemini, clear_cache

clear_cache()

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
        directives = parse_directives_with_gemini(inp)
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
        msg = str(e)
        if "--- raw ---" in msg:
            section = msg.split("--- raw ---")[1].split("--- end ---")[0].strip()
            print(f"{case['id']:<12}ERROR raw: {section[:300]}")
        else:
            print(f"{case['id']:<12}ERROR: {type(e).__name__}: {str(e)[:200]}")

print("-" * 80)
print(f"Accuracy: {n_ok}/{len(data['cases'])} directive-type matches")
print(f"Total: {total_t:.0f}ms  (avg {total_t/len(data['cases']):.0f}ms/case)")
