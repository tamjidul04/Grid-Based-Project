"""Smoke test: run SAMPLE-01 end-to-end and print the full OutputSchema.

Usage:
    python -m scripts.run_demo
"""

from __future__ import annotations

import json
from pathlib import Path

from llm.schema import InputSchema, OutputSchema
from optimizer.model import optimize_schedule
from optimizer.rules_fallback import parse_notes
from optimizer.validate import validate_output


def main() -> None:
    here = Path(__file__).resolve().parent.parent
    cases_path = here / "data" / "raw" / "sample_cases.json"
    with cases_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)["cases"]
    sample01 = next(c for c in cases if c["id"] == "SAMPLE-01")

    inp = InputSchema.model_validate(sample01["input"])
    expected = OutputSchema.model_validate(sample01["expected_output"])

    directives = parse_notes(inp)
    output = optimize_schedule(inp, directives)

    ok, checks = validate_output(inp, output)
    print("=" * 70)
    print(f"SAMPLE-01 end-to-end")
    print(f"  directives parsed   : {len(directives)}")
    for d in directives:
        print(f"    [{d.note_index}] {d.directive_type:<25} applies={d.applies}")
    print(f"  validation          : {'OK' if ok else 'FAIL'}")
    for passed, name, detail in checks:
        flag = "OK" if passed else "FAIL"
        print(f"    [{flag}] {name:<35} {detail}")
    print(f"  total cost          : {output.total_cost_bdt:.2f} BDT "
          f"(reference: {expected.total_cost_bdt:.2f})")
    print(f"  peak grid           : {output.peak_grid_kwh:.2f} kWh")
    print(f"  total grid          : {output.total_grid_kwh:.2f} kWh")
    print(f"  end-of-day battery  : {output.hourly_plan[-1].battery_energy_after_kwh:.2f} kWh "
          f"(initial: {inp.battery.initial_energy_kwh:.2f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
