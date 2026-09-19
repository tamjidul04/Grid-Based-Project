"""End-to-end tests for Phase A.

Each test loads one of the public sample cases, runs it through the full
pipeline (rules_fallback → optimizer → validator), and asserts:
    * output is constraint-satisfying
    * total cost is within $0.01 BDT of the reference
    * directive match for the cases we can handle with rules alone
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm.schema import InputSchema, OutputSchema
from optimizer.model import optimize_schedule
from optimizer.rules_fallback import parse_notes
from optimizer.validate import directive_matches, validate_output


def _load_cases() -> list[dict]:
    here = Path(__file__).resolve().parent.parent
    with (here / "data" / "raw" / "sample_cases.json").open("r", encoding="utf-8") as f:
        return json.load(f)["cases"]


ALL_CASES = _load_cases()
CASE_IDS = [c["id"] for c in ALL_CASES]


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_constraint_satisfaction(case_id: str) -> None:
    case = next(c for c in ALL_CASES if c["id"] == case_id)
    inp = InputSchema.model_validate(case["input"])
    expected = OutputSchema.model_validate(case["expected_output"])

    directives = parse_notes(inp)
    output = optimize_schedule(inp, directives)

    ok, checks = validate_output(inp, output)
    failed = [(name, detail) for passed, name, detail in checks if not passed]
    assert ok, f"{case_id} failed checks: {failed}"


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_cost_within_tolerance(case_id: str) -> None:
    """The LP solves for *minimum cost*. Reference may not be globally optimal,
    so we assert that our cost is <= reference + 0.01 BDT (within rounding)."""
    case = next(c for c in ALL_CASES if c["id"] == case_id)
    inp = InputSchema.model_validate(case["input"])
    expected = OutputSchema.model_validate(case["expected_output"])

    directives = parse_notes(inp)
    output = optimize_schedule(inp, directives)

    # LP is optimal; reference is at most one of multiple optima.
    # Allow a tiny slack for floating-point rounding (≤ 0.01 BDT per spec).
    assert output.total_cost_bdt <= expected.total_cost_bdt + 0.01, (
        f"{case_id}: cost {output.total_cost_bdt} > ref {expected.total_cost_bdt}"
    )


@pytest.mark.parametrize("case_id", ["SAMPLE-02", "SAMPLE-03", "SAMPLE-04", "SAMPLE-05"])
def test_rules_parser_matches_simple_cases(case_id: str) -> None:
    """The rule parser handles single-directive cases (rest have multiple notes
    or %-based reserves that need the LLM)."""
    case = next(c for c in ALL_CASES if c["id"] == case_id)
    inp = InputSchema.model_validate(case["input"])
    expected = OutputSchema.model_validate(case["expected_output"])

    directives = parse_notes(inp)
    output = optimize_schedule(inp, directives)

    dir_ok, msg = directive_matches(output, expected)
    assert dir_ok, f"{case_id}: {msg}"
