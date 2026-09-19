"""Integration tests for the FastAPI server.

These hit the actual app via FastAPI's TestClient (in-process, no network).
By default they exercise the rules_fallback path (GRIDWISE_USE_LLM=0).
A separate env var flips the same tests to use the LLM (skipped if model
isn't trained yet).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the LLM is NOT loaded for the default tests (rules fallback).
os.environ.setdefault("GRIDWISE_USE_LLM", "0")

from api.server import app  # noqa: E402

client = TestClient(app)


def _load_cases() -> list[dict]:
    here = Path(__file__).resolve().parent.parent
    with (here / "data" / "raw" / "sample_cases.json").open("r", encoding="utf-8") as f:
        return json.load(f)["cases"]


ALL_CASES = _load_cases()


def test_root() -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["endpoint"] == "POST /optimize-energy"


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.parametrize("case_id", [c["id"] for c in ALL_CASES])
def test_optimize_energy_endpoint(case_id: str) -> None:
    """POST each sample case and verify the response is constraint-satisfying."""
    case = next(c for c in ALL_CASES if c["id"] == case_id)
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, f"{case_id}: {r.status_code} {r.text[:200]}"
    out = r.json()
    assert out["scenario_id"] == case["input"]["scenario_id"]
    assert len(out["hourly_plan"]) == 24
    assert len(out["directive_interpretation"]) == len(case["input"]["operator_notes"])
    # Constraint sanity (subset — full set in test_pipeline.py):
    assert out["total_grid_kwh"] >= 0
    assert out["total_cost_bdt"] >= 0
    assert out["peak_grid_kwh"] >= 0
    # End-of-day neutrality
    last = out["hourly_plan"][-1]
    init = case["input"]["battery"]["initial_energy_kwh"]
    assert abs(last["battery_energy_after_kwh"] - init) <= 0.02
