"""FastAPI server: POST /optimize-energy.

Phase A behavior:
    1. Validate the input against InputSchema.
    2. Parse operator notes into directives.
       - If GRIDWISE_USE_LLM=1, use the fine-tuned Qwen LLM.
       - Otherwise (default), use the rule-based fallback.
       - If the LLM fails, transparently fall back to the rules.
    3. Solve the 24-hour schedule with the PuLP optimizer.
    4. Validate the output against all official constraints.
    5. Return the OutputSchema.

The endpoint matches the public sample-cases spec exactly:
    POST /optimize-energy
    Body:  InputSchema
    Reply: OutputSchema
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from llm import InputSchema, OutputSchema, llm_enabled, parse_directives_safely
from optimizer.model import InfeasibleScheduleError, optimize_schedule
from optimizer.validate import validate_output

logger = logging.getLogger("gridwise")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GridWise LLM — Optimize Energy",
    version="0.2.0",
    description="BUP CSE FEST 2026 — Track 02, Problem P-08.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "GridWise LLM",
        "phase": "B — LLM directive parser + LP optimizer (rules fallback)",
        "llm_enabled": str(llm_enabled()),
        "endpoint": "POST /optimize-energy",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OutputSchema)
def optimize_energy(payload: InputSchema) -> OutputSchema:
    """Take an InputSchema, return a constraint-satisfying 24-hour schedule."""
    try:
        directives = parse_directives_safely(payload)
        output = optimize_schedule(payload, directives)
    except InfeasibleScheduleError as e:
        logger.warning("Infeasible schedule for %s: %s", payload.scenario_id, e)
        raise HTTPException(status_code=422, detail=f"Infeasible schedule: {e}") from e
    except Exception as e:
        logger.exception("Unexpected error on %s", payload.scenario_id)
        raise HTTPException(status_code=500, detail=str(e)) from e

    # Validate before responding — refuse to return invalid plans.
    ok, checks = validate_output(payload, output)
    if not ok:
        failed = [name for passed, name, _ in checks if not passed]
        logger.error("Validator failed on %s: %s", payload.scenario_id, failed)
        raise HTTPException(
            status_code=500,
            detail=f"Output failed validation: {failed}",
        )

    return output
