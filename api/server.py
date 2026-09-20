"""FastAPI server: POST /optimize-energy.

Production directive parser: rules_fallback (deterministic, 10/10 accuracy
on public sample cases, ~10 ms total).

LLM directive parser (Qwen2.5-0.5B/1.5B) is in the repo as a documented
experiment. It is opt-in via GRIDWISE_USE_LLM=1 but is NOT the default
because:
    - Qwen2.5-0.5B-Instruct misclassifies boundary cases (e.g. panels
      being washed -> solar_reduction) and tends to loop mid-generation.
    - Qwen2.5-1.5B-Instruct is correct but ~0.01 tok/s on CPU (hours
      per case). LoRA fine-tuning is not viable on CPU hardware.

Behavior:
    1. Validate the input against InputSchema.
    2. Parse operator notes into directives via parse_directives_safely:
       - If GRIDWISE_USE_LLM=1, try the LLM first.
       - Otherwise (default), use the rule-based fallback.
       - If the LLM fails, transparently fall back to the rules.
    3. Solve the 24-hour schedule with the PuLP optimizer.
    4. Validate the output against all official constraints.
    5. Return the OutputSchema.

Demo cache (Phase C):
    If GRIDWISE_USE_CACHE=1 and data/demo_cache.json exists with an entry
    whose input_hash matches the incoming payload, we serve the cached
    response instantly. The cache is keyed on (operator_notes + 24h
    data + battery params) so editing notes still falls through to the
    real pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from llm import InputSchema, OutputSchema, llm_enabled, parse_directives_safely
from optimizer.model import InfeasibleScheduleError, optimize_schedule
from optimizer.validate import validate_output

logger = logging.getLogger("gridwise")
logging.basicConfig(level=logging.INFO)

# Cache the sample cases in memory at startup so /sample-cases is fast.
_SAMPLE_CASES: list[dict] = []
# Optional pre-computed demo cache (used when GRIDWISE_USE_CACHE=1).
_DEMO_CACHE: dict = {}
_CACHE_PATH: Path | None = None


def _load_sample_cases() -> list[dict]:
    here = Path(__file__).resolve().parent.parent
    path = here / "data" / "raw" / "sample_cases.json"
    if not path.exists():
        logger.warning("sample_cases.json not found at %s", path)
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("cases", [])


def _load_demo_cache() -> dict:
    global _CACHE_PATH
    here = Path(__file__).resolve().parent.parent
    path = here / "data" / "demo_cache.json"
    _CACHE_PATH = path
    if not path.exists():
        logger.info("No demo cache at %s (run scripts.cache_demo_results)", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded demo cache: %d entries from %s", len(data), path)
        return data
    except Exception as e:
        logger.warning("Failed to load demo cache: %s", e)
        return {}


def _hash_payload(payload: InputSchema) -> str:
    """SHA256 of the optimization-relevant fields, first 16 hex chars."""
    blob = {
        "operator_notes": payload.operator_notes,
        "hours": [
            {"hour": h.hour, "demand_kwh": h.demand_kwh,
             "solar_kwh": h.solar_kwh, "tariff_bdt_per_kwh": h.tariff_bdt_per_kwh}
            for h in payload.hours
        ],
        "battery": payload.battery.model_dump(),
    }
    return hashlib.sha256(
        json.dumps(blob, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def cache_enabled() -> bool:
    return os.environ.get("GRIDWISE_USE_CACHE", "0") == "1"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _SAMPLE_CASES, _DEMO_CACHE
    _SAMPLE_CASES = _load_sample_cases()
    logger.info("Loaded %d sample cases", len(_SAMPLE_CASES))
    # Always preload the cache file if it exists on disk — env var
    # just gates whether we SERVE from it.
    on_disk = _load_demo_cache()
    if on_disk:
        _DEMO_CACHE = on_disk
        logger.info("Loaded %d demo cache entries (cache_enabled=%s)",
                    len(_DEMO_CACHE), cache_enabled())
    else:
        _DEMO_CACHE = {}
    yield


app = FastAPI(
    title="GridWise — Optimize Energy",
    version="0.3.0",
    description="Microgrid schedule optimizer (LP) with deterministic directive parser.",
    lifespan=_lifespan,
)

# Permissive CORS so the static frontend (served from a different
# origin / port during local demo) can call us without hassle.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "GridWise LLM",
        "phase": "B — LLM directive parser + LP optimizer (rules fallback)",
        "llm_enabled": str(llm_enabled()),
        "cache_enabled": str(cache_enabled()),
        "cache_entries": str(len(_DEMO_CACHE)),
        "endpoint": "POST /optimize-energy",
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sample-cases")
def sample_cases() -> dict:
    """List the public sample cases (id + input) so the frontend can populate."""
    return {"cases": _SAMPLE_CASES}


@app.post("/optimize-energy", response_model=OutputSchema)
def optimize_energy(payload: InputSchema) -> OutputSchema:
    """Take an InputSchema, return a constraint-satisfying 24-hour schedule."""
    # Demo cache: serve instantly if input_hash matches a pre-computed case.
    if cache_enabled() and _DEMO_CACHE:
        h = _hash_payload(payload)
        for case_id, entry in _DEMO_CACHE.items():
            if entry.get("input_hash") == h and entry.get("valid"):
                logger.info("[cache HIT] %s (hash=%s)", case_id, h)
                return OutputSchema.model_validate(entry["response"])
        logger.info("[cache miss] hash=%s — running live pipeline", h)

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
