"""LLM output parsing: extract + validate directive_interpretation JSON.

The LLM is asked to return ONLY a JSON array, but in practice it sometimes:
    - wraps the array in ```json ... ``` fences
    - prepends a sentence like "Sure! Here's..."
    - returns invalid JSON if generation was truncated
    - emits a JSON object instead of an array

This module extracts the JSON robustly, parses it, validates the schema,
and returns a list of DirectiveInterpretation objects. On any failure it
raises ParseError so the caller can fall back to the rule-based parser.
"""

from __future__ import annotations

import json
import re
from typing import List

from pydantic import ValidationError

from llm.schema import (
    DirectiveInterpretation,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    SolarReductionAdjustment,
)


class ParseError(Exception):
    """Raised when LLM output cannot be parsed into valid directives."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    """Pull the first JSON array out of `text`, stripping code fences / preamble."""
    s = text.strip()

    # Strip ```json ... ``` fences if present.
    fence = _JSON_FENCE_RE.search(s)
    if fence:
        s = fence.group(1).strip()

    # Find the first '[' and the matching last ']'.
    if "[" in s and "]" in s:
        start = s.index("[")
        end = s.rindex("]") + 1
        s = s[start:end]

    return s.strip()


def _coerce_adjustment(directive_type: str, raw: dict | None) -> object | None:
    """Pydantic Union parsing can map 'hours'-only fields to any subclass.
    We coerce explicitly here so the directive_type and adjustment class
    stay consistent.
    """
    if raw is None:
        return None
    hours = raw.get("hours")
    if directive_type == "solar_reduction":
        return SolarReductionAdjustment(
            hours=hours,
            factor=float(raw.get("factor", 0.0)),
        )
    if directive_type == "minimum_battery_reserve":
        return MinimumBatteryReserveAdjustment(
            hours=hours,
            minimum_energy_kwh=float(raw.get("minimum_energy_kwh", 0.0)),
        )
    if directive_type == "no_charge_window":
        return NoChargeWindowAdjustment(hours=hours)
    if directive_type == "no_discharge_window":
        return NoDischargeWindowAdjustment(hours=hours)
    if directive_type == "max_grid_window":
        return MaxGridWindowAdjustment(
            hours=hours,
            max_grid_kwh=float(raw.get("max_grid_kwh", 0.0)),
        )
    return None


def parse_directive_json(raw_text: str, expected_n: int) -> List[DirectiveInterpretation]:
    """Parse LLM output text into a list of DirectiveInterpretation objects.

    Args:
        raw_text:    The raw LLM generation.
        expected_n:  Number of operator notes (length of the input list).
                     We use this to ensure the LLM returned exactly one
                     directive per note.

    Returns:
        A list of length `expected_n` of validated DirectiveInterpretation.

    Raises:
        ParseError: if the output cannot be parsed or fails validation.
    """
    json_text = _extract_json(raw_text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ParseError(f"invalid JSON: {e}\n--- raw ---\n{raw_text}\n--- end ---") from e

    if not isinstance(data, list):
        # Sometimes the LLM emits {"directive_interpretation": [...]} by mistake.
        if isinstance(data, dict) and "directive_interpretation" in data:
            data = data["directive_interpretation"]
        else:
            raise ParseError(f"expected a JSON array, got {type(data).__name__}")

    if len(data) != expected_n:
        raise ParseError(f"expected {expected_n} directives, got {len(data)}")

    out: list[DirectiveInterpretation] = []
    for i, item in enumerate(data):
        try:
            dtype = item["directive_type"]
            adj_raw = item.get("structured_adjustment")
            adj = _coerce_adjustment(dtype, adj_raw)
            out.append(
                DirectiveInterpretation(
                    note_index=int(item.get("note_index", i)),
                    applies=bool(item.get("applies", dtype != "no_op")),
                    directive_type=dtype,
                    structured_adjustment=adj,
                    explanation=str(item.get("explanation", "")),
                )
            )
        except (KeyError, ValidationError, TypeError, ValueError) as e:
            raise ParseError(f"directive {i} invalid: {e}\n--- item ---\n{item}") from e

    return out
