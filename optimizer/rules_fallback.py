"""Rule-based directive extractor.

Used as a safety net when the LLM produces invalid output, and as the
default backend in Phase A (before the LLM is trained).

The key challenge: percentage-based reserve notes ("keep at least 50% of
the battery capacity") require knowing the battery capacity to convert
to kWh. So the parse_notes() entry point accepts the InputSchema and
threads capacity into the per-note classifier.
"""

from __future__ import annotations

import re
from typing import List

from llm.schema import (
    DirectiveInterpretation,
    InputSchema,
    MaxGridWindowAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    SolarReductionAdjustment,
)

# Map hour-int (0-23) to a human phrase used in the notes. Stored in uppercase
# so we can normalize input by .upper() before lookup.
HOUR_PHRASES = {
    "MIDNIGHT": 0, "NOON": 12,
    "12 AM": 0, "1 AM": 1, "2 AM": 2, "3 AM": 3, "4 AM": 4, "5 AM": 5,
    "6 AM": 6, "7 AM": 7, "8 AM": 8, "9 AM": 9, "10 AM": 10, "11 AM": 11,
    "12 PM": 12, "1 PM": 13, "2 PM": 14, "3 PM": 15, "4 PM": 16, "5 PM": 17,
    "6 PM": 18, "7 PM": 19, "8 PM": 20, "9 PM": 21, "10 PM": 22, "11 PM": 23,
}

_HOUR_ALTS = sorted(HOUR_PHRASES.keys(), key=len, reverse=True)
# Non-capturing group so embedding this pattern inside another regex doesn't
# add stray capture groups (m.group(1), m.group(2) would otherwise be the
# inner HOUR_RE matches).
_HOUR_PATTERN = r"(?:\b(?:" + "|".join(re.escape(p) for p in _HOUR_ALTS) + r")\b)"
_HOUR_RE = re.compile(_HOUR_PATTERN)


def _parse_time_window(text: str) -> list[int] | None:
    """Extract a [start-inclusive, end-exclusive) list of hours from a phrase.

    Examples:
        "from noon until 2 PM"  -> [12, 13]
        "between 11 AM and 2 PM" -> [11, 12, 13]
        "from 6 PM until 9 PM" -> [18, 19, 20]
    """
    m = re.search(
        r"(?:from|between)\s+(" + _HOUR_PATTERN + r")\s+"
        r"(?:to|until|and)\s+(" + _HOUR_PATTERN + r")",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    # The regex matches case-insensitively, so the matched groups may be in
    # any case. Normalize using the canonical uppercase forms.
    start = HOUR_PHRASES[m.group(1).upper()]
    end = HOUR_PHRASES[m.group(2).upper()]
    if end <= start:
        return None
    return list(range(start, end))


def _pct_to_factor(pct_remaining: float) -> float:
    """For solar_reduction: 'X% reduction' means factor = 1 - X/100."""
    return max(0.0, min(1.0, 1.0 - pct_remaining / 100.0))


def _classify_note(note: str, battery_capacity_kwh: float) -> DirectiveInterpretation:
    """Map a single operator note to a DirectiveInterpretation."""
    text = note.strip()
    hours = _parse_time_window(text)

    # 1. Solar reduction.
    # Three phrasings:
    #   a) "X% reduction ... during <window>"
    #   b) "treated as / about / roughly / around / leave ... ~Y% of the forecast solar"
    #   c) "leave about half of the forecast solar" (word-fraction form)
    m_red_pct = re.search(
        r"(\d{1,3})\s*%\s*(?:reduction|less|shortfall|drop)",
        text,
        flags=re.IGNORECASE,
    )
    m_factor_pct = re.search(
        r"(?:about|roughly|around|approximately|expect(?:s)?|use(?:s)?|treated\s+as|"
        r"available\s+(?:is|will\s+be)|leave)?\s*"
        r"(\d{1,3})\s*%\s+of\s+(?:the\s+)?(?:forecast|usable|expected|the\s+forecast)"
        r"(?:\s+solar(?:\s+output)?)?",
        text,
        flags=re.IGNORECASE,
    )
    # Word fractions -> percentage mapping for solar reduction.
    WORD_FRACTIONS = {
        "half": 50, "quarter": 25, "third": 33, "two thirds": 67, "two-thirds": 67,
        "three quarters": 75, "three-quarters": 75,
    }
    m_factor_word = None
    for phrase, pct in WORD_FRACTIONS.items():
        if re.search(
            rf"(?:about|roughly|around|approximately|leave|only)?\s*{re.escape(phrase)}\s+"
            r"of\s+(?:the\s+)?(?:forecast|usable|expected)?\s*solar",
            text,
            flags=re.IGNORECASE,
        ):
            m_factor_word = (phrase, pct)
            break
    if m_red_pct and hours:
        pct = float(m_red_pct.group(1))
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment=SolarReductionAdjustment(
                hours=hours, factor=_pct_to_factor(pct)
            ),
            explanation=f"Solar reduced by {pct:.0f}% during the noted window.",
        )
    if m_factor_pct and hours:
        pct_remaining = float(m_factor_pct.group(1))
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment=SolarReductionAdjustment(
                hours=hours, factor=pct_remaining / 100.0
            ),
            explanation=f"Only {pct_remaining:.0f}% of solar is usable during the window.",
        )
    if m_factor_word and hours:
        _, pct = m_factor_word
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment=SolarReductionAdjustment(
                hours=hours, factor=pct / 100.0
            ),
            explanation=f"About {pct}% of solar is usable during the window.",
        )

    # 2. Minimum battery reserve.
    # Two phrasings:
    #   a) "at least X kWh ... <window>"
    #   b) "X% of (the) battery capacity ... <window>"
    m_reserve_abs = re.search(
        r"(?:at\s+least|keep|reserve(?:\s+of)?|maintain|with\s+at\s+least)\s+"
        r"(\d+(?:\.\d+)?)\s*kWh",
        text,
        flags=re.IGNORECASE,
    )
    m_reserve_pct = re.search(
        r"(?:at\s+least|keep|reserve)\s+(\d{1,3})\s*%\s+(?:of\s+(?:the\s+)?)?battery(?:\s+capacity)?",
        text,
        flags=re.IGNORECASE,
    )
    if m_reserve_abs and hours:
        min_kwh = float(m_reserve_abs.group(1))
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="minimum_battery_reserve",
            structured_adjustment=MinimumBatteryReserveAdjustment(
                hours=hours, minimum_energy_kwh=min_kwh
            ),
            explanation=f"At least {min_kwh} kWh must remain in the battery during the window.",
        )
    if m_reserve_pct and hours and battery_capacity_kwh > 0:
        pct = float(m_reserve_pct.group(1))
        min_kwh = battery_capacity_kwh * pct / 100.0
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="minimum_battery_reserve",
            structured_adjustment=MinimumBatteryReserveAdjustment(
                hours=hours, minimum_energy_kwh=round(min_kwh, 4)
            ),
            explanation=f"{pct:.0f}% of capacity = {min_kwh:.2f} kWh must remain during the window.",
        )

    # 3. No-charge window — any phrasing that says "charger/charging ... isolated/disabled/unavailable".
    # Use a loose match: charger or charging followed (within ~30 chars) by a stop word.
    if re.search(
        r"(?:charger|charging(?:\s+circuit)?).{0,40}"
        r"(?:isolat|disabl|unavailab|offline|locked|blocked|maintenance|down|off|not\s+available)",
        text,
        flags=re.IGNORECASE,
    ) and hours:
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="no_charge_window",
            structured_adjustment=NoChargeWindowAdjustment(hours=hours),
            explanation="Battery charging is disabled during the noted window.",
        )

    # 4. No-discharge window — "must not discharge / do not discharge".
    if re.search(
        r"(?:must\s+not\s+discharge|do\s+not\s+discharge|discharg(?:e|ing)\s+(?:is|will\s+be)\s+"
        r"(?:disabled|forbidden|blocked)|cannot\s+discharge)",
        text,
        flags=re.IGNORECASE,
    ) and hours:
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="no_discharge_window",
            structured_adjustment=NoDischargeWindowAdjustment(hours=hours),
            explanation="Battery discharging is blocked during the noted window.",
        )

    # 5. Max-grid window — "must not exceed X kWh / capped at X / no more than X / stay at or below X kWh ... <window>".
    m_grid_cap = re.search(
        r"(?:must\s+not\s+exceed|cap(?:ped)?\s+at|limit(?:ed)?\s+to|no\s+more\s+than|"
        r"stay\s+at\s+or\s+below|grid\s+intake\s+must\s+stay\s+at\s+or\s+below|"
        r"transformer\s+limit\s+is|feeder\s+is\s+(?:overloaded|constrained))"
        r"\s*(\d+(?:\.\d+)?)\s*kWh",
        text,
        flags=re.IGNORECASE,
    )
    # Catch "X kWh of grid import" pattern when in the same sentence as the window.
    if not m_grid_cap and hours:
        m_grid_cap2 = re.search(
            r"(\d+(?:\.\d+)?)\s*kWh\s+of\s+grid\s+import",
            text,
            flags=re.IGNORECASE,
        )
        if m_grid_cap2:
            m_grid_cap = m_grid_cap2
    if m_grid_cap and hours:
        cap = float(m_grid_cap.group(1))
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="max_grid_window",
            structured_adjustment=MaxGridWindowAdjustment(hours=hours, max_grid_kwh=cap),
            explanation=f"Grid import is capped at {cap} kWh per hour during the window.",
        )

    # 6. Default: no_op.
    return DirectiveInterpretation(
        note_index=0,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation="This note does not affect today's 24-hour energy schedule.",
    )


def parse_notes(inp: InputSchema) -> list[DirectiveInterpretation]:
    """Parse a list of 1–3 operator notes into one directive per note.

    Each returned DirectiveInterpretation has note_index reflecting its
    position in the input list.
    """
    capacity = inp.battery.capacity_kwh if inp.battery else 0.0
    out: list[DirectiveInterpretation] = []
    for i, n in enumerate(inp.operator_notes):
        d = _classify_note(n, capacity)
        d = d.model_copy(update={"note_index": i})
        out.append(d)
    return out
