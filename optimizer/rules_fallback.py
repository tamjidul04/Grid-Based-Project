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

    Handles many connector phrasings:
        "from noon until 2 PM"          -> [12, 13]
        "between 11 AM and 2 PM"        -> [11, 12, 13]
        "from 6 PM until 9 PM"          -> [18, 19, 20]
        "1 PM to 3 PM"                  -> [13, 14]
        "10 PM until midnight"          -> [22, 23]
        "3 PM - 5 PM"                   -> [15, 16] (also handles en-dash/em-dash)
        "6 PM onward"                   -> [18..23]
        "from 5 PM onward"              -> [17..23]
    """
    # Pattern A: explicit "from X to/until Y"
    m = re.search(
        r"(?:from|between)?\s*(" + _HOUR_PATTERN + r")\s+"
        r"(?:to|until|and|-|–|—)\s*(" + _HOUR_PATTERN + r")",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        start = HOUR_PHRASES[m.group(1).upper()]
        end = HOUR_PHRASES[m.group(2).upper()]
        if end > start:
            return list(range(start, end))

    # Pattern B: "X to/until Y" with the connector right after a number phrase
    # (catches "1 PM to 3 PM" where the leading "from" is absent).
    m = re.search(
        r"(" + _HOUR_PATTERN + r")\s+(?:to|until|and|-|–|—)\s*(" + _HOUR_PATTERN + r")",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        start = HOUR_PHRASES[m.group(1).upper()]
        end = HOUR_PHRASES[m.group(2).upper()]
        if end > start:
            return list(range(start, end))

    # Pattern C: "X PM onward" / "from X PM onward" / "X onward"
    m = re.search(
        r"(?:from\s+)?(" + _HOUR_PATTERN + r")\s+onward",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        start = HOUR_PHRASES[m.group(1).upper()]
        return list(range(start, 24))

    # Pattern D: "X PM until/to midnight" -> [X, ..., 23]
    m = re.search(
        r"(" + _HOUR_PATTERN + r")\s+(?:to|until|through|till)\s+midnight",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        start = HOUR_PHRASES[m.group(1).upper()]
        return list(range(start, 24))

    return None


def _pct_to_factor(pct_remaining: float) -> float:
    """For solar_reduction: 'X% reduction' means factor = 1 - X/100."""
    return max(0.0, min(1.0, 1.0 - pct_remaining / 100.0))


def _classify_note(note: str, battery_capacity_kwh: float) -> DirectiveInterpretation:
    """Map a single operator note to a DirectiveInterpretation."""
    text = note.strip()
    hours = _parse_time_window(text)

    # 1. Solar reduction.
    # Many phrasings. The trigger is: a percentage (or fraction) is given AND
    # the note clearly pertains to solar (mentions "solar", "panel",
    # "inverter", "array", or an event that only affects solar: cleaning,
    # dust, cloud, inspection, etc.).
    solar_event = re.search(
        r"\b(solar|panels?|array|inverter|cloud|cloud\s+cover|dust|soiling|cleaning|"
        r"clean|wash|scrub|inspection|storm)\b",
        text,
        flags=re.IGNORECASE,
    )

    # a) "X% reduction ... during <window>"
    m_red_pct = re.search(
        r"(\d{1,3})\s*%\s*(?:reduction|less|shortfall|drop)",
        text,
        flags=re.IGNORECASE,
    )
    # b) "treated as / about / roughly / around / leave ... ~Y% of the forecast solar"
    m_factor_pct = re.search(
        r"(?:about|roughly|around|approximately|expect(?:s)?|use(?:s)?|treated\s+as|"
        r"available\s+(?:is|will\s+be)|leave|at|to|limit(?:ed)?\s+to|"
        r"drops?\s+to|output\s+limited\s+to|usable)?\s*"
        r"(\d{1,3})\s*%\s*(?:\s+of\s+(?:the\s+)?(?:forecast|usable|expected|the\s+forecast)"
        r"(?:\s+solar(?:\s+output)?)?)?",
        text,
        flags=re.IGNORECASE,
    )
    # c) "leave about half of the forecast solar" (word-fraction form)
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

    # Helper: decide which percentage to use, given solar-context and window.
    def _try_solar(pct: float) -> DirectiveInterpretation | None:
        if not hours or not solar_event:
            return None
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type="solar_reduction",
            structured_adjustment=SolarReductionAdjustment(
                hours=hours, factor=max(0.0, min(1.0, pct / 100.0))
            ),
            explanation=f"Only {pct:.0f}% of solar is usable during the window.",
        )

    if m_red_pct and hours and solar_event:
        # "X% reduction" -> only X% remains
        pct_remaining = 100.0 - float(m_red_pct.group(1))
        r = _try_solar(pct_remaining)
        if r:
            return r
    if m_factor_pct and hours and solar_event:
        pct_remaining = float(m_factor_pct.group(1))
        # If the matched context was "of (the) forecast solar", the number
        # is the remaining fraction. If it was just "at X%" with no "of",
        # treat it as remaining too.
        r = _try_solar(pct_remaining)
        if r:
            return r
    if m_factor_word and hours and solar_event:
        _, pct = m_factor_word
        r = _try_solar(pct)
        if r:
            return r

    # 2. Minimum battery reserve.
    # Many phrasings:
    #   "at least X kWh ... <window>"
    #   "Reserve X kWh minimum ... <window>"
    #   "Hold at least X kWh ... <window>"
    #   "Maintain at least X kWh in the battery ... <window>"
    #   "Battery floor of X kWh ... <window>"
    #   "X% of (the) battery capacity ... <window>"
    m_reserve_abs = re.search(
        r"(?:at\s+least|keep|reserve(?:\s+\d+(?:\.\d+)?)?|maintain|hold|"
        r"with\s+at\s+least|floor\s+of|reserve\s+of|minimum\s+of)\s+"
        r"(\d+(?:\.\d+)?)\s*kWh(?:\s+minimum|\s+in\s+the\s+battery|\s+from)?",
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

    # 3. No-charge window — "charger/charging ... isolated/disabled/suspended" OR
    # "do not charge between X and Y".
    if re.search(
        r"(?:charger|charging(?:\s+circuit)?).{0,40}"
        r"(?:isolat|disabl|unavailab|offline|locked|blocked|maintenance|"
        r"down|off|not\s+available|suspend)",
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
    if re.search(
        r"\b(?:do\s+not|don'?t|cannot|can'?t|will\s+not|won'?t|no|"
        r"charge(?:\s+is)?\s+disabled|"
        r"charging\s+(?:is|will\s+be)\s+(?:disabled|suspended|blocked|forbidden))\s+charge",
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

    # 4. No-discharge window — many phrasings:
    #   "do not discharge from X to Y"
    #   "no discharge from X to Y"
    #   "Discharge disabled from X to Y"
    #   "Battery discharge disabled from X to Y"
    #   "cannot discharge"
    if re.search(
        r"(?:must\s+not\s+discharge|do\s+not\s+discharge|no\s+discharge|"
        r"discharg(?:e|ing)\s+(?:is|will\s+be)\s+(?:disabled|forbidden|blocked|suspended)|"
        r"discharg(?:e|ing)\s+disabled|"
        r"cannot\s+discharge|"
        r"(?:battery\s+)?discharge\s+(?:is|will\s+be)\s+disabled)",
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

    # 5. Max-grid window.
    # Many phrasings:
    #   "Cap grid at X kWh"
    #   "Limit grid imports to X kWh"
    #   "Limit grid to X kWh"
    #   "Grid max X kWh"
    #   "Grid cap X kWh"   (kWh right after cap, window after)
    #   "Restrict grid to X kWh"
    #   "Grid import cap of X kWh"
    #   "must not exceed X kWh"
    #   "no more than X kWh"
    #   "X kWh of grid import"
    m_grid_cap = re.search(
        r"(?:must\s+not\s+exceed|cap(?:ped|ping)?\s+(?:grid\s+(?:at|to)|at|of)?|"
        r"limit(?:ed|ing)?\s+(?:grid\s+imports?\s+to|grid\s+to)|"
        r"no\s+more\s+than|stay\s+at\s+or\s+below|"
        r"grid\s+intake\s+must\s+stay\s+at\s+or\s+below|"
        r"transformer\s+limit\s+is|feeder\s+is\s+(?:overloaded|constrained)|"
        r"restrict\s+grid\s+to|grid\s+max|grid\s+import\s+cap\s+of|"
        r"grid\s+(?:is\s+)?(?:capped|limited)\s+(?:to|at)|"
        r"grid\s+cap)"
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
