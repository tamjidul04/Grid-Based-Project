"""Synthetic training data generator for the directive-interpretation LLM.

The fine-tuned LLM's only job is to read 1-3 operator notes and emit a JSON
list of `directive_interpretation` objects. We generate training examples by:

    1. For each of the 6 directive types, write a bank of natural phrasings
       and parameter ranges (hours, percentages, kWh values).
    2. Sample a directive type + phrasing + parameters, then build the
       matching operator note string and the expected JSON output.
    3. Mix in 40-50% distractors (no_op notes) and 30% multi-note cases so
       the LLM learns to handle 1-, 2-, and 3-note inputs.

Output: data/synthetic/train.jsonl — one JSON object per line:
    {"messages": [
        {"role": "system", "content": <system prompt>},
        {"role": "user", "content": <user prompt with notes>},
        {"role": "assistant", "content": <JSON output>},
    ]}

We use the Qwen2.5-Instruct chat format (system + user + assistant turns).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HOURS_AMPM = {
    0: "12 AM", 1: "1 AM", 2: "2 AM", 3: "3 AM", 4: "4 AM", 5: "5 AM",
    6: "6 AM", 7: "7 AM", 8: "8 AM", 9: "9 AM", 10: "10 AM", 11: "11 AM",
    12: "noon", 13: "1 PM", 14: "2 PM", 15: "3 PM", 16: "4 PM", 17: "5 PM",
    18: "6 PM", 19: "7 PM", 20: "8 PM", 21: "9 PM", 22: "10 PM", 23: "11 PM",
}

# Generic distractor phrases — these are operator notes that should NOT
# affect the 24-hour energy schedule (per the spec's `no_op` rule).
DISTRACTOR_PHRASES = [
    "The sports office moved next month's registration deadline.",
    "The library is extending book-return hours next week.",
    "A seminar room booking was moved to next week.",
    "The student affairs office will publish club notices tomorrow.",
    "Reminder: submit next month's budget by Friday.",
    "Quarterly safety briefing is scheduled for next Wednesday.",
    "The campus shuttle schedule will change starting Monday.",
    "Cafeteria menu has been updated for the upcoming week.",
    "Maintenance staff will repaint the corridor this weekend.",
    "IT services will perform a routine network check overnight.",
    "The dean's office has issued a memo about parking.",
    "Building access cards will be reissued next month.",
    "The grounds crew is planting new trees near the entrance.",
    "A staff meeting has been rescheduled to next Tuesday.",
    "Lost-and-found items can be claimed at the security desk.",
]


SYSTEM_PROMPT = """\
You are a directive interpretation assistant for a campus microgrid \
controller. You receive 1-3 short free-text operator notes and must return \
a JSON array describing the energy-schedule directives they imply.

For EACH note, emit ONE entry with these fields:
- "note_index": integer (0, 1, or 2 — the position of the note in the input)
- "applies": boolean (false if the note doesn't affect the energy schedule)
- "directive_type": one of: "solar_reduction", "minimum_battery_reserve", \
  "no_charge_window", "no_discharge_window", "max_grid_window", "no_op"
- "structured_adjustment": an object describing the directive, or null
- "explanation": one short sentence explaining your reasoning

The 6 directive types:
- solar_reduction: {"hours": [h1, h2, ...], "factor": 0.0-1.0} — \
  factor is the FRACTION REMAINING (so 80% reduction means factor=0.2).
- minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": float}
- no_charge_window: {"hours": [...]} — charging must be zero in these hours
- no_discharge_window: {"hours": [...]} — discharging must be zero in these hours
- max_grid_window: {"hours": [...], "max_grid_kwh": float}
- no_op: structured_adjustment is null, applies is false

CRITICAL RULES:
- Hours are integers 0-23. Windows are START-INCLUSIVE, END-EXCLUSIVE \
  ("noon until 2 PM" -> hours [12, 13]).
- For irrelevant notes (announcements, schedule changes, etc.), emit \
  directive_type "no_op" with applies=false and structured_adjustment=null.
- Return ONLY the JSON array. No markdown, no commentary, no preamble.
"""


def fmt_hours(start: int, end: int) -> str:
    """Format a [start, end) hour window in natural language."""
    return f"from {HOURS_AMPM[start]} until {HOURS_AMPM[end]}"


def sample_window(rng: random.Random, min_len: int = 1, max_len: int = 6) -> tuple[int, int]:
    """Sample a (start, end) window with end > start and length in [min_len, max_len]."""
    length = rng.randint(min_len, max_len)
    max_start = 23 - length
    start = rng.randint(0, max_start)
    return start, start + length


def gen_solar_reduction(rng: random.Random) -> tuple[str, dict]:
    """Generate a solar_reduction note + the matching directive JSON."""
    start, end = sample_window(rng, 1, 4)
    window = fmt_hours(start, end)
    pct_remaining = rng.choice([10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
    pct_reduction = 100 - pct_remaining

    templates = [
        f"Solar panels will be cleaned {window}; during cleaning, usable solar should be treated as roughly {pct_remaining}% of the forecast.",
        f"Expect about a {pct_reduction}% reduction in rooftop solar output {window} due to cloud cover.",
        f"Rooftop solar production will drop to approximately {pct_remaining}% of forecast {window} while technicians inspect the array.",
        f"From the inverter logs: solar generation is reduced by {pct_reduction}% {window}.",
        f"Solar availability will be limited to about {pct_remaining}% of normal output {window}.",
        f"Expect an {pct_reduction}% shortfall in solar {window} because of scheduled maintenance.",
        f"Cloud cover during panel inspection will leave about half of the forecast solar output from {HOURS_AMPM[start]} until {HOURS_AMPM[end]}." if pct_remaining == 50 else
            f"Solar generation will be reduced to about {pct_remaining}% {window}.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {
            "hours": list(range(start, end)),
            "factor": pct_remaining / 100.0,
        },
        "explanation": f"Solar reduced to {pct_remaining}% (factor {pct_remaining/100:.2f}) during {window}.",
    }
    return note, directive


def gen_min_reserve_abs(rng: random.Random) -> tuple[str, dict]:
    """Generate a minimum_battery_reserve note (absolute kWh) + directive."""
    start, end = sample_window(rng, 2, 5)
    window = fmt_hours(start, end)
    min_kwh = rng.choice([40, 50, 60, 70, 80, 90, 100, 120, 150, 180])
    reason = rng.choice([
        "for emergency operations",
        "for the data center backup",
        "to cover evening peak demand",
        "for critical load support",
        "for fire-safety reserve requirements",
        "for emergency services",
    ])
    templates = [
        f"Keep at least {min_kwh} kWh in the battery {window} {reason}.",
        f"Reserve {min_kwh} kWh of stored battery energy {window} {reason}.",
        f"Maintain a minimum reserve of {min_kwh} kWh {window} {reason}.",
        f"The battery must hold at least {min_kwh} kWh {window} {reason}.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {
            "hours": list(range(start, end)),
            "minimum_energy_kwh": float(min_kwh),
        },
        "explanation": f"At least {min_kwh} kWh must remain in the battery {window}.",
    }
    return note, directive


def gen_min_reserve_pct(rng: random.Random) -> tuple[str, dict]:
    """Generate a percentage-based minimum_battery_reserve note + directive.

    The LLM should learn to compute the kWh value from the percentage.
    We assume a default battery capacity of 200 kWh in the synthetic data
    (matches the most common in the sample cases).
    """
    start, end = sample_window(rng, 2, 5)
    window = fmt_hours(start, end)
    pct = rng.choice([25, 30, 40, 50, 60, 75])
    capacity = 200  # assumed baseline
    min_kwh = capacity * pct / 100.0
    templates = [
        f"Keep at least {pct}% of the battery capacity stored in the battery {window} for emergency operations.",
        f"Maintain {pct}% battery reserve {window} for emergency services.",
        f"Hold at least {pct}% of total capacity in the battery {window}.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {
            "hours": list(range(start, end)),
            "minimum_energy_kwh": min_kwh,
        },
        "explanation": f"{pct}% of {capacity} kWh capacity = {min_kwh} kWh reserve {window}.",
    }
    return note, directive


def gen_no_charge(rng: random.Random) -> tuple[str, dict]:
    start, end = sample_window(rng, 1, 4)
    window = fmt_hours(start, end)
    templates = [
        f"The battery charger will be isolated {window} for electrical maintenance.",
        f"Charging will be disabled {window} while technicians inspect the charger.",
        f"The charging circuit will be unavailable {window}.",
        f"Battery charging is offline {window} for routine service.",
        f"Do not charge the battery {window} — the charging system is locked out.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "no_charge_window",
        "structured_adjustment": {
            "hours": list(range(start, end)),
        },
        "explanation": f"Battery charging is disabled {window}.",
    }
    return note, directive


def gen_no_discharge(rng: random.Random) -> tuple[str, dict]:
    start, end = sample_window(rng, 1, 4)
    window = fmt_hours(start, end)
    templates = [
        f"For protection testing, the battery must not discharge {window}.",
        f"Do not discharge the battery {window} during relay testing.",
        f"Battery discharging is blocked {window} for grid support.",
        f"The battery discharge circuit is offline {window}.",
        f"Discharging must be prevented {window} for safety testing.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "no_discharge_window",
        "structured_adjustment": {
            "hours": list(range(start, end)),
        },
        "explanation": f"Battery discharging is blocked {window}.",
    }
    return note, directive


def gen_max_grid(rng: random.Random) -> tuple[str, dict]:
    start, end = sample_window(rng, 1, 4)
    window = fmt_hours(start, end)
    cap = rng.choice([80, 100, 120, 150, 155, 180, 190, 200, 220, 250])
    templates = [
        f"From {HOURS_AMPM[start]} until {HOURS_AMPM[end]}, campus grid import must not exceed {cap} kWh in any hour because the feeder is overloaded.",
        f"Grid intake must stay at or below {cap} kWh {window} while the substation is constrained.",
        f"The evening transformer limit is {cap} kWh of grid import {window}.",
        f"Campus grid import capped at {cap} kWh per hour {window}.",
        f"Limit grid draw to {cap} kWh {window} for feeder maintenance.",
    ]
    note = rng.choice(templates)
    directive = {
        "note_index": 0,
        "applies": True,
        "directive_type": "max_grid_window",
        "structured_adjustment": {
            "hours": list(range(start, end)),
            "max_grid_kwh": float(cap),
        },
        "explanation": f"Grid import capped at {cap} kWh per hour {window}.",
    }
    return note, directive


def gen_no_op(rng: random.Random) -> tuple[str, dict]:
    """A no_op directive. Note: we don't actually use this as a primary
    directive — distractors come from DISTRACTOR_PHRASES below."""
    note = rng.choice(DISTRACTOR_PHRASES)
    directive = {
        "note_index": 0,
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "This note does not affect today's 24-hour energy schedule.",
    }
    return note, directive


DIRECTIVE_GENERATORS = {
    "solar_reduction": gen_solar_reduction,
    "min_reserve_abs": gen_min_reserve_abs,
    "min_reserve_pct": gen_min_reserve_pct,
    "no_charge": gen_no_charge,
    "no_discharge": gen_no_discharge,
    "max_grid": gen_max_grid,
}


def build_example(rng: random.Random) -> dict:
    """Build one (notes, directives) training example.

    The example has 1, 2, or 3 notes (chosen by weighted random). Each note
    is either a real directive or a distractor. Each generated directive has
    its `note_index` set to its position in the final notes list.
    """
    n_notes = rng.choices([1, 2, 3], weights=[3, 5, 2])[0]
    has_distractor = rng.random() < 0.4  # ~40% of multi-note cases have a distractor

    note_types: list[str] = []
    for _ in range(n_notes):
        if has_distractor and rng.random() < 0.35:
            note_types.append("distractor")
        else:
            note_types.append(rng.choice(list(DIRECTIVE_GENERATORS.keys())))

    notes: list[str] = []
    directives: list[dict] = []
    for i, nt in enumerate(note_types):
        if nt == "distractor":
            note_text = rng.choice(DISTRACTOR_PHRASES)
            d = {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "This note does not affect today's 24-hour energy schedule.",
            }
        else:
            note_text, d = DIRECTIVE_GENERATORS[nt](rng)
            d["note_index"] = i
        notes.append(note_text)
        directives.append(d)

    user_prompt = (
        "Operator notes:\n"
        + "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
        + "\n\nReturn the directive_interpretation JSON array only."
    )
    assistant = json.dumps(directives, indent=2)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic training data.")
    parser.add_argument("--n", type=int, default=1500, help="Number of examples to generate.")
    parser.add_argument("--out", type=str, default="data/synthetic/train.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    here = Path(__file__).resolve().parent.parent
    out_path = here / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(args.n):
            ex = build_example(rng)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if (i + 1) % 200 == 0:
                print(f"  generated {i+1}/{args.n}")

    print(f"Wrote {args.n} examples to {out_path}")


if __name__ == "__main__":
    main()
