"""Prompt templates for the directive-interpretation LLM.

We point at the base Qwen2.5-0.5B-Instruct directly (no LoRA, no
fine-tuning) and rely on a tight, focused prompt. The 0.5B model is
less capable than 1.5B so we use stronger few-shot examples and
explicit disambiguation rules.

Speed matters on CPU so the prompt is intentionally small:

- Single system prompt describing the contract + directive mapping rules.
- TWO few-shot examples covering the trickiest boundary cases
  (cleaning → solar_reduction; ambiguous time phrasing → hours).
- Live user prompt with the actual notes.

Total prompt size: ~650 tokens. At ~3 tok/s on CPU that's ~3-4 minutes
per inference call on the first run, faster on repeated notes thanks
to our response cache.
"""

# Tight system prompt. Includes schema + hour-handling + key mappings.
SYSTEM_PROMPT = """\
You convert operator notes into a JSON array of energy directives.

ONE directive per note, in the same order.

DIRECTIVE_TYPES:
  solar_reduction         {"hours":[..], "factor":0.0-1.0}
                          FACTOR = fraction REMAINING (0.2 = 20% left)
  minimum_battery_reserve {"hours":[..], "minimum_energy_kwh":float}
  no_charge_window        {"hours":[..]}   (charging forced to zero)
  no_discharge_window     {"hours":[..]}   (discharging forced to zero)
  max_grid_window         {"hours":[..], "max_grid_kwh":float}
  no_op                   adjustment=null, applies=false

KEY DISAMBIGUATION RULES:
- "washing / cleaning / soiling / maintenance on the panels or solar
  arrays"  ->  ALWAYS solar_reduction. NEVER no_charge_window.
  Solar panels being washed means less sun reaches them, not that
  the battery cannot charge.
- "no charge / don't charge / cannot charge / charging disabled"
  ->  no_charge_window.
- "no discharge / don't discharge / cannot discharge" -> no_discharge_window.
- "reserve / keep at least / minimum N kWh" -> minimum_battery_reserve.
- "grid cap / limit grid / max grid / no more than N kWh from grid"
  -> max_grid_window.

HOURS (0..23, integers, start-inclusive end-exclusive):
  "noon to 2 PM"     -> [12, 13]
  "6 PM to 9 PM"     -> [18, 19, 20]
  "from 5 to 9 PM"   -> [17, 18, 19, 20]
  "evening peak 5-9" -> [17, 18, 19, 20]
  "at 3 PM"          -> [15]   (single hour)
  "all day" / "the entire shift" -> every hour 0..23

RELEVANCE:
- Announcements, inspections, schedule changes, equipment notices
  that don't change the energy schedule -> "no_op".

OUTPUT (strict):
- JSON array. One object per note, in order.
- Each object: note_index (int), applies (bool),
  directive_type (one of the 6 strings), structured_adjustment (object|null),
  explanation (short sentence).
- Return ONLY the JSON. No markdown. No commentary. No preamble.
"""


def build_user_prompt(notes: list[str]) -> str:
    body = "\n".join(f"{i+1}. {n}" for i, n in enumerate(notes))
    return (
        "Notes:\n"
        + body
        + "\n\nReturn only the JSON array."
    )


# Two few-shot examples. First one explicitly resolves the
# "washing panels -> solar_reduction" ambiguity that the 0.5B model
# misclassified in earlier tests. Second one shows multiple directives
# in a single shot, including a no_op.
FEW_SHOT_USER_1 = """\
Notes:
1. The facilities team will wash the rooftop solar panels from noon until 2 PM, so usable solar should be treated as 25%.
JSON array only."""

FEW_SHOT_ASSISTANT_1 = (
    '['
    '{"note_index":0,"applies":true,"directive_type":"solar_reduction",'
    '"structured_adjustment":{"hours":[12,13],"factor":0.25},'
    '"explanation":"Solar reduced to 25% during noon-2 PM while panels are washed."}'
    ']'
)


FEW_SHOT_USER_2 = """\
Notes:
1. Hold at least 25 kWh in the battery from 6 PM onward.
2. Do not charge between 4 and 6 PM.
3. Inspection notice at 10 AM, no impact on schedule.
4. Cap grid at 4 kWh from 5 PM to 9 PM.
JSON array only."""

FEW_SHOT_ASSISTANT_2 = (
    '['
    '{"note_index":0,"applies":true,"directive_type":"minimum_battery_reserve",'
    '"structured_adjustment":{"hours":[18,19,20,21,22,23],"minimum_energy_kwh":25},'
    '"explanation":"Battery floor 25 kWh from 6 PM onward."},'

    '{"note_index":1,"applies":true,"directive_type":"no_charge_window",'
    '"structured_adjustment":{"hours":[16,17]},'
    '"explanation":"Charging disabled in the 4-6 PM window."},'

    '{"note_index":2,"applies":false,"directive_type":"no_op",'
    '"structured_adjustment":null,'
    '"explanation":"Inspection does not affect the energy schedule."},'

    '{"note_index":3,"applies":true,"directive_type":"max_grid_window",'
    '"structured_adjustment":{"hours":[17,18,19,20],"max_grid_kwh":4},'
    '"explanation":"Grid capped at 4 kWh during 5-9 PM."}'
    ']'
)


def build_messages(notes: list[str]) -> list[dict]:
    """Chat-template input: system + 2 few-shot turns + the live user prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_USER_1},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT_1},
        {"role": "user", "content": FEW_SHOT_USER_2},
        {"role": "assistant", "content": FEW_SHOT_ASSISTANT_2},
        {"role": "user", "content": build_user_prompt(notes)},
    ]
