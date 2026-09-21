"""Generate 40 synthetic test cases from templates and validate against
the live Render URL. Each case paraphrases a real operator directive.

Categories:
  1-8:   single-note solar_reduction (varied hour phrasings)
  9-16:  single-note no_charge_window / no_discharge_window
  17-24: single-note minimum_battery_reserve (different kWh and hours)
  25-32: single-note max_grid_window (different kWh caps)
  33-40: multi-note cases (2-3 directives per case, including no_op)
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(42)  # reproducible


def make_hours_24():
    """24 hours of dummy grid data: realistic-ish demand, solar, tariff."""
    hours = []
    # Tariff: low overnight, high in evening peak
    for h in range(24):
        if 0 <= h <= 5 or h == 23:
            tariff = 5.5
        elif 6 <= h <= 16:
            tariff = 8.0
        else:  # 17-22 evening peak
            tariff = 14.0
        # Demand: low overnight, ramp up morning, dip midday, peak evening
        if 0 <= h <= 5:
            demand = 60 + random.randint(-5, 5)
        elif 6 <= h <= 9:
            demand = 90 + random.randint(-5, 10)
        elif 10 <= h <= 15:
            demand = 75 + random.randint(-5, 10)
        elif 16 <= h <= 21:
            demand = 130 + random.randint(-10, 10)
        else:
            demand = 100 + random.randint(-5, 10)
        # Solar: zero at night, peak midday
        if 6 <= h <= 18:
            solar = max(0.0, 80 * (1 - abs(h - 12) / 6) + random.uniform(-5, 5))
        else:
            solar = 0.0
        hours.append({
            "hour": h,
            "demand_kwh": round(demand, 2),
            "solar_kwh": round(solar, 2),
            "tariff_bdt_per_kwh": tariff,
        })
    return hours


def make_battery():
    return {
        "capacity_kwh": 200.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 20.0,
        "max_charge_kwh_per_hour": 50.0,
        "max_discharge_kwh_per_hour": 50.0,
    }


def case(scenario_id, notes, expected_directives, expected_cost):
    return {
        "id": scenario_id,
        "input": {
            "scenario_id": scenario_id,
            "operator_notes": notes,
            "hours": make_hours_24(),
            "battery": make_battery(),
        },
        "expected_directives": expected_directives,
        "expected_cost": expected_cost,
    }


# Solar reduction (cases 1-8)
solar_cases = [
    ("Facilities will wash the rooftop solar panels from noon until 2 PM; usable solar should be treated as 25%.", [12, 13], 0.25),
    ("Cleaning crew will scrub the solar array from 10 AM to noon, solar at 30%.", [10, 11], 0.30),
    ("Dust storm forecast; solar output limited to 40% from 11 AM to 3 PM.", [11, 12, 13, 14], 0.40),
    ("Maintenance on solar inverters from 1 PM to 4 PM; treat solar as 50%.", [13, 14, 15], 0.50),
    ("Solar soiling event from 9 AM to 11 AM; reduce solar to 60%.", [9, 10], 0.60),
    ("Heavy cloud cover forecast 2 PM - 5 PM; usable solar drops to 35%.", [14, 15, 16], 0.35),
    ("Panel cleaning scheduled from 8 AM until 10 AM; solar at 20%.", [8, 9], 0.20),
    ("Inspection of solar arrays from 3 PM to 6 PM; treat solar as 70%.", [15, 16, 17], 0.70),
]

# No charge / discharge windows (cases 9-16)
window_cases = [
    ("Battery charger isolated from 2 AM to 5 AM for maintenance.", "no_charge_window", [2, 3, 4], None),
    ("Do not charge the battery between 4 PM and 6 PM today.", "no_charge_window", [16, 17], None),
    ("Charging disabled from midnight to 6 AM.", "no_charge_window", [0, 1, 2, 3, 4, 5], None),
    ("Battery cannot discharge from 6 PM to 9 PM.", "no_discharge_window", [18, 19, 20], None),
    ("For protection testing, no discharge from 7 PM to 8 PM.", "no_discharge_window", [19], None),
    ("Discharge disabled from 10 PM until midnight.", "no_discharge_window", [22, 23], None),
    ("Charging suspended between noon and 1 PM for grid balancing.", "no_charge_window", [12], None),
    ("Battery discharge disabled from 5 PM to 7 PM.", "no_discharge_window", [17, 18], None),
]

# Minimum battery reserve (cases 17-24)
reserve_cases = [
    ("Keep at least 50 kWh in battery from 6 PM to 9 PM.", [18, 19, 20], 50.0),
    ("Reserve 80 kWh minimum in the battery from 5 PM onward.", [17, 18, 19, 20, 21, 22, 23], 80.0),
    ("Hold at least 30 kWh from 8 PM to 11 PM for emergencies.", [20, 21, 22], 30.0),
    ("Battery floor of 100 kWh from 4 PM until 8 PM.", [16, 17, 18, 19], 100.0),
    ("Maintain at least 60 kWh in the battery from 7 PM to 10 PM.", [19, 20, 21], 60.0),
    ("Reserve 40 kWh minimum from noon to 6 PM.", [12, 13, 14, 15, 16, 17], 40.0),
    ("Hold at least 25 kWh from 9 PM until midnight.", [21, 22, 23], 25.0),
    ("Battery reserve of 150 kWh from 3 PM to 9 PM.", [15, 16, 17, 18, 19, 20], 150.0),
]

# Max grid window (cases 25-32)
grid_cap_cases = [
    ("Cap grid at 100 kWh from 5 PM to 9 PM.", [17, 18, 19, 20], 100.0),
    ("Limit grid imports to 80 kWh between 6 PM and 10 PM.", [18, 19, 20, 21], 80.0),
    ("Grid max 50 kWh from 4 PM to 8 PM due to feeder limit.", [16, 17, 18, 19], 50.0),
    ("Restrict grid to 120 kWh per hour from 7 PM to 11 PM.", [19, 20, 21, 22], 120.0),
    ("Cap grid at 60 kWh from 5 PM to 7 PM.", [17, 18], 60.0),
    ("Limit grid to 200 kWh from 6 PM to 9 PM.", [18, 19, 20], 200.0),
    ("Grid import cap of 90 kWh from 8 PM to midnight.", [20, 21, 22, 23], 90.0),
    ("Cap grid at 150 kWh from 3 PM to 6 PM.", [15, 16, 17], 150.0),
]

# Multi-note cases (cases 33-40)
multi_cases = [
    (
        ["Solar cleaning 1 PM to 3 PM at 30%.", "Sports event rescheduled, no impact."],
        "solar_reduction + no_op",
        [{"directive_type":"solar_reduction","structured_adjustment":{"hours":[13,14],"factor":0.30}},
         {"directive_type":"no_op","applies":False,"structured_adjustment":None}],
    ),
    (
        ["Reserve 70 kWh from 6 PM onward.", "Do not charge between 4 PM and 6 PM."],
        "reserve + no_charge",
        [{"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[18,19,20,21,22,23],"minimum_energy_kwh":70}},
         {"directive_type":"no_charge_window","structured_adjustment":{"hours":[16,17]}}],
    ),
    (
        ["No discharge 5 PM to 7 PM.", "Cap grid at 100 kWh from 5 PM to 9 PM."],
        "no_discharge + grid_cap",
        [{"directive_type":"no_discharge_window","structured_adjustment":{"hours":[17,18]}},
         {"directive_type":"max_grid_window","structured_adjustment":{"hours":[17,18,19,20],"max_grid_kwh":100}}],
    ),
    (
        ["Reserve 50 kWh from 7 PM to 10 PM.", "Solar cleaning 11 AM to 1 PM at 40%."],
        "reserve + solar_reduction",
        [{"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[19,20,21],"minimum_energy_kwh":50}},
         {"directive_type":"solar_reduction","structured_adjustment":{"hours":[11,12],"factor":0.40}}],
    ),
    (
        ["Grid cap 80 kWh 6 PM to 10 PM.", "Reserve 100 kWh from 4 PM onward.", "Building inspection 10 AM, no impact."],
        "grid_cap + reserve + no_op",
        [{"directive_type":"max_grid_window","structured_adjustment":{"hours":[18,19,20,21],"max_grid_kwh":80}},
         {"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[16,17,18,19,20,21,22,23],"minimum_energy_kwh":100}},
         {"directive_type":"no_op","applies":False,"structured_adjustment":None}],
    ),
    (
        ["Solar cleaning noon to 2 PM at 25%.", "Do not charge 4 PM to 6 PM.", "Reserve 50 kWh from 6 PM onward."],
        "solar + no_charge + reserve",
        [{"directive_type":"solar_reduction","structured_adjustment":{"hours":[12,13],"factor":0.25}},
         {"directive_type":"no_charge_window","structured_adjustment":{"hours":[16,17]}},
         {"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[18,19,20,21,22,23],"minimum_energy_kwh":50}}],
    ),
    (
        ["Holiday notice: offices closed tomorrow, no energy impact.", "Reserve 80 kWh from 5 PM to 9 PM."],
        "no_op + reserve",
        [{"directive_type":"no_op","applies":False,"structured_adjustment":None},
         {"directive_type":"minimum_battery_reserve","structured_adjustment":{"hours":[17,18,19,20],"minimum_energy_kwh":80}}],
    ),
    (
        ["No discharge 6 PM to 8 PM.", "No charge 4 PM to 6 PM.", "Cap grid at 50 kWh 5 PM to 9 PM."],
        "no_discharge + no_charge + grid_cap",
        [{"directive_type":"no_discharge_window","structured_adjustment":{"hours":[18,19]}},
         {"directive_type":"no_charge_window","structured_adjustment":{"hours":[16,17]}},
         {"directive_type":"max_grid_window","structured_adjustment":{"hours":[17,18,19,20],"max_grid_kwh":50}}],
    ),
]


def main():
    cases = []

    # 1-8: solar_reduction
    for i, (note, hours, factor) in enumerate(solar_cases, 1):
        cases.append(case(
            f"SOLAR-{i:02d}",
            [note],
            [{"directive_type": "solar_reduction", "structured_adjustment": {"hours": hours, "factor": factor}}],
            None,
        ))

    # 9-16: window directives
    for i, (note, dtype, hours, _) in enumerate(window_cases, 9):
        cases.append(case(
            f"WINDOW-{i:02d}",
            [note],
            [{"directive_type": dtype, "structured_adjustment": {"hours": hours}}],
            None,
        ))

    # 17-24: reserves
    for i, (note, hours, kwh) in enumerate(reserve_cases, 17):
        cases.append(case(
            f"RESERVE-{i:02d}",
            [note],
            [{"directive_type": "minimum_battery_reserve", "structured_adjustment": {"hours": hours, "minimum_energy_kwh": kwh}}],
            None,
        ))

    # 25-32: grid caps
    for i, (note, hours, kwh) in enumerate(grid_cap_cases, 25):
        cases.append(case(
            f"GRID-{i:02d}",
            [note],
            [{"directive_type": "max_grid_window", "structured_adjustment": {"hours": hours, "max_grid_kwh": kwh}}],
            None,
        ))

    # 33-40: multi-note
    for i, (notes, label, expected) in enumerate(multi_cases, 33):
        cases.append(case(
            f"MULTI-{i:02d}",
            notes,
            expected,
            None,
        ))

    out_path = Path("data/synthetic/test_40_cases.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, indent=2)

    print(f"Wrote {len(cases)} cases to {out_path}")
    print(f"  - 8 solar_reduction")
    print(f"  - 8 charge/discharge windows")
    print(f"  - 8 minimum_battery_reserve")
    print(f"  - 8 max_grid_window")
    print(f"  - 8 multi-note (2-3 directives each)")


if __name__ == "__main__":
    main()
