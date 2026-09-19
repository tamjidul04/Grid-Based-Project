"""Deterministic validator for GridWise outputs.

Replays every constraint listed in the official sample-cases spec
(`_meta.constraint_reminders`) against a fully-built OutputSchema instance.

This module has no external dependencies — pure Python — so it can run
in unit tests, the evaluator, and the API as a safety net.

Each check returns a (passed: bool, name: str, detail: str) tuple. The
top-level `validate_output()` returns a list of all checks plus a derived
summary.
"""

from __future__ import annotations

from typing import List, Tuple

from llm.schema import (
    BatteryParams,
    HourData,
    InputSchema,
    OutputSchema,
)

# Tolerance per spec: "Absolute differences up to 0.01 kWh or 0.01 BDT".
TOL_KWH = 0.01
TOL_BDT = 0.01


def _approx_eq(a: float, b: float, tol: float = TOL_KWH) -> bool:
    return abs(a - b) <= tol


Check = Tuple[bool, str, str]


def _check_hours_unique_24(plan: OutputSchema) -> Check:
    hours = sorted([row.hour for row in plan.hourly_plan])
    expected = list(range(24))
    if hours == expected:
        return True, "hours_0_through_23_present_and_unique", f"{len(hours)} unique hours"
    return False, "hours_0_through_23_present_and_unique", f"got {hours}"


def _check_energy_balance(
    plan: OutputSchema,
    inp: InputSchema,
    effective_solar: List[float],
) -> Check:
    demand_by_hour = {h.hour: h.demand_kwh for h in inp.hours}
    for row in plan.hourly_plan:
        charge = row.battery_kwh if row.battery_action == "charge" else 0.0
        discharge = row.battery_kwh if row.battery_action == "discharge" else 0.0
        lhs = row.grid_kwh + row.solar_used_kwh + discharge
        rhs = demand_by_hour[row.hour] + charge
        if not _approx_eq(lhs, rhs):
            return (
                False,
                "energy_balance_per_hour",
                f"hour {row.hour}: grid+solar+discharge={lhs:.3f} != demand+charge={rhs:.3f}",
            )
    return True, "energy_balance_per_hour", "all 24 hours balanced"


def _check_solar_used_within_effective(
    plan: OutputSchema,
    effective_solar: List[float],
) -> Check:
    for row, cap in zip(plan.hourly_plan, effective_solar):
        if row.solar_used_kwh > cap + TOL_KWH:
            return (
                False,
                "solar_used_kwh_within_effective",
                f"hour {row.hour}: solar_used={row.solar_used_kwh:.3f} > effective={cap:.3f}",
            )
    return True, "solar_used_kwh_within_effective", "all hours within solar cap"


def _check_battery_bounds_and_reserve(
    plan: OutputSchema,
    battery: BatteryParams,
    reserve_floors: dict[int, float],
) -> Check:
    for row in plan.hourly_plan:
        floor = max(battery.minimum_energy_kwh, reserve_floors.get(row.hour, 0.0))
        if row.battery_energy_after_kwh < floor - TOL_KWH:
            return (
                False,
                "battery_within_bounds_and_reserve",
                f"hour {row.hour}: battery_after={row.battery_energy_after_kwh:.3f} < floor={floor:.3f}",
            )
        if row.battery_energy_after_kwh > battery.capacity_kwh + TOL_KWH:
            return (
                False,
                "battery_within_bounds_and_reserve",
                f"hour {row.hour}: battery_after={row.battery_energy_after_kwh:.3f} > cap={battery.capacity_kwh:.3f}",
            )
    return True, "battery_within_bounds_and_reserve", "all hours within bounds"


def _check_action_rates(
    plan: OutputSchema,
    battery: BatteryParams,
    no_charge_hours: set[int],
    no_discharge_hours: set[int],
) -> Check:
    for row in plan.hourly_plan:
        if row.battery_action == "idle":
            if not _approx_eq(row.battery_kwh, 0.0):
                return (
                    False,
                    "battery_action_rates",
                    f"hour {row.hour}: idle but battery_kwh={row.battery_kwh}",
                )
        elif row.battery_action == "charge":
            if row.battery_kwh > battery.max_charge_kwh_per_hour + TOL_KWH:
                return (
                    False,
                    "battery_action_rates",
                    f"hour {row.hour}: charge {row.battery_kwh} > max_charge {battery.max_charge_kwh_per_hour}",
                )
            if row.hour in no_charge_hours and row.battery_kwh > TOL_KWH:
                return (
                    False,
                    "battery_action_rates",
                    f"hour {row.hour}: charging during no_charge_window",
                )
        elif row.battery_action == "discharge":
            if row.battery_kwh > battery.max_discharge_kwh_per_hour + TOL_KWH:
                return (
                    False,
                    "battery_action_rates",
                    f"hour {row.hour}: discharge {row.battery_kwh} > max_discharge {battery.max_discharge_kwh_per_hour}",
                )
            if row.hour in no_discharge_hours and row.battery_kwh > TOL_KWH:
                return (
                    False,
                    "battery_action_rates",
                    f"hour {row.hour}: discharging during no_discharge_window",
                )
    return True, "battery_action_rates", "all action rates honored"


def _check_max_grid_window(
    plan: OutputSchema,
    grid_caps: dict[int, float],
) -> Check:
    for row in plan.hourly_plan:
        if row.hour in grid_caps and row.grid_kwh > grid_caps[row.hour] + TOL_KWH:
            return (
                False,
                "max_grid_window",
                f"hour {row.hour}: grid_kwh={row.grid_kwh:.3f} > cap={grid_caps[row.hour]:.3f}",
            )
    return True, "max_grid_window", "all hours within grid caps"


def _check_end_of_day_neutrality(
    plan: OutputSchema,
    battery: BatteryParams,
) -> Check:
    last = plan.hourly_plan[-1].battery_energy_after_kwh
    if not _approx_eq(last, battery.initial_energy_kwh):
        return (
            False,
            "end_of_day_neutrality",
            f"final battery={last:.3f} != initial={battery.initial_energy_kwh:.3f}",
        )
    return True, "end_of_day_neutrality", f"final battery == initial ({battery.initial_energy_kwh} kWh)"


def _check_totals_recomputed(
    plan: OutputSchema,
    inp: InputSchema,
) -> Check:
    recomputed_total_grid = sum(r.grid_kwh for r in plan.hourly_plan)
    recomputed_peak_grid = max(r.grid_kwh for r in plan.hourly_plan)
    tariff_by_hour = {h.hour: h.tariff_bdt_per_kwh for h in inp.hours}
    recomputed_total_cost = sum(
        r.grid_kwh * tariff_by_hour[r.hour] for r in plan.hourly_plan
    )

    if not _approx_eq(plan.total_grid_kwh, recomputed_total_grid, TOL_KWH):
        return (
            False,
            "totals_consistent_with_plan",
            f"total_grid_kwh {plan.total_grid_kwh} != recomputed {recomputed_total_grid}",
        )
    if not _approx_eq(plan.peak_grid_kwh, recomputed_peak_grid, TOL_KWH):
        return (
            False,
            "totals_consistent_with_plan",
            f"peak_grid_kwh {plan.peak_grid_kwh} != recomputed {recomputed_peak_grid}",
        )
    if not _approx_eq(plan.total_cost_bdt, recomputed_total_cost, TOL_BDT):
        return (
            False,
            "totals_consistent_with_plan",
            f"total_cost_bdt {plan.total_cost_bdt} != recomputed {recomputed_total_cost}",
        )
    return (
        True,
        "totals_consistent_with_plan",
        f"total={plan.total_grid_kwh:.2f}, peak={plan.peak_grid_kwh:.2f}, cost={plan.total_cost_bdt:.2f}",
    )


def _check_directive_structure(
    plan: OutputSchema,
) -> Check:
    if not plan.directive_interpretation:
        return False, "directive_interpretation_nonempty", "empty list"
    return True, "directive_interpretation_nonempty", f"{len(plan.directive_interpretation)} entries"


def _check_directive_hours_valid(plan: OutputSchema) -> Check:
    for d in plan.directive_interpretation:
        if d.directive_type == "no_op":
            continue
        adj = d.structured_adjustment
        if adj is None:
            return (
                False,
                "directive_hours_valid",
                f"note {d.note_index} ({d.directive_type}) has null adjustment",
            )
        hours = getattr(adj, "hours", None)
        if hours is None:
            return (
                False,
                "directive_hours_valid",
                f"note {d.note_index} ({d.directive_type}) has no hours field",
            )
        if sorted(hours) != sorted(set(hours)):
            return (
                False,
                "directive_hours_valid",
                f"note {d.note_index}: duplicate hours in {hours}",
            )
        for h in hours:
            if not (0 <= h <= 23):
                return (
                    False,
                    "directive_hours_valid",
                    f"note {d.note_index}: hour {h} out of [0,23]",
                )
    return True, "directive_hours_valid", "all directive hours in [0,23] and unique"


def compute_effective_solar(inp: InputSchema, plan: OutputSchema) -> List[float]:
    """Apply any solar_reduction directives to the input solar profile.

    Returns a 24-length list indexed by hour.
    """
    base = [h.solar_kwh for h in inp.hours]
    for d in plan.directive_interpretation:
        if d.directive_type == "solar_reduction" and d.applies and d.structured_adjustment:
            factor = d.structured_adjustment.factor
            for h in d.structured_adjustment.hours:
                base[h] *= factor
    return base


def compute_directive_effects(
    inp: InputSchema, plan: OutputSchema
) -> tuple[dict[int, float], set[int], set[int], dict[int, float]]:
    """Pull out all directive-driven effects: reserve floors, no-charge/discharge
    hours, and grid caps.

    Returns:
        reserve_floors:  {hour: minimum_energy_kwh during that hour}
        no_charge_hours: set of hours where charging is forbidden
        no_discharge_hours: set of hours where discharging is forbidden
        grid_caps:       {hour: max_grid_kwh during that hour}
    """
    reserve_floors: dict[int, float] = {}
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    grid_caps: dict[int, float] = {}

    for d in plan.directive_interpretation:
        if not d.applies or d.directive_type == "no_op" or d.structured_adjustment is None:
            continue
        adj = d.structured_adjustment
        if d.directive_type == "minimum_battery_reserve":
            for h in adj.hours:
                reserve_floors[h] = max(
                    reserve_floors.get(h, 0.0), adj.minimum_energy_kwh
                )
        elif d.directive_type == "no_charge_window":
            no_charge_hours.update(adj.hours)
        elif d.directive_type == "no_discharge_window":
            no_discharge_hours.update(adj.hours)
        elif d.directive_type == "max_grid_window":
            for h in adj.hours:
                grid_caps[h] = adj.max_grid_kwh

    return reserve_floors, no_charge_hours, no_discharge_hours, grid_caps


def validate_output(
    inp: InputSchema,
    plan: OutputSchema,
) -> tuple[bool, list[Check]]:
    """Run all constraint checks. Returns (all_passed, [(passed, name, detail)])."""
    effective_solar = compute_effective_solar(inp, plan)
    reserve_floors, no_charge, no_discharge, grid_caps = compute_directive_effects(inp, plan)

    checks: list[Check] = [
        _check_hours_unique_24(plan),
        _check_directive_structure(plan),
        _check_directive_hours_valid(plan),
        _check_energy_balance(plan, inp, effective_solar),
        _check_solar_used_within_effective(plan, effective_solar),
        _check_battery_bounds_and_reserve(plan, inp.battery, reserve_floors),
        _check_action_rates(plan, inp.battery, no_charge, no_discharge),
        _check_max_grid_window(plan, grid_caps),
        _check_end_of_day_neutrality(plan, inp.battery),
        _check_totals_recomputed(plan, inp),
    ]
    all_passed = all(c[0] for c in checks)
    return all_passed, checks


def directive_matches(
    actual: OutputSchema,
    expected: OutputSchema,
) -> tuple[bool, str]:
    """Compare directive interpretations between two outputs.

    For each note, compare (applies, directive_type, structured_adjustment
    fields). The plan_summary text is NOT compared (per spec: "Free-text
    explanation wording does not need to match byte-for-byte").

    Note: Pydantic Union parsing can map `hours`-only adjustments to any of
    the window/no-op subclasses since they share fields. We compare by the
    `directive_type` discriminator + the field values, not by class identity.
    """
    a = {d.note_index: d for d in actual.directive_interpretation}
    e = {d.note_index: d for d in expected.directive_interpretation}
    if set(a.keys()) != set(e.keys()):
        return False, f"note indices differ: actual={sorted(a.keys())} expected={sorted(e.keys())}"

    for idx in sorted(a.keys()):
        ad, ed = a[idx], e[idx]
        if ad.directive_type != ed.directive_type:
            return False, f"note {idx}: directive_type {ad.directive_type} != {ed.directive_type}"
        if ad.applies != ed.applies:
            return False, f"note {idx}: applies {ad.applies} != {ed.applies}"
        if (ad.structured_adjustment is None) != (ed.structured_adjustment is None):
            return False, f"note {idx}: one adjustment is null"
        if ad.structured_adjustment is None:
            continue
        # Compare by fields, not by class (Pydantic Union can pick any subclass).
        a_hours = getattr(ad.structured_adjustment, "hours", None)
        e_hours = getattr(ed.structured_adjustment, "hours", None)
        if sorted(a_hours or []) != sorted(e_hours or []):
            return False, f"note {idx}: hours differ {a_hours} vs {e_hours}"
        a_factor = getattr(ad.structured_adjustment, "factor", None)
        e_factor = getattr(ed.structured_adjustment, "factor", None)
        if a_factor is not None or e_factor is not None:
            if not _approx_eq(a_factor or 0.0, e_factor or 0.0):
                return False, f"note {idx}: factor differs {a_factor} vs {e_factor}"
        a_min = getattr(ad.structured_adjustment, "minimum_energy_kwh", None)
        e_min = getattr(ed.structured_adjustment, "minimum_energy_kwh", None)
        if a_min is not None or e_min is not None:
            if not _approx_eq(a_min or 0.0, e_min or 0.0):
                return False, f"note {idx}: minimum_energy_kwh differs {a_min} vs {e_min}"
        a_cap = getattr(ad.structured_adjustment, "max_grid_kwh", None)
        e_cap = getattr(ed.structured_adjustment, "max_grid_kwh", None)
        if a_cap is not None or e_cap is not None:
            if not _approx_eq(a_cap or 0.0, e_cap or 0.0):
                return False, f"note {idx}: max_grid_kwh differs {a_cap} vs {e_cap}"

    return True, "directives match"
