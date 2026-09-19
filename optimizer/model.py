"""24-hour energy schedule optimizer.

A linear program (PuLP / CBC solver) that, given an InputSchema + a list of
DirectiveInterpretation objects, produces a 24-hour hourly_plan that:

  * satisfies all physics constraints (energy balance, battery dynamics)
  * respects all directive-driven constraints (solar caps, battery reserve
    floors, no-charge/no-discharge windows, max-grid caps)
  * minimizes total electricity cost in BDT
  * ends the day at the same battery level it started (end-of-day neutrality)

This is the deterministic safety net behind the LLM. Even if the LLM emits
imperfect directives, the optimizer will produce a *valid* (constraint-
satisfying) plan. If directives conflict with the input, the LP may be
infeasible — in that case we return a partial plan and a clear error.

The model uses continuous variables only (no binaries) by treating each
hour's battery action as a *signed* net flow:

    net[h] = charge[h] - discharge[h]

If net[h] >= 0, the battery is charging (cost incurred to add to grid draw).
If net[h] <= 0, the battery is discharging (offsetting grid draw).

The signed-flow formulation is exact here because there is no efficiency
loss or cost asymmetry between charge and discharge in the spec. PuLP
infers the integer battery_action label from the sign of net and the value
of battery_kwh in the output.
"""

from __future__ import annotations

from typing import List

import pulp

from llm.schema import (
    HourlyPlanRow,
    InputSchema,
    OutputSchema,
)
from .validate import compute_directive_effects, compute_effective_solar  # noqa: F401  (re-exported for back-compat)


HOURS = list(range(24))


class InfeasibleScheduleError(Exception):
    """Raised when the LP has no feasible solution (over-constrained inputs)."""


def _compute_effective_solar_from_directives(inp, directives):
    """Apply any solar_reduction directives to the input solar profile."""
    base = [h.solar_kwh for h in inp.hours]
    for d in directives:
        if d.directive_type == "solar_reduction" and d.applies and d.structured_adjustment:
            factor = d.structured_adjustment.factor
            for h in d.structured_adjustment.hours:
                base[h] *= factor
    return base


def _compute_directive_effects_from_directives(directives):
    """Pull reserve floors, no-charge/discharge hours, and grid caps from a raw directives list."""
    reserve_floors: dict[int, float] = {}
    no_charge_hours: set[int] = set()
    no_discharge_hours: set[int] = set()
    grid_caps: dict[int, float] = {}

    for d in directives:
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


def _battery_action_label(net_value: float, battery_kwh: float) -> str:
    """Map a signed net flow to (charge / discharge / idle) and the kWh amount."""
    eps = 1e-6
    if abs(net_value) < eps or abs(battery_kwh) < eps:
        return "idle", 0.0
    return ("charge" if net_value > 0 else "discharge"), abs(battery_kwh)


def optimize_schedule(
    inp: InputSchema,
    directives: list,
    plan_summary: str = "",
) -> OutputSchema:
    """Solve the 24-hour energy LP.

    Args:
        inp:         InputSchema with 24 hours + battery params.
        directives:  List of DirectiveInterpretation objects (already validated).
        plan_summary: Free-text summary to attach to the output.

    Returns:
        OutputSchema with a constraint-satisfying hourly_plan and computed totals.

    Raises:
        InfeasibleScheduleError: if the LP is infeasible.
    """
    # Compute directive effects directly from the directives list (the
    # validate.py helpers need a full OutputSchema, but we don't need a
    # full plan yet — we're about to build one).
    effective_solar = _compute_effective_solar_from_directives(inp, directives)
    reserve_floors, no_charge, no_discharge, grid_caps = _compute_directive_effects_from_directives(
        directives
    )

    demand = {h.hour: h.demand_kwh for h in inp.hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in inp.hours}
    b = inp.battery

    # ---- LP ----
    prob = pulp.LpProblem("gridwise_24h", pulp.LpMinimize)

    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in HOURS}
    solar = {h: pulp.LpVariable(f"solar_{h}", lowBound=0) for h in HOURS}
    net = {h: pulp.LpVariable(f"net_{h}", lowBound=-b.max_discharge_kwh_per_hour,
                              upBound=b.max_charge_kwh_per_hour) for h in HOURS}
    bat_after = {h: pulp.LpVariable(f"bat_after_{h}", lowBound=0) for h in HOURS}

    # Objective: minimize total electricity cost.
    prob += pulp.lpSum(grid[h] * tariff[h] for h in HOURS)

    # Energy balance: grid + solar - net = demand  (net>0 charges from grid)
    for h in HOURS:
        prob += grid[h] + solar[h] - net[h] == demand[h], f"balance_{h}"

    # Solar cap (after directives).
    for h in HOURS:
        prob += solar[h] <= effective_solar[h], f"solar_cap_{h}"

    # Battery dynamics and bounds.
    prob += bat_after[0] == b.initial_energy_kwh + net[0], "bat_init"
    for h in HOURS[1:]:
        prob += bat_after[h] == bat_after[h - 1] + net[h], f"bat_dyn_{h}"

    for h in HOURS:
        # Reserve floor (combines base minimum and any minimum_battery_reserve).
        floor = max(b.minimum_energy_kwh, reserve_floors.get(h, 0.0))
        prob += bat_after[h] >= floor, f"bat_floor_{h}"
        prob += bat_after[h] <= b.capacity_kwh, f"bat_cap_{h}"

    # Directive windows.
    for h in no_charge:
        prob += net[h] <= 0, f"no_charge_{h}"
        prob += net[h] >= -b.max_discharge_kwh_per_hour, f"no_charge_below_{h}"
    for h in no_discharge:
        prob += net[h] >= 0, f"no_discharge_{h}"
        prob += net[h] <= b.max_charge_kwh_per_hour, f"no_discharge_above_{h}"
    for h, cap in grid_caps.items():
        prob += grid[h] <= cap, f"grid_cap_{h}"

    # End-of-day neutrality.
    prob += bat_after[23] == b.initial_energy_kwh, "end_neutral"

    # ---- Solve ----
    # Silence PuLP's chatter; raise on infeasibility.
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        raise InfeasibleScheduleError(
            f"LP status: {pulp.LpStatus[status]}. "
            "Check that directives + battery constraints are mutually satisfiable."
        )

    # ---- Extract solution ----
    hourly_plan: List[HourlyPlanRow] = []
    for h in HOURS:
        net_val = pulp.value(net[h])
        bat_kwh = abs(net_val)
        action, kwh = _battery_action_label(net_val, bat_kwh)
        hourly_plan.append(
            HourlyPlanRow(
                hour=h,
                grid_kwh=round(pulp.value(grid[h]), 4),
                solar_used_kwh=round(pulp.value(solar[h]), 4),
                battery_action=action,
                battery_kwh=round(kwh, 4),
                battery_energy_after_kwh=round(pulp.value(bat_after[h]), 4),
            )
        )

    total_grid = sum(r.grid_kwh for r in hourly_plan)
    peak_grid = max(r.grid_kwh for r in hourly_plan)
    total_cost = sum(r.grid_kwh * tariff[r.hour] for r in hourly_plan)

    return OutputSchema(
        scenario_id=inp.scenario_id,
        directive_interpretation=list(directives),
        hourly_plan=hourly_plan,
        total_grid_kwh=round(total_grid, 4),
        total_cost_bdt=round(total_cost, 4),
        peak_grid_kwh=round(peak_grid, 4),
        plan_summary=plan_summary or "Optimal 24-hour schedule minimizing total electricity cost.",
    )
