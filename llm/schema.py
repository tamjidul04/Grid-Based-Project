"""Pydantic v2 schemas that exactly mirror the official GridWise spec.

Reference: BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
    `_meta.schema_notes.{input,output}_required_fields`
    `_meta.allowed_enums`

These are the source of truth for our validator and the LLM prompt.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ----- Input -----

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]


class SolarReductionAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1, max_length=24)
    factor: float = Field(..., ge=0.0, le=1.0)


class MinimumBatteryReserveAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1, max_length=24)
    minimum_energy_kwh: float = Field(..., ge=0.0)


class NoChargeWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1, max_length=24)


class NoDischargeWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1, max_length=24)


class MaxGridWindowAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: List[int] = Field(..., min_length=1, max_length=24)
    max_grid_kwh: float = Field(..., ge=0.0)


class NoOpAdjustment(BaseModel):
    """Empty/null adjustment for no_op directives."""


StructuredAdjustment = Union[
    SolarReductionAdjustment,
    MinimumBatteryReserveAdjustment,
    NoChargeWindowAdjustment,
    NoDischargeWindowAdjustment,
    MaxGridWindowAdjustment,
    NoOpAdjustment,
    None,
]


class DirectiveInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str

    @field_validator("structured_adjustment", mode="before")
    @classmethod
    def _normalize_null(cls, v):
        # Allow null/None to pass through unchanged.
        return v


class HourData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0.0)
    solar_kwh: float = Field(..., ge=0.0)
    tariff_bdt_per_kwh: float = Field(..., ge=0.0)


class BatteryParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    capacity_kwh: float = Field(..., gt=0.0)
    initial_energy_kwh: float = Field(..., ge=0.0)
    minimum_energy_kwh: float = Field(..., ge=0.0)
    max_charge_kwh_per_hour: float = Field(..., ge=0.0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0.0)


class InputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourData] = Field(..., min_length=24, max_length=24)
    battery: BatteryParams


# ----- Output -----

BatteryAction = Literal["charge", "discharge", "idle"]


class HourlyPlanRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0.0)
    solar_used_kwh: float = Field(..., ge=0.0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0.0)
    battery_energy_after_kwh: float = Field(..., ge=0.0)


class OutputSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlanRow] = Field(..., min_length=24, max_length=24)
    total_grid_kwh: float = Field(..., ge=0.0)
    total_cost_bdt: float = Field(..., ge=0.0)
    peak_grid_kwh: float = Field(..., ge=0.0)
    plan_summary: str
