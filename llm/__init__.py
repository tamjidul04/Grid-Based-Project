"""LLM module: schema, prompts, inference, fine-tuning."""

from .schema import (
    HourData,
    BatteryParams,
    InputSchema,
    DirectiveInterpretation,
    HourlyPlanRow,
    OutputSchema,
)
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .directives import ParseError, parse_directive_json
from .inference import (
    parse_directives_with_llm,
    parse_directives_safely,
    llm_enabled,
)

__all__ = [
    "HourData",
    "BatteryParams",
    "InputSchema",
    "DirectiveInterpretation",
    "HourlyPlanRow",
    "OutputSchema",
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "ParseError",
    "parse_directive_json",
    "parse_directives_with_llm",
    "parse_directives_safely",
    "llm_enabled",
]
