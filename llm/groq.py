"""Groq API directive parser.

Production LLM path. Uses Groq's hosted inference of open-source LLMs
(LLaMA 3.1 8B Instant, Mixtral, etc.) via the official Groq Python SDK.

Why Groq:
    - Free tier with very generous limits (~30 RPM, no daily cap).
    - Sub-second latency (Groq's LPU hardware is fast).
    - LLaMA 3.1 8B Instant is dramatically better at instruction
      following than 0.5B Qwen; reliable JSON output.
    - OpenAI-compatible chat completions API — clean SDK.

Set these env vars to enable:
    GRIDWISE_USE_GROQ=1
    GROQ_API_KEY=<your key from https://console.groq.com/keys>

If the call fails (network, quota, parse error), parse_directives_safely_groq
falls back to the rules parser, then the local Qwen LLM (if installed).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from llm.directives import ParseError, parse_directive_json
from llm.prompts import build_messages
from llm.schema import DirectiveInterpretation, InputSchema

logger = logging.getLogger("gridwise.groq")

_GROQ_CACHE: dict[str, tuple[List[DirectiveInterpretation] | None, str]] = {}


def groq_enabled() -> bool:
    return os.environ.get("GRIDWISE_USE_GROQ", "0") == "1"


def _api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def _model_name() -> str:
    """Default to GPT-OSS 20B — strong at JSON, free tier, fast on Groq LPU."""
    return os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")


def _notes_key(notes: list[str]) -> str:
    return hashlib.sha256("|".join(n.strip() for n in notes).encode("utf-8")).hexdigest()[:16]


def _call_groq(messages: list[dict]) -> str:
    """Make the API call. Imported lazily so the dependency is optional."""
    try:
        from groq import Groq  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "groq SDK not installed. `pip install groq` to enable Groq inference."
        ) from e

    key = _api_key()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")

    client = Groq(api_key=key)
    # OpenAI-compatible chat completions with JSON mode hint.
    chat_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
    resp = client.chat.completions.create(
        model=_model_name(),
        messages=chat_messages,
        temperature=0.0,
        max_tokens=4096,
        response_format={"type": "json_object"},  # hint, not strict
    )
    return (resp.choices[0].message.content or "").strip()


def parse_directives_with_groq(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Parse operator notes using Groq. Cached, retries once on parse fail."""
    key = _notes_key(inp.operator_notes)
    cached = _GROQ_CACHE.get(key)
    if cached is not None:
        parsed, raw = cached
        if parsed is not None:
            logger.info("[groq cache HIT] key=%s", key)
            return parsed

    messages = build_messages(inp.operator_notes)

    try:
        raw = _call_groq(messages)
        parsed = parse_directive_json(raw, expected_n=len(inp.operator_notes))
        _GROQ_CACHE[key] = (parsed, raw)
        return parsed
    except ParseError as e:
        logger.warning("Groq first parse failed (%s); retrying with stricter prompt", e)
        try:
            strict_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return ONLY the JSON array. No markdown, no commentary."},
            ]
            raw2 = _call_groq(strict_messages)
            parsed2 = parse_directive_json(raw2, expected_n=len(inp.operator_notes))
            _GROQ_CACHE[key] = (parsed2, raw2)
            return parsed2
        except ParseError as e2:
            logger.error("Groq parse failed after retry: %s", e2)
            _GROQ_CACHE[key] = (None, raw2 if "raw2" in locals() else raw)
            raise


def parse_directives_safely_groq(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Try Groq first; fall back to rules, then local LLM."""
    if groq_enabled():
        try:
            return parse_directives_with_groq(inp)
        except Exception as e:
            logger.exception("Groq parsing failed; falling back: %s", e)

    from optimizer.rules_fallback import parse_notes
    return parse_notes(inp)


def clear_cache() -> None:
    """Reset the response cache."""
    _GROQ_CACHE.clear()
