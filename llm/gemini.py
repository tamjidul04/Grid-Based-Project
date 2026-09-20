"""Gemini API directive parser.

Production-grade alternative to running a local Qwen model. Uses Google's
Gemini Flash model via the official `google-genai` SDK. Free tier gives
15 RPM and ~250 ms latency per call — more than enough for a hackathon
demo, and dramatically more accurate than 0.5B/1.5B local models on
instruction-following tasks.

Set these env vars to enable:
    GRIDWISE_USE_GEMINI=1
    GEMINI_API_KEY=<your key from https://aistudio.google.com/apikey>

If the call fails (network, quota, parse error), parse_directives_safely
falls back to the rules parser.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import List

from llm.directives import ParseError, parse_directive_json
from llm.prompts import build_messages
from llm.schema import DirectiveInterpretation, InputSchema

logger = logging.getLogger("gridwise.gemini")

_GEMINI_CACHE: dict[str, tuple[List[DirectiveInterpretation] | None, str]] = {}


def gemini_enabled() -> bool:
    return os.environ.get("GRIDWISE_USE_GEMINI", "0") == "1"


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _model_name() -> str:
    """Default to Gemini 2.5 Flash — fast + smart, free tier."""
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _notes_key(notes: list[str]) -> str:
    return hashlib.sha256("|".join(n.strip() for n in notes).encode("utf-8")).hexdigest()[:16]


def _messages_to_prompt(messages: list[dict]) -> str:
    """Flatten the chat-format messages into a single Gemini prompt.

    Gemini's `generate_content` accepts text or structured prompts but
    doesn't have a chat-role concept in the same way as OpenAI/HF
    chat-template APIs. We concatenate with role tags so the model
    understands the structure.
    """
    out = []
    for m in messages:
        role = m["role"].upper()
        content = m["content"]
        out.append(f"<<<{role}>>>\n{content}")
    return "\n\n".join(out) + "\n\n<<<ASSISTANT>>>"


def _call_gemini(messages: list[dict]) -> str:
    """Make the API call. Imported lazily so the dependency is optional."""
    try:
        from google import genai  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "google-genai not installed. `pip install google-genai` to enable Gemini."
        ) from e

    key = _api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")

    client = genai.Client(api_key=key)
    prompt = _messages_to_prompt(messages)
    resp = client.models.generate_content(
        model=_model_name(),
        contents=prompt,
        config={
            "temperature": 0.0,        # greedy — deterministic for directives
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",  # hint for JSON output
        },
    )
    return (resp.text or "").strip()


def parse_directives_with_gemini(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Parse operator notes using Gemini. Cached, retries once on parse fail."""
    key = _notes_key(inp.operator_notes)
    cached = _GEMINI_CACHE.get(key)
    if cached is not None:
        parsed, raw = cached
        if parsed is not None:
            logger.info("[gemini cache HIT] key=%s", key)
            return parsed
        # cached None means previous attempt failed; fall through to retry.

    messages = build_messages(inp.operator_notes)

    try:
        raw = _call_gemini(messages)
        parsed = parse_directive_json(raw, expected_n=len(inp.operator_notes))
        _GEMINI_CACHE[key] = (parsed, raw)
        return parsed
    except ParseError as e:
        logger.warning("Gemini first parse failed (%s); retrying with stricter prompt", e)
        try:
            strict_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return ONLY the JSON array. No markdown, no commentary."},
            ]
            raw2 = _call_gemini(strict_messages)
            parsed2 = parse_directive_json(raw2, expected_n=len(inp.operator_notes))
            _GEMINI_CACHE[key] = (parsed2, raw2)
            return parsed2
        except ParseError as e2:
            logger.error("Gemini parse failed after retry: %s", e2)
            _GEMINI_CACHE[key] = (None, raw2 if "raw2" in locals() else raw)
            raise


def parse_directives_safely_gemini(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Try Gemini first; fall back to rules, then to local LLM if present."""
    if gemini_enabled():
        try:
            return parse_directives_with_gemini(inp)
        except Exception as e:
            logger.exception("Gemini parsing failed; falling back: %s", e)

    from optimizer.rules_fallback import parse_notes
    return parse_notes(inp)


def clear_cache() -> None:
    """Reset the response cache. Useful in tests."""
    _GEMINI_CACHE.clear()
