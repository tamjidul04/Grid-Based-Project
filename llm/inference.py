"""Inference wrapper for the fine-tuned directive-interpretation LLM.

Loads the model + tokenizer once at import time (cached) and exposes a
single function `parse_directives_with_llm(inp)` that returns a list of
DirectiveInterpretation objects.

The model is loaded lazily on first use and cached as a module-level
singleton so the FastAPI server doesn't reload it on every request.

Environment variables:
    GRIDWISE_LLM_MODEL: path or HF id of the model to load
        (default: Qwen/Qwen2.5-1.5B-Instruct)
    GRIDWISE_USE_LLM:   "1" / "0" — whether to actually use the LLM
        (default: "0" — Phase A safe default)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.directives import ParseError, parse_directive_json
from llm.prompts import SYSTEM_PROMPT, build_user_prompt
from llm.schema import DirectiveInterpretation, InputSchema

logger = logging.getLogger("gridwise.inference")


def _model_path() -> str:
    return os.environ.get(
        "GRIDWISE_LLM_MODEL",
        "models/qwen2.5-1.5b-gridwise-merged",
    )


def llm_enabled() -> bool:
    return os.environ.get("GRIDWISE_USE_LLM", "0") == "1"


@lru_cache(maxsize=1)
def _load_model():
    """Load model + tokenizer once and cache the result."""
    path = _model_path()
    logger.info("Loading LLM from %s", path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _generate(messages: list[dict], max_new_tokens: int = 512) -> str:
    """Run a chat-templated generation. Greedy decoding."""
    model, tok = _load_model()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def parse_directives_with_llm(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Parse operator_notes using the LLM. Retries once on ParseError."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(inp.operator_notes)},
    ]
    raw = _generate(messages)
    try:
        return parse_directive_json(raw, expected_n=len(inp.operator_notes))
    except ParseError as e:
        logger.warning("First parse failed (%s); retrying with stricter prompt", e)
        # Retry: prepend "Return ONLY JSON. No markdown, no preamble."
        strict_messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Return ONLY the JSON array. No markdown fences, no commentary."},
        ]
        raw2 = _generate(strict_messages)
        return parse_directive_json(raw2, expected_n=len(inp.operator_notes))


def parse_directives_safely(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Try the LLM first; fall back to the rule-based parser on failure.

    This is the function the API server should call.
    """
    if llm_enabled():
        try:
            return parse_directives_with_llm(inp)
        except Exception as e:
            logger.exception("LLM parsing failed; falling back to rules: %s", e)

    from optimizer.rules_fallback import parse_notes
    return parse_notes(inp)
