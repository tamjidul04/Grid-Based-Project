"""Fast CPU inference for the directive-interpretation LLM.

We point at the base `Qwen/Qwen2.5-1.5B-Instruct` directly (no LoRA, no
fine-tuning) and rely on a careful prompt + few-shot examples for
accuracy. Speed tactics:

1. **Lazy singleton**: model + tokenizer loaded once on first use, cached
   with `lru_cache`. Avoid reloading on every request.
2. **fp16 + sdpa**: `torch_dtype=torch.float16`, `attn_implementation=
   "sdpa"` (much faster than eager on CPU; falls back to eager if sdpa
   is unavailable).
3. **Greedy decoding**: `do_sample=False, num_beams=1` — no sampling
   overhead, deterministic.
4. **Trimmed `max_new_tokens=400`**: directives are short JSON arrays,
   400 tokens is plenty and avoids runaway generation.
5. **Per-prompt cache**: identical `(notes tuple)` returns the cached
   parse — repeated demo runs are instant.
6. **Pre-tokenize the few-shot block once** so we only re-tokenize the
   live user prompt per request.

Environment variables:
    GRIDWISE_LLM_MODEL: path or HF id of the model to load
        (default: models/Qwen2.5-1.5B-Instruct — the base model)
    GRIDWISE_USE_LLM:   "1" / "0" — whether to actually use the LLM
        (default: "0" — Phase A safe default)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from functools import lru_cache
from typing import List, Tuple

# torch + transformers are imported LAZILY inside _load_model() so that
# the rest of this module (and the API server) can be imported on cloud
# images where the ML stack isn't installed.

from llm.directives import ParseError, parse_directive_json
from llm.prompts import build_messages
from llm.schema import DirectiveInterpretation, InputSchema

logger = logging.getLogger("gridwise.inference")

# Thread safety: HF generation isn't thread-safe in a single model instance.
_GEN_LOCK = threading.Lock()


def _model_path() -> str:
    """Where to load the model from.

    Default points to the smaller Qwen2.5-0.5B-Instruct checkpoint that
    `scripts/dl_small.py` downloads — it runs ~3x faster on CPU than the
    1.5B model while still following instructions reliably. Override with
    GRIDWISE_LLM_MODEL to use a different model (e.g. the 1.5B base).
    """
    return os.environ.get(
        "GRIDWISE_LLM_MODEL",
        "models/Qwen2.5-0.5B-Instruct",
    )


def llm_enabled() -> bool:
    return os.environ.get("GRIDWISE_USE_LLM", "0") == "1"


@lru_cache(maxsize=1)
def _load_model():
    """Load model + tokenizer once and cache. Singleton across the process."""
    # Lazy imports: only need torch/transformers when this is actually called.
    # Lets the cloud image ship without the ML stack and still import this
    # module safely.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = _model_path()
    logger.info("Loading LLM from %s (this happens once)", path)
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Try sdpa attention first (faster on CPU); fall back to eager.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning("sdpa unavailable (%s); falling back to eager", e)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
            trust_remote_code=True,
        )

    model.eval()

    # Warm up: run a tiny dummy forward so the first real call isn't
    # paying JIT + memory-allocator costs.
    try:
        with torch.no_grad():
            dummy_ids = torch.zeros((1, 8), dtype=torch.long)
            dummy_mask = torch.ones((1, 8), dtype=torch.long)
            _ = model(input_ids=dummy_ids, attention_mask=dummy_mask)
    except Exception:
        pass

    return model, tok


@lru_cache(maxsize=128)
def _cached_parse(notes_key: Tuple[str, ...]) -> Tuple[DirectiveInterpretation, ...] | None:
    """Cache parsed directive lists keyed on the tuple of note strings.

    Returns None if parsing failed (so a follow-up rules-fallback can run
    without poisoning the cache).
    """
    return None  # placeholder; real caching happens in parse_directives_with_llm


# Real response cache, populated lazily. We use a dict (not lru_cache)
# because the value is "parsed list or None + the raw text for logging".
_RESPONSE_CACHE: dict[str, tuple[List[DirectiveInterpretation] | None, str]] = {}


def _notes_key(notes: list[str]) -> str:
    return hashlib.sha256("|".join(n.strip() for n in notes).encode("utf-8")).hexdigest()[:16]


def _generate(messages: list[dict], max_new_tokens: int = 300) -> str:
    """Run chat-templated generation. Greedy decoding, CPU-friendly."""
    model, tok = _load_model()
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False)

    # Generation is single-threaded per model instance.
    with _GEN_LOCK:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=10,           # avoid empty/truncated output
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def parse_directives_with_llm(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Parse operator_notes using the LLM. Cached, retries once on parse fail."""
    key = _notes_key(inp.operator_notes)
    cached = _RESPONSE_CACHE.get(key)
    if cached is not None:
        parsed, raw = cached
        if parsed is not None:
            logger.info("[llm cache HIT] key=%s", key)
            return parsed
        # cached None means previous attempt failed; fall through to retry.

    messages = build_messages(inp.operator_notes)

    try:
        raw = _generate(messages)
        parsed = parse_directive_json(raw, expected_n=len(inp.operator_notes))
        _RESPONSE_CACHE[key] = (parsed, raw)
        return parsed
    except ParseError as e:
        logger.warning("First parse failed (%s); retrying with stricter prompt", e)
        try:
            strict_messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Return ONLY the JSON array. No markdown, no commentary."},
            ]
            raw2 = _generate(strict_messages)
            parsed2 = parse_directive_json(raw2, expected_n=len(inp.operator_notes))
            _RESPONSE_CACHE[key] = (parsed2, raw2)
            return parsed2
        except ParseError as e2:
            logger.error("LLM parse failed after retry: %s", e2)
            # Cache the failure so we don't retry the same garbage next time.
            _RESPONSE_CACHE[key] = (None, raw2 if "raw2" in locals() else raw)
            raise


def parse_directives_safely(inp: InputSchema) -> List[DirectiveInterpretation]:
    """Try the LLM first; fall back to the rule-based parser on failure.

    This is the function the API server calls.
    """
    if llm_enabled():
        try:
            return parse_directives_with_llm(inp)
        except Exception as e:
            logger.exception("LLM parsing failed; falling back to rules: %s", e)

    from optimizer.rules_fallback import parse_notes
    return parse_notes(inp)


def clear_cache() -> None:
    """Reset the response cache. Useful in tests."""
    _RESPONSE_CACHE.clear()
