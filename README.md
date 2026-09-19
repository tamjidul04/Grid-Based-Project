# GridWise LLM — BUP CSE FEST 2026 (Phase 2)

**Track 02 · Problem P-08 · Team GridMind**

A fine-tuned small LLM (Qwen2.5-1.5B with LoRA) that interprets free-text grid-operator notes into structured directives, paired with a deterministic LP-based 24-hour energy optimizer that produces an optimal battery/solar/grid schedule.

## Architecture

```
      ┌─────────────────────┐
      │  Operator Notes     │  (free text, 1–3 notes)
      │  + 24h grid data    │
      │  + battery params   │
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Fine-tuned LLM     │  Qwen2.5-1.5B + LoRA
      │  (directive parser) │  → JSON: directive_interpretation[]
      └──────────┬──────────┘
                 │     │
                 │     └──── if invalid ───► rules_fallback.py (regex)
                 ▼
      ┌─────────────────────┐
      │  PuLP Optimizer     │  Linear program
      │  (24h scheduler)    │  → minimizes total cost (BDT)
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  Validator          │  11+ physics + directive constraints
      └──────────┬──────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │  POST /optimize-    │
      │  energy response    │
      └─────────────────────┘
```

## Quick start

```bash
pip install -r requirements.txt

# Phase A: works without any LLM weights loaded (rule-based only)
python scripts/evaluate.py

# Phase B: fine-tune the LLM on CPU
python scripts/gen_synthetic_data.py
python -m llm.fine_tune.train_lora

# Run the API
uvicorn api.server:app --reload

# Run all tests
pytest tests/
```

## Team

| Member | Role |
|---|---|
| Md.Tamjidul Islam | Lead / LLM training / Data pipeline / Backend / optimizer / Frontend / visualization |

## License

MIT.
