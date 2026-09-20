# GridWise — Microgrid Schedule Optimizer

A directive-aware microgrid scheduler. Free-text operator notes are
classified into typed directives, then a linear program produces an
optimal 24-hour battery / solar / grid schedule that satisfies all
hard physics and directive constraints.

Built two ways:

| Mode | Parser | Backend | When to use |
|---|---|---|---|
| **Phase A** (default) | hand-written rule-based extractor (~6 enum types) | FastAPI + PuLP LP (CBC) | production: 10/10 sample cases match reference exactly |
| **Phase B** | fine-tuned Qwen2.5-1.5B-Instruct + LoRA (if weights present) | FastAPI + PuLP LP | shows the LLM-based pipeline; falls back to rules on parse failure |

Toggle phase B at runtime with `GRIDWISE_USE_LLM=1`.

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

# Run the optimizer end-to-end on the 10 public sample cases (Phase A)
python scripts/evaluate.py

# Run the API
python -m uvicorn api.server:app --port 8000 --reload
# (set GRIDWISE_USE_LLM=1 to enable LLM directive parsing if a merged
#  model exists at models/qwen2.5-1.5b-gridwise-merged/)
# (set GRIDWISE_USE_CACHE=1 to serve pre-computed demo results instantly)

# Frontend dashboard
cd frontend && python -m http.server 5500
# open http://localhost:5500

# Run all tests
pytest tests/
```

## Optional: Phase B — fine-tuned LLM directive parser

```bash
python scripts/dl.py                                # download Qwen2.5-1.5B (~3 GB)
python scripts/gen_synthetic_data.py                # 1500 synthetic training pairs
python -m llm.fine_tune.train_lora                  # LoRA fine-tune
python -m llm.fine_tune.merge_weights               # merge LoRA into base
GRIDWISE_USE_LLM=1 python -m uvicorn api.server:app --port 8000
```

> **CPU note:** LoRA fine-tuning of 1.5B-parameter models on CPU is
> very slow — measured step times were ~30 s for the first step and
> ballooned afterwards. The training script is provided as the
> documented methodology; the rules fallback already achieves 10/10
> on the public sample cases, so production runs use Phase A.

## Live demo

```bash
# Live pipeline (rules fallback, ~5-10s for 10 cases)
python scripts/demo.py

# Instant cached responses — for the polished demo
python scripts/demo.py --cache

# With the fine-tuned LLM (Phase B — if a merged model exists)
python scripts/demo.py --use-llm
```

## Author

Tamjidul Islam.

## License

MIT.
