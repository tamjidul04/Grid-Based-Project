---
title: GridWise — Microgrid Schedule Optimizer
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# GridWise — Microgrid Schedule Optimizer

A directive-aware microgrid scheduler. Free-text operator notes are
classified into typed directives, then a linear program produces an
optimal 24-hour battery / solar / grid schedule that satisfies all
hard physics and directive constraints.

This Space runs the **deterministic rules-based parser** by default
(fast, ~10 ms per request, 10/10 accuracy on the public sample
cases). Set `GRIDWISE_USE_LLM=1` in the Space settings to switch
to the Qwen2.5 LLM directive parser (requires uncommenting the ML
stack in `space/requirements.txt`).

## Endpoint

`POST /optimize-energy` — body matches `InputSchema`, response matches
`OutputSchema`. The same schema used by the public sample cases.

Example:
```bash
curl -X POST https://<your-space>.hf.space/optimize-energy \
  -H "Content-Type: application/json" \
  -d @sample01.json
```

See the project repository for the full schema, sample cases, and
local-run instructions.
