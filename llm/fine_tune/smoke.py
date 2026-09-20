"""Smoke test for LoRA fine-tuning on this hardware.

Times a single forward+backward pass to predict whether full training
is feasible. If a single step takes >60s, training is too slow to
realistically complete and we should rely on the rules fallback.

Usage:
    python -m llm.fine_tune.smoke
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    model_dir = here / "models" / "Qwen2.5-1.5B-Instruct"
    print(f"Loading model from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float32,  # CPU-friendly: avoid fp16 matmul overhead on small batches
        device_map="cpu",
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],  # tiny: only 2 projections
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.train()

    # Build a tiny synthetic batch.
    text = "The quick brown fox jumps over the lazy dog. " * 30
    enc = tokenizer([text, text], return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc["labels"] = enc["input_ids"].clone()
    print(f"input shape: {tuple(enc['input_ids'].shape)}")

    print("Warm-up pass (no grad)...")
    t0 = time.time()
    with torch.no_grad():
        _ = model(**enc)
    print(f"  warm-up forward: {time.time() - t0:.1f}s")

    # 2 timed steps with backward.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    times = []
    for i in range(2):
        t0 = time.time()
        out = model(**enc)
        loss = out.loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  step {i + 1}: {elapsed:.1f}s   loss={loss.item():.3f}")

    avg = sum(times) / len(times)
    print()
    print(f"Average step time (bs=2, seq=256, lora r=8, 2 targets): {avg:.1f}s")
    if avg > 60:
        print("VERDICT: too slow for full LoRA training on this hardware.")
        print("         Recommend relying on rules fallback + base model.")
    elif avg > 30:
        print("VERDICT: training feasible but slow. ~25-40 min for 30 steps.")
    else:
        print("VERDICT: training feasible. ~10-20 min for 30 steps.")
    return 0


if __name__ == "__main__":
    main()
