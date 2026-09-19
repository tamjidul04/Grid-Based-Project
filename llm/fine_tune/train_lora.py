"""LoRA fine-tuning for the directive-interpretation LLM.

Trains Qwen2.5-1.5B-Instruct on the synthetic dataset at
`data/synthetic/train.jsonl` using LoRA (rank=16) on the attention +
MLP projections.

Why LoRA:
    - Trains in <30 min on a CPU laptop for 1500 examples × 1 epoch.
    - Tiny adapter files (~30 MB) — easy to commit and ship.
    - The base model stays untouched; we can swap adapters per task.

Why fp16 (not 4-bit) at 1.5B:
    - 1.5B in fp16 fits in ~3 GB RAM — well within a typical laptop budget.
    - 4-bit would slow inference and reduce fine-tuning quality.

Usage:
    python -m llm.fine_tune.train_lora
    python -m llm.fine_tune.train_lora --epochs 2 --lr 2e-4
    python -m llm.fine_tune.train_lora --data data/synthetic/train.jsonl --steps 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_chat(example: dict, tokenizer) -> dict:
    """Apply Qwen2.5 chat template and return input_ids + labels.

    We mask the system + user tokens with -100 so loss is computed only on
    the assistant turn — that's standard for instruction tuning.
    """
    messages = example["messages"]
    # Full conversation (system + user + assistant).
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    # Just the prompt (system + user) — used to find where assistant starts.
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # If the tokenizer's two passes yielded different lengths (rare), trim.
    if len(labels) > len(full_ids):
        labels = labels[: len(full_ids)]
    elif len(labels) < len(full_ids):
        labels += [-100] * (len(full_ids) - len(labels))

    return {
        "input_ids": full_ids,
        "labels": labels,
        "attention_mask": [1] * len(full_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen2.5-1.5B.")
    parser.add_argument("--model", type=str,
                        default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", type=str,
                        default="data/synthetic/train.jsonl")
    parser.add_argument("--out", type=str,
                        default="models/qwen2.5-1.5b-gridwise-lora")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--steps", type=int, default=0,
                        help="If >0, overrides --epochs and stops after this many steps.")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    here = Path(__file__).resolve().parents[2]  # gridwise-llm/

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Wrapping with LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # --- Dataset ---
    print(f"Loading dataset: {here / args.data}")
    raw = load_jsonl(here / args.data)
    print(f"  {len(raw)} examples")

    ds = Dataset.from_list(raw)
    ds = ds.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=ds.column_names,
        num_proc=1,
    )
    # Filter out examples longer than max_seq_len to keep memory bounded.
    ds = ds.filter(lambda ex: len(ex["input_ids"]) <= args.max_seq_len)
    print(f"  {len(ds)} examples after length filter (<= {args.max_seq_len} tokens)")

    # --- Training ---
    out_dir = here / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.steps if args.steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",  # save once at the end via the explicit call below
        bf16=False,          # CPU
        fp16=False,          # CPU fp16 is finicky; use fp32 weights + fp16 only for storage
        optim="adamw_torch",
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        gradient_checkpointing=False,  # CPU-only — saves memory but is slow; off by default
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {out_dir}")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Done.")


if __name__ == "__main__":
    main()
