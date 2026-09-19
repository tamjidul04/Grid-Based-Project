"""Merge a LoRA adapter into the base model and save the merged weights.

After fine-tuning, the LoRA adapter is a small set of delta matrices. To
serve the model with a single weights file (no adapter-loading at
inference time), we merge them into the base model.

Usage:
    python -m llm.fine_tune.merge_weights
    python -m llm.fine_tune.merge_weights --adapter models/qwen2.5-1.5b-gridwise-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA into base model.")
    parser.add_argument("--base", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str,
                        default="models/qwen2.5-1.5b-gridwise-lora")
    parser.add_argument("--out", type=str,
                        default="models/qwen2.5-1.5b-gridwise-merged")
    args = parser.parse_args()

    here = Path(__file__).resolve().parents[2]

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)

    adapter_path = here / args.adapter
    print(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base, str(adapter_path))

    print("Merging...")
    model = model.merge_and_unload()

    out_path = here / args.out
    out_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {out_path}")
    model.save_pretrained(str(out_path), safe_serialization=True)
    tok.save_pretrained(str(out_path))
    print("Done.")


if __name__ == "__main__":
    main()
