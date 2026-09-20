"""Time inference: how fast is generation on this CPU?"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    model_dir = here / "models" / "Qwen2.5-1.5B-Instruct"
    print(f"Loading {model_dir} ...")
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()

    prompts = [
        "Operator notes:\n1. Reserve at least 30 kWh from 6 PM onward.\n\nReturn ONLY the JSON array.",
        "Operator notes:\n1. Cap grid at 5 kWh from 5 to 9 PM.\n\nReturn ONLY the JSON array.",
    ]

    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False)
        n_in = ids["input_ids"].shape[1]
        print(f"prompt tokens: {n_in}")
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=120,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
        elapsed = time.time() - t0
        n_out = out.shape[1] - n_in
        rate = n_out / elapsed if elapsed > 0 else 0
        text = tok.decode(out[0, n_in:], skip_special_tokens=True)
        print(f"  generated {n_out} tokens in {elapsed:.1f}s = {rate:.1f} tok/s")
        print(f"  output preview: {text[:200]}")
        print()
    return 0


if __name__ == "__main__":
    main()
