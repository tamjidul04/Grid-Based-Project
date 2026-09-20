"""Direct test: build a real chat-template prompt and time a single generation."""
from __future__ import annotations
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.prompts import build_messages


def main() -> int:
    here = Path(__file__).resolve().parents[2]
    model_dir = here / "models" / "Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    notes = ["Reserve at least 30 kWh from 6 PM onward."]
    messages = build_messages(notes)
    print(f"messages: {len(messages)} turns")

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
    n_in = ids["input_ids"].shape[1]
    print(f"prompt tokens: {n_in}")
    print(f"prompt preview: {prompt[:200]}...")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="cpu",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    print("Loaded. Generating...")

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=200,
            do_sample=False,
            num_beams=1,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
            use_cache=True,
        )
    elapsed = time.time() - t0
    n_out = out.shape[1] - n_in
    print(f"\ngenerated {n_out} tokens in {elapsed:.1f}s = {n_out/elapsed:.2f} tok/s")
    print("\n--- generated text ---")
    text = tok.decode(out[0, n_in:], skip_special_tokens=True)
    print(text)
    print("--- end ---")
    return 0


if __name__ == "__main__":
    main()
