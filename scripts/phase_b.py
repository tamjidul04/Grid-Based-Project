"""End-to-end Phase B runner: fine-tune → merge → evaluate.

After you've downloaded the base model into
`models/Qwen2.5-1.5B-Instruct/`, this script runs:
    1. python -m llm.fine_tune.train_lora    (LoRA fine-tune, ~15-25 min CPU)
    2. python -m llm.fine_tune.merge_weights (merge adapter into base)
    3. python -m scripts.evaluate --use-llm (score against 10 sample cases)

All paths default to local; override with flags if needed.

Usage:
    python scripts/phase_b.py
    python scripts/phase_b.py --skip-train    # if you've already trained
    python scripts/phase_b.py --skip-merge    # if you've already merged
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
BASE_MODEL = HERE / "models" / "Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = HERE / "models" / "qwen2.5-1.5b-gridwise-lora"
MERGED_DIR = HERE / "models" / "qwen2.5-1.5b-gridwise-merged"


def _run(cmd: list[str], label: str) -> None:
    print("\n" + "=" * 70)
    print(f">>> {label}")
    print("    $ " + " ".join(cmd))
    print("=" * 70)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(HERE))
    elapsed = time.time() - t0
    if rc != 0:
        print(f"\n[ERROR] {label} exited with code {rc} after {elapsed:.0f}s")
        sys.exit(rc)
    print(f"\n[OK] {label} completed in {elapsed:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase B end-to-end.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=0,
                        help="If >0, train for this many steps only (smoke test).")
    args = parser.parse_args()

    # Sanity: base model must exist.
    if not (BASE_MODEL / "model.safetensors").exists():
        print(f"[ERROR] Base model not found at {BASE_MODEL}")
        print("        Run: python scripts/dl.py")
        sys.exit(1)
    safetensors_size_gb = (BASE_MODEL / "model.safetensors").stat().st_size / (1024 ** 3)
    print(f"Base model: {BASE_MODEL}  ({safetensors_size_gb:.2f} GB safetensors)")

    if not args.skip_train:
        train_cmd = [
            sys.executable, "-m", "llm.fine_tune.train_lora",
            "--model", str(BASE_MODEL),
            "--out", str(ADAPTER_DIR.relative_to(HERE)),
        ]
        if args.steps > 0:
            train_cmd += ["--steps", str(args.steps)]
        else:
            train_cmd += ["--epochs", str(args.epochs)]
        _run(train_cmd, "STEP 1/3 — LoRA fine-tuning")
    else:
        print(f"[skip-train] Using existing adapter at {ADAPTER_DIR}")

    if not args.skip_merge:
        merge_cmd = [
            sys.executable, "-m", "llm.fine_tune.merge_weights",
            "--base", str(BASE_MODEL),
            "--adapter", str(ADAPTER_DIR.relative_to(HERE)),
            "--out", str(MERGED_DIR.relative_to(HERE)),
        ]
        _run(merge_cmd, "STEP 2/3 — Merging LoRA into base model")
    else:
        print(f"[skip-merge] Using existing merged model at {MERGED_DIR}")

    if not args.skip_eval:
        eval_cmd = [
            sys.executable, "-m", "scripts.evaluate",
            "--use-llm",
        ]
        env_note = (
            f"GRIDWISE_USE_LLM=1 GRIDWISE_LLM_MODEL={MERGED_DIR}"
        )
        print(f"\n>>> STEP 3/3 — Evaluating LLM (with env: {env_note})")
        t0 = time.time()
        import os
        env = os.environ.copy()
        env["GRIDWISE_USE_LLM"] = "1"
        env["GRIDWISE_LLM_MODEL"] = str(MERGED_DIR)
        rc = subprocess.call(eval_cmd, cwd=str(HERE), env=env)
        elapsed = time.time() - t0
        if rc != 0:
            print(f"\n[ERROR] evaluate exited with code {rc} after {elapsed:.0f}s")
            sys.exit(rc)
        print(f"\n[OK] evaluate completed in {elapsed:.0f}s")

    print("\n" + "=" * 70)
    print("Phase B complete.")
    print(f"  Adapter:  {ADAPTER_DIR}")
    print(f"  Merged:   {MERGED_DIR}")
    print(f"\nTo run the API with the LLM:")
    print(f"  set GRIDWISE_USE_LLM=1")
    print(f"  set GRIDWISE_LLM_MODEL={MERGED_DIR}")
    print(f"  python -m uvicorn api.server:app --port 8000")
    print("=" * 70)


if __name__ == "__main__":
    main()
