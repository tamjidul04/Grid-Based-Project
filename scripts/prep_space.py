"""Prepare a HuggingFace Space directory from the main project.

Reads .gitignored / heavyweight paths and copies only what the Space
needs:

- api/                  (the FastAPI app)
- llm/                  (schema + rules fallback; the LLM code stays
                         dormant without torch/transformers installed)
- optimizer/            (PuLP LP + validator)
- data/raw/sample_cases.json
- space/Dockerfile       → ./Dockerfile
- space/app.py           → ./app.py
- space/requirements.txt → ./requirements.txt  (overwrites default)
- space/README.md        → ./README.md  (HF metadata frontmatter wins)

After this runs, the current directory is a valid HF Docker Space.

Usage:
    python scripts/prep_space.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SPACE = HERE / "space"


def main() -> int:
    if not SPACE.exists():
        raise SystemExit(f"Space source dir not found: {SPACE}")

    # Copy subdirs
    for sub in ("api", "llm", "optimizer"):
        src = HERE / sub
        dst = SPACE / sub
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  copied {sub}/")

    # Copy sample cases (small, <100 KB)
    raw = HERE / "data" / "raw"
    sample_dst = SPACE / "data" / "raw"
    sample_dst.mkdir(parents=True, exist_ok=True)
    cases = raw / "sample_cases.json"
    if cases.exists():
        shutil.copy2(cases, sample_dst / "sample_cases.json")
        print(f"  copied data/raw/sample_cases.json")

    # Promote the deployment files to repo root
    for fn in ("Dockerfile", "app.py", "requirements.txt", "README.md"):
        src = SPACE / fn
        dst = SPACE / fn
        if not src.exists():
            raise SystemExit(f"Missing deployment file: {src}")
        print(f"  found {fn}")

    print()
    print(f"Space ready at: {SPACE}")
    print()
    print("Next steps:")
    print("  1. Create the Space (one-time):")
    print("     huggingface-cli login")
    print("     huggingface-cli repo create <your-username>/gridwise --type space --space-type docker")
    print("  2. Push the space/ contents:")
    print("     cd space")
    print("     git init && git add . && git commit -m 'Deploy GridWise'")
    print("     git remote add space https://huggingface.co/spaces/<your-username>/gridwise")
    print("     git push space main")
    print("  3. Your public URL will be:")
    print("     https://<your-username>-gridwise.hf.space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
