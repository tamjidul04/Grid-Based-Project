"""HuggingFace Spaces entrypoint.

HF Spaces looks for `app.py` at the repo root and runs it. For Gradio
Spaces, `app` should be a Gradio interface; for Docker Spaces, anything
goes. We're a Docker Space running uvicorn, but HF still requires
this file to exist for the Space metadata to validate.

This file is also where you can opt into the LLM parser:
    - Set GRIDWISE_USE_LLM=1 in the Space env vars to use the
      Qwen2.5-0.5B/1.5B parser (slower, requires the ML stack
      in requirements.txt).

When `app.py` is the entrypoint, uvicorn should still be invoked via
the CMD in the Dockerfile. This file just exposes the FastAPI app
under the name `app` so HF's runtime introspection works.
"""
import os
import sys

# Ensure the Space container can import the project's modules.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Toggle LLM at runtime via env var; default OFF on Spaces for speed.
os.environ.setdefault("GRIDWISE_USE_LLM", "0")

from api.server import app as application  # noqa: E402

# HF Spaces expects a top-level `app` symbol.
app = application
