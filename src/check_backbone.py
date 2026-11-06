"""Check that one backbone serves on this machine: start vLLM, send one request, stop.

    python3 src/check_backbone.py --backbone qwen2.5-7b-instruct

Backbone names come from src/model_registry.csv. Run from the project venv. All logic lives in
agent_evals.serving.check; this file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.serving.check import main

if __name__ == "__main__":
    src = Path(__file__).resolve().parent
    sys.exit(main(sys.argv[1:], repo_root=src.parent, default_config=src / "configs" / "base.yaml"))
