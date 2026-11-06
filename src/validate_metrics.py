"""Reproduce the published validation number of each metric port.

    python3 src/validate_metrics.py
    python3 src/validate_metrics.py --metric summac --metric minicheck

Run from the project venv on a CUDA machine. All logic lives in agent_evals.metrics.reproduce;
this file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.metrics.reproduce import main

if __name__ == "__main__":
    src = Path(__file__).resolve().parent
    sys.exit(main(sys.argv[1:], repo_root=src.parent, default_config=src / "configs" / "base.yaml"))
