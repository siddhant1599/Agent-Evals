"""Compare finished runs with their baseline and write the result as a run.

    python3 src/run_analysis.py --config src/configs/analysis/asqa_diagnostics.yaml \\
        --set run.owner=sameer

CPU only; it reads ``results/`` and generates nothing. All logic lives in
agent_evals.analysis.diagnostics; this file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.analysis.diagnostics import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], repo_root=Path(__file__).resolve().parents[1]))
