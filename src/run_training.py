"""The T5 training entrypoint: one GRPO run, written to results/<run_id>/.

    python3 src/run_training.py --config src/configs/rl/grpo_xsum.yaml \
        --backbone qwen2.5-7b-instruct --set run.owner=sameer --set run.seed=0

An interrupted run continues from its latest checkpoint, with its own config (D-059):

    python3 src/run_training.py --resume results/<run_id>

Run from the project venv on a GPU instance. All logic lives in agent_evals.rl.trainer; this
file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.rl.trainer import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], repo_root=Path(__file__).resolve().parents[1]))
