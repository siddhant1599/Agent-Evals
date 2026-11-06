"""The benchmark entrypoint: run cells and write one results/<run_id>/ per cell.

    python3 src/run_benchmark.py --topology t0_single_agent --backbone qwen2.5-7b-instruct \
        --task tier1_xsum --set run.owner=sameer --set run.seed=0

Each selector also accepts ``all``. Run from the project venv. All logic lives in
agent_evals.benchmarks.runner; this file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.benchmarks.runner import main

if __name__ == "__main__":
    src = Path(__file__).resolve().parent
    sys.exit(main(sys.argv[1:], repo_root=src.parent, configs_dir=src / "configs"))
