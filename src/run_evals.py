"""Build the agentic eval scorecard from every run in results/.

    python3 src/run_evals.py                  # writes evals/SCORECARD.md, leaderboard.csv,
                                              # scorecard.json and evals/fig/*.svg
    python3 src/run_evals.py --pass-threshold 0.7 --no-figures

CPU only; it reads results/ and generates nothing. All logic lives in agent_evals.scorecard;
this file only supplies the repository location.
"""

import sys
from pathlib import Path

from agent_evals.scorecard import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], repo_root=Path(__file__).resolve().parents[1]))
