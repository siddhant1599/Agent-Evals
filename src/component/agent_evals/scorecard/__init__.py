"""The agentic eval scorecard: score every agent configuration in ``results/`` on outcome,
reliability, trajectory, efficiency and value over the single-agent baseline.

Entry point: ``python3 src/run_evals.py`` (see :mod:`agent_evals.scorecard.report`).
"""

from __future__ import annotations

from .collect import Example, RunRecord, canonical_runs, read_examples, scan_results
from .compute import AGENT_PROFILES, TASKS, AgentProfile, ScorecardConfig, TaskSpec, build_scorecard
from .report import main, write_outputs

__all__ = [
    "AGENT_PROFILES",
    "TASKS",
    "AgentProfile",
    "Example",
    "RunRecord",
    "ScorecardConfig",
    "TaskSpec",
    "build_scorecard",
    "canonical_runs",
    "main",
    "read_examples",
    "scan_results",
    "write_outputs",
]
