"""Read every run under ``results/`` and sort it into agent-evaluation runs and training runs.

An *agent-evaluation run* is one cell of the benchmark: an agent architecture (topology) run on
a backbone and a task, under one seed, written by ``run_benchmark.py``. Its manifest carries a
``cell``. A *training run* is a GRPO fine-tune written by ``run_training.py``; it carries a
``task`` and a ``backbone`` but no ``cell``. Analysis runs (``run_analysis.py``) carry neither and
are ignored here.

Every attempt is kept, including failed and out-of-memory ones, because how often an agent
configuration fails to complete is itself an evaluation result (see ``reliability`` in
:mod:`agent_evals.scorecard.compute`). Scores are only ever read from the *canonical* run of a
cell and seed: the latest successful attempt, so a re-run that supersedes an earlier one (for
example after a prompt fix logged in ``src/docs/decisions.md``) is what gets scored.

This module reads files only; it never generates or scores text.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Example",
    "RunRecord",
    "canonical_runs",
    "read_examples",
    "scan_results",
]

#: ``summary.json`` statuses that count as a finished run.
SUCCESS = "success"

#: Per-example columns in ``metrics.csv`` that are trajectory or cost fields, not quality scores.
TRAJECTORY_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "wall_clock_s",
    "n_rounds",
    "n_messages",
)
#: Per-example boolean columns.
FLAG_FIELDS = ("converged", "hit_length_cap", "blank_output")
#: Per-example columns that are neither scores nor trajectory fields.
NON_SCORE_FIELDS = frozenset(
    {"run_id", "output", "example_id", "source_words", "finish_reason"}
    | set(TRAJECTORY_FIELDS)
    | set(FLAG_FIELDS)
)


@dataclass(frozen=True)
class RunRecord:
    """One attempt recorded under ``results/<run_id>/``.

    Attributes:
        run_id: The folder name.
        run_dir: Path to the folder.
        kind: ``"agent_eval"`` or ``"training"``.
        status: ``summary.json`` status (``success``, ``failed``, ``oom``, ...).
        topology: Agent architecture; ``None`` for training runs.
        task: Task name, e.g. ``tier1_xsum``.
        backbone: Backbone name.
        seed: The run's seed.
        started_at: ISO timestamp, used to order attempts.
        clean_tree: Whether the run was launched from a clean git tree.
        summary: The parsed ``summary.json``.
        adapter_run_id: For an RL-trained agent, the training run whose adapter it served.
        config_name: The config the run was launched from.
    """

    run_id: str
    run_dir: Path
    kind: str
    status: str
    topology: str | None
    task: str | None
    backbone: str | None
    seed: int | None
    started_at: str
    clean_tree: bool
    summary: dict[str, Any] = field(repr=False)
    adapter_run_id: str | None = None
    config_name: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == SUCCESS

    @property
    def cell(self) -> tuple[str, str, str]:
        """``(task, topology, backbone)``; only meaningful for agent-evaluation runs."""
        return (self.task or "", self.topology or "", self.backbone or "")


@dataclass(frozen=True)
class Example:
    """One task instance as one agent run handled it.

    Attributes:
        example_id: The dataset item id.
        scores: Metric key -> score, for every quality metric the run scored.
        trajectory: Trajectory and cost fields (tokens, wall clock, rounds, messages).
        flags: Boolean outcome flags (converged, hit_length_cap, blank_output).
    """

    example_id: str
    scores: dict[str, float]
    trajectory: dict[str, float]
    flags: dict[str, bool]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _adapter_run_id(manifest: dict[str, Any]) -> str | None:
    """The training run id inside a served adapter path such as ``.../<run_id>/final``."""
    adapter = manifest.get("lora_adapter")
    if not isinstance(adapter, dict) or not adapter.get("path"):
        return None
    parts = Path(str(adapter["path"])).parts
    for part in reversed(parts):
        if "_grpo_" in part:
            return part
    return None


def _record(run_dir: Path) -> RunRecord | None:
    manifest_path, summary_path = run_dir / "manifest.json", run_dir / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest, summary = _load_json(manifest_path), _load_json(summary_path)
    cell = manifest.get("cell")
    seeding = manifest.get("seeding") or {}
    git = manifest.get("git") or {}
    common = dict(
        run_id=run_dir.name,
        run_dir=run_dir,
        status=str(summary.get("status", "unknown")),
        seed=seeding.get("seed", summary.get("seed")),
        started_at=str(manifest.get("started_at") or summary.get("started_at") or ""),
        clean_tree=not bool(git.get("dirty", False)),
        summary=summary,
        config_name=manifest.get("config_name"),
    )
    if isinstance(cell, dict) and cell.get("topology"):
        return RunRecord(
            kind="agent_eval",
            topology=cell["topology"],
            task=cell.get("task"),
            backbone=cell.get("backbone"),
            adapter_run_id=_adapter_run_id(manifest),
            **common,
        )
    backbone = manifest.get("backbone")
    if "grpo_args" in manifest and manifest.get("task"):
        return RunRecord(
            kind="training",
            topology=None,
            task=manifest["task"] if isinstance(manifest["task"], str) else None,
            backbone=backbone.get("name") if isinstance(backbone, dict) else backbone,
            **common,
        )
    return None


def scan_results(results_dir: Path) -> list[RunRecord]:
    """Every agent-evaluation and training attempt under ``results_dir``, oldest first.

    Args:
        results_dir: The ``results/`` folder.

    Returns:
        One record per run folder that has a manifest and a summary; analysis runs are skipped.
    """
    records = [
        record
        for run_dir in sorted(p for p in Path(results_dir).iterdir() if p.is_dir())
        if (record := _record(run_dir)) is not None
    ]
    return sorted(records, key=lambda r: (r.started_at, r.run_id))


def canonical_runs(records: Iterable[RunRecord]) -> list[RunRecord]:
    """The latest successful agent-evaluation run for each ``(task, topology, backbone, seed)``.

    A later successful attempt supersedes an earlier one, so prompt fixes and re-runs replace the
    runs they were made to fix.
    """
    latest: dict[tuple[str, str, str, int | None], RunRecord] = {}
    for record in records:
        if record.kind != "agent_eval" or not record.succeeded:
            continue
        key = (*record.cell, record.seed)
        if key not in latest or (record.started_at, record.run_id) > (
            latest[key].started_at,
            latest[key].run_id,
        ):
            latest[key] = record
    return sorted(latest.values(), key=lambda r: (*r.cell, r.seed if r.seed is not None else -1))


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_bool(value: str) -> bool | None:
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    return None


def read_examples(record: RunRecord) -> tuple[list[Example], str]:
    """Per-example scores and trajectory fields from a run's ``metrics.csv``.

    Args:
        record: An agent-evaluation run.

    Returns:
        The examples, and the SHA-256 of the file read (for provenance).
    """
    path = record.run_dir / "metrics.csv"
    raw = path.read_bytes()
    rows = csv.DictReader(raw.decode("utf-8").splitlines())
    examples: list[Example] = []
    for row in rows:
        scores = {
            key: score
            for key, value in row.items()
            if key not in NON_SCORE_FIELDS and key is not None
            if (score := _to_float(value)) is not None
        }
        trajectory = {
            key: number
            for key in TRAJECTORY_FIELDS
            if (number := _to_float(row.get(key, ""))) is not None
        }
        flags = {
            key: flag for key in FLAG_FIELDS if (flag := _to_bool(row.get(key, ""))) is not None
        }
        examples.append(Example(str(row["example_id"]), scores, trajectory, flags))
    return examples, hashlib.sha256(raw).hexdigest()


def group_by_cell(records: Sequence[RunRecord]) -> dict[tuple[str, str, str], list[RunRecord]]:
    """Records grouped by ``(task, topology, backbone)``."""
    grouped: dict[tuple[str, str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.cell, []).append(record)
    return grouped
