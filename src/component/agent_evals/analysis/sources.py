"""Finished runs read back from ``results/<run_id>/`` for analysis (AGENTS.md §3).

Only ``summary.json``, ``manifest.json``, ``metrics.csv`` and ``trace.jsonl`` are read. A run that
did not succeed, or ran from a dirty tree and so may not be cited, is refused rather than skipped.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .bootstrap import AnalysisError

__all__ = ["SourceRun", "TraceMessage", "read_source_run"]

#: ``finish_reason`` of a call that stopped at the completion cap.
LENGTH_CAP = "length"


@dataclass(frozen=True)
class TraceMessage:
    """One model call of a run, as ``trace.jsonl`` recorded it."""

    role: str
    round_index: int
    content: str
    completion_tokens: int
    finish_reason: str

    @property
    def hit_cap(self) -> bool:
        """Whether the call stopped at the completion cap."""
        return self.finish_reason == LENGTH_CAP


@dataclass(frozen=True)
class SourceRun:
    """A finished run: its cell, its per-question scores and its per-question calls.

    Attributes:
        run_id: The run's id.
        topology: Topology name from the manifest's cell.
        backbone: Backbone name from the manifest's cell.
        task: Task name from the manifest's cell.
        seed: The run's seed.
        metrics_sha256: SHA-256 of the ``metrics.csv`` read.
        trace_sha256: SHA-256 of the ``trace.jsonl`` read.
        scores: Question id -> metric key -> score.
        calls: Question id -> its calls in ``message_index`` order.
    """

    run_id: str
    topology: str
    backbone: str
    task: str
    seed: int
    metrics_sha256: str
    trace_sha256: str
    scores: Mapping[str, Mapping[str, float]]
    calls: Mapping[str, tuple[TraceMessage, ...]]


def read_source_run(path: Path, metric_keys: Sequence[str]) -> SourceRun:
    """Read a finished run's scores and calls.

    Args:
        path: The run directory.
        metric_keys: ``metrics.csv`` columns to read as scores.

    Returns:
        The run.

    Raises:
        AnalysisError: If the directory is missing, the run did not succeed or is dirty, a
            requested column is absent, or a question has scores but no calls.
    """
    if not path.is_dir():
        raise AnalysisError(f"no run directory at {path}")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise AnalysisError(f"{path.name}: status {summary.get('status')!r}, not success")
    if (manifest.get("git") or {}).get("dirty") is not False:
        raise AnalysisError(f"{path.name}: not recorded as a clean tree, so it may not be cited")
    cell = manifest.get("cell") or {}
    metrics_raw = (path / "metrics.csv").read_bytes()
    trace_raw = (path / "trace.jsonl").read_bytes()
    scores = _read_scores(path.name, metrics_raw, metric_keys)
    calls = _read_calls(trace_raw)
    missing = sorted(set(scores) - set(calls))
    if missing:
        raise AnalysisError(f"{path.name}: {len(missing)} question(s) have scores but no calls")
    return SourceRun(
        run_id=summary["run_id"],
        topology=cell.get("topology", ""),
        backbone=cell.get("backbone", ""),
        task=cell.get("task", ""),
        seed=int(summary["seed"]),
        metrics_sha256=hashlib.sha256(metrics_raw).hexdigest(),
        trace_sha256=hashlib.sha256(trace_raw).hexdigest(),
        scores=scores,
        calls=calls,
    )


def _read_scores(name: str, raw: bytes, metric_keys: Sequence[str]) -> dict[str, dict[str, float]]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    absent = sorted(set(metric_keys) - set(reader.fieldnames or ()))
    if absent:
        raise AnalysisError(f"{name}: metrics.csv has no column(s) {absent}")
    return {row["example_id"]: {k: float(row[k]) for k in metric_keys} for row in reader}


def _read_calls(raw: bytes) -> dict[str, tuple[TraceMessage, ...]]:
    by_question: dict[str, list[tuple[int, TraceMessage]]] = defaultdict(list)
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        message = TraceMessage(
            role=record["role"],
            round_index=int(record["round_index"]),
            content=record["content"],
            completion_tokens=int(record["completion_tokens"]),
            finish_reason=record["finish_reason"],
        )
        by_question[record["example_id"]].append((int(record["message_index"]), message))
    return {q: tuple(m for _, m in sorted(ms, key=lambda x: x[0])) for q, ms in by_question.items()}
