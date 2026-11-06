"""Run directories, manifests and the metrics writer -- the reproducibility contract in code.

Every run produces ``results/<run_id>/`` holding exactly six files:

===========================  =========================================================
``config.resolved.yaml``     the merged config, enough to re-run from alone
``manifest.json``            git SHA, dirty flag, seed, GPU, versions, owner, timing
``run.log``                  the human-readable log
``metrics.csv``              one row per example or per step, append-only
``trace.jsonl``              every message of every result, one JSON line per message
``summary.json``             terminal state: status, wall time, cost, latency, peak VRAM
===========================  =========================================================

``RunDirectory`` is a context manager so the failure path always writes a summary. A run
directory that just stops halfway with no explanation is the thing this design exists to prevent:
OOM is a recorded outcome with a status of its own, not a crash.
"""

from __future__ import annotations

import csv
import json
import platform
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

from .config import ExperimentConfig, to_dict
from .logging_utils import configure_logging, get_logger, reset_logging
from .seeding import set_seed
from .types import TopologyResult

__all__ = [
    "RunDirectory",
    "MetricsWriter",
    "TraceWriter",
    "make_run_id",
    "git_info",
    "gpu_info",
    "RUN_ID_FORMAT",
    "STATUS_SUCCESS",
    "STATUS_FAILED",
    "STATUS_OOM",
]

logger = get_logger(__name__)

#: The run id format. Deliberately not a config field -- one format everywhere, always.
RUN_ID_FORMAT = "%Y%m%d_%H%M%S"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_OOM = "oom"

#: Keys ``RunDirectory`` writes itself; callers may add fields but never replace these.
_MANIFEST_KEYS = frozenset(
    {
        "run_id",
        "owner",
        "config_name",
        "started_at",
        "git",
        "seeding",
        "gpu",
        "libraries",
        "python",
        "platform",
        "hostname",
    }
)
_SUMMARY_KEYS = frozenset(
    {
        "run_id",
        "status",
        "owner",
        "seed",
        "started_at",
        "ended_at",
        "wall_clock_s",
        "records_written",
        "peak_vram_gib",
        "error",
    }
)

#: Libraries whose versions change results and therefore belong in every manifest.
_TRACKED_LIBRARIES = (
    "torch",
    "transformers",
    "vllm",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "numpy",
)


def make_run_id(config_name: str, seed: int, now: datetime | None = None) -> str:
    """Build a run id of the form ``YYYYMMDD_HHMMSS_<config_name>_seed<N>``.

    Args:
        config_name: Short config name, from ``ExperimentConfig.name``.
        seed: This run's seed.
        now: Timestamp to use. Defaults to the current local time; injectable for tests.

    Returns:
        The run id. This is the only format used anywhere.
    """
    stamp = (now or datetime.now()).strftime(RUN_ID_FORMAT)
    return f"{stamp}_{config_name}_seed{seed}"


def _git(args: list[str], repo: Path) -> str | None:
    """Run a read-only git command, returning None if it fails or git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Collect git provenance for the manifest. Read-only; never writes to history.

    Args:
        repo: Repository root. Defaults to the package's own location.

    Returns:
        Keys ``sha``, ``branch``, ``dirty``, ``untracked_files``. ``dirty`` counts tracked
        modifications only; untracked files are counted separately because a scratch file in the
        working tree is not the same as edited source. A run with ``dirty`` true may not be
        cited.
    """
    root = repo or Path(__file__).resolve().parents[3]
    porcelain = _git(["status", "--porcelain"], root)
    tracked_changes = untracked = 0
    if porcelain:
        for line in porcelain.splitlines():
            if line.startswith("??"):
                untracked += 1
            elif line.strip():
                tracked_changes += 1
    return {
        "sha": _git(["rev-parse", "HEAD"], root),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": tracked_changes > 0,
        "untracked_files": untracked,
    }


def gpu_info() -> dict[str, Any]:
    """Collect GPU details for the manifest.

    Returns:
        Keys ``available``, ``name``, ``total_vram_gib``, ``driver``, ``cuda``. All None when no
        GPU or no torch, so a CPU-only run records that fact rather than omitting the field.
        VRAM is reported in **GiB** (bytes / 1024**3), matching torch and ``nvidia-smi``; see
        decisions.md D-004 for why the key is not named ``_gb``.
    """
    info: dict[str, Any] = {
        "available": False,
        "name": None,
        "total_vram_gib": None,
        "driver": None,
        "cuda": None,
    }
    try:
        import torch
    except ImportError:
        return info
    if not torch.cuda.is_available():
        return info

    props = torch.cuda.get_device_properties(0)
    info.update(
        available=True,
        name=props.name,
        total_vram_gib=round(props.total_memory / 1024**3, 2),
        cuda=torch.version.cuda,
        driver=_nvidia_driver(),
    )
    return info


def _nvidia_driver() -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout else None


def _library_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str | None] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def _peak_vram_gib() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_reserved() / 1024**3, 2)


def _is_oom(exc: BaseException) -> bool:
    """Whether an exception is an out-of-memory condition.

    OOM is a first-class outcome with its own status, not a crash: a topology that
    does not fit on the target hardware is a reportable result.
    """
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except (ImportError, AttributeError):
        pass
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


_CSV_SCALARS = (str, int, float, bool, type(None))


class MetricsWriter:
    """Append-only CSV writer: one row per example (inference) or per step (training).

    Structured numbers go here, never into log text, so ``analysis/`` reads a table rather than
    parsing prose. The first record fixes the columns. Every later record must carry the same keys,
    in any order: a column that appears, disappears or is renamed mid-file breaks every table built
    from the rows before it. Values must be scalars, since a nested object has no faithful CSV
    cell. ``None`` is written as an empty cell.

    Args:
        path: Destination file, normally ``results/<run_id>/metrics.csv``. It must not exist yet:
            an existing file is evidence and is never appended to or overwritten.
        flush_every: Flush after this many records. 1 means every record, so a run killed by OOM
            keeps the evidence written up to that point.

    Raises:
        FileExistsError: If ``path`` already exists.
    """

    def __init__(self, path: Path, flush_every: int = 1) -> None:
        self._path = path
        self._flush_every = max(1, flush_every)
        self._since_flush = 0
        self._count = 0
        self._columns: tuple[str, ...] | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("x", encoding="utf-8", newline="")
        self._writer = csv.writer(self._handle)

    @property
    def count(self) -> int:
        """How many records have been written."""
        return self._count

    def write(self, record: dict[str, Any]) -> None:
        """Append one record.

        Args:
            record: Flat mapping of column name to scalar. Keys are stable forever once a run has
                used them -- renaming one invalidates every earlier run.

        Raises:
            TypeError: If a value is not a str, int, float, bool or None.
            ValueError: If the keys differ from the columns the first record fixed.
        """
        nested = sorted(k for k, v in record.items() if not isinstance(v, _CSV_SCALARS))
        if nested:
            raise TypeError(f"metrics columns {nested} are not scalars; flatten them first")
        if self._columns is None:
            self._columns = tuple(record)
            self._writer.writerow(self._columns)
        elif set(record) != set(self._columns):
            raise ValueError(
                f"record columns {sorted(record)} differ from the file's {sorted(self._columns)}"
            )
        self._writer.writerow(["" if record[c] is None else record[c] for c in self._columns])
        self._count += 1
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._handle.flush()
            self._since_flush = 0

    def close(self) -> None:
        """Flush and close the file."""
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


class TraceWriter:
    """Append-only JSON Lines writer: one line per message of every topology result.

    ``metrics.csv`` holds one final output per example; a multi-round topology's drafts, feedback
    and critiques would otherwise be lost. Each line names the run, the example and the message's
    position in the trace, with its role, round, text, token cost, latency and finish reason.

    Args:
        path: Destination file, normally ``results/<run_id>/trace.jsonl``. It must not exist yet.
        flush_every: Flush after this many results, as :class:`MetricsWriter` does per record.

    Raises:
        FileExistsError: If ``path`` already exists.
    """

    def __init__(self, path: Path, flush_every: int = 1) -> None:
        self._flush_every = max(1, flush_every)
        self._since_flush = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("x", encoding="utf-8")

    def write(self, run_id: str, example_id: str, result: TopologyResult) -> None:
        """Append one line per message of one result's trace, oldest first.

        Args:
            run_id: The run this result belongs to.
            example_id: The example the result is for.
            result: The result whose trace is written.
        """
        for index, message in enumerate(result.trace):
            record = {
                "run_id": run_id,
                "example_id": example_id,
                "message_index": index,
                "role": message.role,
                "round_index": message.round_index,
                "content": message.content,
                **message.token_cost.as_dict(),
                "latency_s": message.latency_s,
                "finish_reason": message.finish_reason,
            }
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._since_flush += 1
        if self._since_flush >= self._flush_every:
            self._handle.flush()
            self._since_flush = 0

    def close(self) -> None:
        """Flush and close the file."""
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


class RunDirectory:
    """Creates and owns one ``results/<run_id>/`` directory.

    Used as a context manager so that every exit path -- success, failure, or OOM -- writes a
    ``summary.json``. Entering seeds the process, configures logging into the run directory, and
    writes the resolved config and manifest before any work begins.

    Args:
        cfg: The fully resolved experiment config.
        repo_root: Repository root, used for git provenance and to resolve ``output_root``.
            Defaults to the repository this package lives in.
        now: Timestamp for the run id. Injectable for tests.
        manifest_extra: Fields added to ``manifest.json``, such as the backbone and prompt-template
            versions. They may not replace a field the manifest already records.

    Raises:
        ValueError: If ``manifest_extra`` repeats a built-in manifest key.

    Attributes:
        run_id: The generated run id.
        path: The run directory.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        repo_root: Path | None = None,
        now: datetime | None = None,
        manifest_extra: Mapping[str, Any] | None = None,
    ) -> None:
        clash = sorted(_MANIFEST_KEYS.intersection(manifest_extra or {}))
        if clash:
            raise ValueError(f"manifest_extra may not replace built-in manifest keys: {clash}")
        self._manifest_extra = dict(manifest_extra or {})
        self._summary_extra: dict[str, Any] = {}
        self._cfg = cfg
        self._repo_root = repo_root or Path(__file__).resolve().parents[3]
        self.run_id = make_run_id(cfg.name, cfg.run.seed, now)
        self.path = self._repo_root / cfg.run.output_root / self.run_id
        self._metrics: MetricsWriter | None = None
        self._traces: TraceWriter | None = None
        self._start_monotonic = 0.0
        self._start_iso = ""
        self._seed_record: dict[str, Any] = {}

    @property
    def metrics(self) -> MetricsWriter:
        """The metrics writer. Only available inside the context manager."""
        if self._metrics is None:
            raise RuntimeError("RunDirectory must be entered before writing metrics")
        return self._metrics

    def __enter__(self) -> RunDirectory:
        self.path.mkdir(parents=True, exist_ok=True)
        self._start_monotonic = time.monotonic()
        self._start_iso = datetime.now(timezone.utc).isoformat()

        configure_logging(self._cfg.logging, self.path / "run.log")
        self._seed_record = set_seed(self._cfg.run.seed, self._cfg.seeding.deterministic)

        (self.path / "config.resolved.yaml").write_text(
            yaml.safe_dump(to_dict(self._cfg), sort_keys=False), encoding="utf-8"
        )
        (self.path / "manifest.json").write_text(
            json.dumps(self._build_manifest(), indent=2, default=str), encoding="utf-8"
        )
        self._metrics = MetricsWriter(
            self.path / "metrics.csv", self._cfg.logging.flush_metrics_every
        )
        self._traces = TraceWriter(self.path / "trace.jsonl", self._cfg.logging.flush_metrics_every)

        logger.info("run %s started, owner=%s", self.run_id, self._cfg.run.owner)
        git = git_info(self._repo_root)
        if git["dirty"]:
            logger.warning(
                "working tree is dirty; this run is marked dirty and may not be cited. "
                "Commit first."
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc is None:
            status, error = STATUS_SUCCESS, None
        elif _is_oom(exc):
            status = STATUS_OOM
            error = {"type": type(exc).__name__, "message": str(exc)}
            logger.error("run %s ran out of memory; recorded as a result, not a crash", self.run_id)
        else:
            status = STATUS_FAILED
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception(exc_type, exc, tb)),
            }
            logger.error("run %s failed: %s", self.run_id, exc)

        records = self._metrics.count if self._metrics else 0
        if self._metrics is not None:
            self._metrics.close()
        if self._traces is not None:
            self._traces.close()

        summary = {
            "run_id": self.run_id,
            "status": status,
            "owner": self._cfg.run.owner,
            "seed": self._cfg.run.seed,
            "started_at": self._start_iso,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "wall_clock_s": round(time.monotonic() - self._start_monotonic, 3),
            "records_written": records,
            "peak_vram_gib": _peak_vram_gib(),
            "error": error,
            **self._summary_extra,
        }
        (self.path / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        logger.info("run %s finished with status=%s", self.run_id, status)
        reset_logging()
        return False

    def add_summary(self, **fields: Any) -> None:
        """Add run-level results, such as token totals and metric means, to ``summary.json``.

        Written on exit whatever the status, so call it once the values exist.

        Args:
            **fields: JSON-serializable values to record.

        Raises:
            ValueError: If a field would replace one ``summary.json`` already records.
        """
        clash = sorted(_SUMMARY_KEYS.intersection(fields))
        if clash:
            raise ValueError(f"add_summary may not replace built-in summary keys: {clash}")
        self._summary_extra.update(fields)

    def record_result(self, result: TopologyResult, *, example_id: str, **extra: Any) -> None:
        """Write one topology result to ``metrics.csv``, and its full trace to ``trace.jsonl``.

        Cost and latency are reported results, not telemetry, so a result missing them is
        rejected rather than stored as a null. ``TopologyResult`` already validates
        at construction; this is the second gate, because the writer is what the master DataFrame
        is built from.

        Args:
            result: The topology result to record.
            example_id: The example the result is for. Its ``metrics.csv`` column follows
                ``n_messages``, and it labels every ``trace.jsonl`` line of this result.
            **extra: Additional fields to merge into the record, e.g. per-example scores.

        Raises:
            TypeError: If ``result`` is not a TopologyResult.
            ValueError: If cost or latency is missing.
        """
        if not isinstance(result, TopologyResult):
            raise TypeError(f"expected TopologyResult, got {type(result).__name__}")
        if result.token_cost is None or result.wall_clock_s is None:
            raise ValueError(
                "TopologyResult is missing token cost or latency; H4 and H5 depend on them "
                "being present for every example"
            )

        self.metrics.write(
            {
                "run_id": self.run_id,
                "output": result.output,
                **result.token_cost.as_dict(),
                "wall_clock_s": result.wall_clock_s,
                "n_rounds": result.n_rounds,
                "converged": result.converged,
                "n_messages": len(result.trace),
                "example_id": example_id,
                **extra,
            }
        )
        if self._traces is not None:
            self._traces.write(self.run_id, example_id, result)

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "owner": self._cfg.run.owner,
            "config_name": self._cfg.name,
            "started_at": self._start_iso,
            "git": git_info(self._repo_root),
            "seeding": self._seed_record,
            "gpu": gpu_info(),
            "libraries": _library_versions(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            **self._manifest_extra,
        }
