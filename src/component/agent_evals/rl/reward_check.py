"""The ASQA reward-signal check (D-136): the reward's citation judge against the evaluation judge.

T5 trains on citation recall and precision judged by AlignScore (D-135), while every reported ASQA
score is judged by MiniCheck. This check rescores the outputs of finished T0 ASQA runs with the
AlignScore judge, generating nothing, and compares the two judges on the same outputs. A reward
that disagreed with the evaluation judge would train T5 towards a score it is not reported on.

It also counts ties: how often a question's outputs, one per seed, score the same on a reward
term. Each T0 seed draws its own pair of worked examples (D-118), so this approximates, and does
not reproduce, the ties within a GRPO group.

Usage, after the T0 runs and with the GPU free::

    python -m agent_evals.rl.reward_check \\
        --config src/configs/rl/asqa_reward_check.yaml --set run.owner=sameer \\
        --runs ~/fall-2026-group1/results/<t0 run id> ...

The check writes an ordinary ``results/<run_id>/``: one row per scored output in
``metrics.csv`` and the comparison in ``summary.json``. The source runs are named in its manifest
with the SHA-256 of each ``metrics.csv`` read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, LoggingConfig, load_config
from ..logging_utils import configure_logging, get_logger
from ..metrics.citation import CitationQuality
from ..registry import DATASETS
from ..run import RunDirectory
from ..types import Example

__all__ = [
    "ReviewedOutput",
    "RewardCheckError",
    "SourceRun",
    "agreement",
    "main",
    "pearson",
    "read_source_run",
    "spearman",
    "tie_rate",
]

logger = get_logger(__name__)

#: The topology and task every source run must be.
_TOPOLOGY = "t0_single_agent"
_TASK = "tier2_asqa"
#: Terms compared across seeds for ties: correctness and the two AlignScore-judged citation terms.
_TIE_TERMS = ("str_em", "citation_recall_alignscore", "citation_precision_alignscore")


class RewardCheckError(RuntimeError):
    """A source run cannot be read, or does not match the check's dataset."""


@dataclass(frozen=True)
class ReviewedOutput:
    """One T0 output with its MiniCheck-judged scores, as the T0 run recorded them."""

    example_id: str
    output: str
    str_em: float
    citation_recall: float
    citation_precision: float


@dataclass(frozen=True)
class SourceRun:
    """A finished T0 ASQA run read from its directory."""

    run_id: str
    backbone: str
    seed: int
    metrics_sha256: str
    outputs: tuple[ReviewedOutput, ...]


def read_source_run(path: Path) -> SourceRun:
    """Read a T0 ASQA run's non-blank outputs and their recorded scores.

    Args:
        path: The run directory.

    Returns:
        The run, its outputs in file order.

    Raises:
        RewardCheckError: If the run did not succeed, is not T0 on ASQA, or has no outputs.
    """
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if summary.get("status") != "success":
        raise RewardCheckError(f"{path.name}: status {summary.get('status')!r}, not success")
    cell = manifest.get("cell") or {}
    if cell.get("topology") != _TOPOLOGY or cell.get("task") != _TASK:
        raise RewardCheckError(f"{path.name}: not a {_TOPOLOGY} run on {_TASK}: {cell}")
    raw = (path / "metrics.csv").read_bytes()
    outputs = []
    for row in csv.DictReader(raw.decode("utf-8").splitlines()):
        if row["blank_output"] == "True":
            continue
        outputs.append(
            ReviewedOutput(
                example_id=row["example_id"],
                output=row["output"],
                str_em=float(row["str_em"]),
                citation_recall=float(row["citation_recall"]),
                citation_precision=float(row["citation_precision"]),
            )
        )
    if not outputs:
        raise RewardCheckError(f"{path.name}: no non-blank outputs")
    return SourceRun(
        run_id=summary["run_id"],
        backbone=cell["backbone"],
        seed=int(summary["seed"]),
        metrics_sha256=hashlib.sha256(raw).hexdigest(),
        outputs=tuple(outputs),
    )


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation, or None when either series is constant or shorter than 2."""
    if len(x) != len(y):
        raise ValueError("series differ in length")
    n = len(x)
    if n < 2:
        return None
    mean_x, mean_y = math.fsum(x) / n, math.fsum(y) / n
    cov = math.fsum((a - mean_x) * (b - mean_y) for a, b in zip(x, y, strict=True))
    var_x = math.fsum((a - mean_x) ** 2 for a in x)
    var_y = math.fsum((b - mean_y) ** 2 for b in y)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: Sequence[float]) -> list[float]:
    """Ranks from 1, ties given their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for k in range(start, end + 1):
            ranks[order[k]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Spearman correlation (Pearson of tie-averaged ranks), or None as for :func:`pearson`."""
    return pearson(_ranks(x), _ranks(y))


def agreement(minicheck: Sequence[float], alignscore: Sequence[float]) -> dict[str, Any]:
    """How closely the two judges' scores of the same outputs agree.

    Args:
        minicheck: Scores under the MiniCheck judge.
        alignscore: Scores of the same outputs under the AlignScore judge.

    Returns:
        Both means, the mean difference (AlignScore minus MiniCheck), Pearson and Spearman
        correlations, and the share of outputs scored exactly the same.
    """
    n = len(minicheck)
    if n == 0 or n != len(alignscore):
        raise ValueError("need two equal, non-empty series")
    return {
        "n": n,
        "mean_minicheck": math.fsum(minicheck) / n,
        "mean_alignscore": math.fsum(alignscore) / n,
        "mean_difference": math.fsum(b - a for a, b in zip(minicheck, alignscore, strict=True)) / n,
        "pearson": pearson(minicheck, alignscore),
        "spearman": spearman(minicheck, alignscore),
        "exact_agreement": sum(a == b for a, b in zip(minicheck, alignscore, strict=True)) / n,
    }


def tie_rate(scores_by_question: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Share of questions whose outputs all score the same.

    Args:
        scores_by_question: Each question's scores, one per output; questions with fewer than two
            outputs are left out.

    Returns:
        The number of questions counted and the share tied.
    """
    groups = [s for s in scores_by_question.values() if len(s) >= 2]
    if not groups:
        return {"questions": 0, "tied": None}
    return {"questions": len(groups), "tied": sum(len(set(s)) == 1 for s in groups) / len(groups)}


class RewardCheck:
    """Rescore T0 ASQA outputs with the AlignScore citation judge and compare the judges.

    Args:
        cfg: The check's config: the ASQA evaluation data and ``metrics.citation.judge:
            alignscore``.
        repo_root: Repository root.
    """

    def __init__(self, cfg: ExperimentConfig, repo_root: Path) -> None:
        if cfg.metrics.citation.judge != "alignscore":
            raise RewardCheckError("set metrics.citation.judge: alignscore, the reward's judge")
        self._cfg = cfg
        self._repo_root = repo_root

    def run(self, run_dirs: Sequence[Path]) -> Path:
        """Score every source run and write the comparison.

        Args:
            run_dirs: T0 ASQA run directories.

        Returns:
            The check's run directory.

        Raises:
            RewardCheckError: If a source run cannot be read or names a question outside the
                configured sample, or two source runs are the same backbone and seed.
        """
        sources = [read_source_run(Path(d).expanduser()) for d in run_dirs]
        cells = [(s.backbone, s.seed) for s in sources]
        if len(set(cells)) != len(cells):
            raise RewardCheckError(f"a backbone and seed appear twice: {cells}")
        adapter = DATASETS.build(self._cfg.data.name, self._cfg.data)
        examples = {e.id: e for e in (adapter.as_example(row) for row in adapter.build())}
        for source in sources:
            missing = {o.example_id for o in source.outputs} - examples.keys()
            if missing:
                raise RewardCheckError(
                    f"{source.run_id}: {len(missing)} question(s) outside the configured sample"
                )
        manifest = {
            "source_runs": [
                {"run_id": s.run_id, "metrics_sha256": s.metrics_sha256} for s in sources
            ]
        }
        with RunDirectory(self._cfg, self._repo_root, manifest_extra=manifest) as run:
            metric = CitationQuality(self._cfg.metrics)
            rows: list[dict[str, Any]] = []
            for source in sources:
                logger.info("scoring %d outputs of %s", len(source.outputs), source.run_id)
                rows += self._score(metric, source, examples)
            for row in rows:
                run.metrics.write(row)
            run.add_summary(**summarize(rows))
        return run.path

    def _score(
        self, metric: CitationQuality, source: SourceRun, examples: Mapping[str, Example]
    ) -> list[dict[str, Any]]:
        used = [examples[o.example_id] for o in source.outputs]
        scores = metric.score(
            [o.output for o in source.outputs],
            [e.reference for e in used],
            [e.evidence for e in used],
            [e.gold for e in used],
        )
        return [
            {
                "source_run_id": source.run_id,
                "backbone": source.backbone,
                "seed": source.seed,
                "example_id": o.example_id,
                "str_em": o.str_em,
                "citation_recall_minicheck": o.citation_recall,
                "citation_precision_minicheck": o.citation_precision,
                "citation_recall_alignscore": s["citation_recall"],
                "citation_precision_alignscore": s["citation_precision"],
            }
            for o, s in zip(source.outputs, scores, strict=True)
        ]


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Judge agreement and tie rates, per backbone and over all rows.

    Args:
        rows: The check's per-output rows.

    Returns:
        ``{"by_backbone": {...}, "all": {...}}``, each with ``citation_recall`` and
        ``citation_precision`` agreement and a tie rate per term in ``_TIE_TERMS``.
    """
    by_backbone: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_backbone[row["backbone"]].append(row)

    def block(group: Sequence[Mapping[str, Any]], per_backbone: bool) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for term in ("citation_recall", "citation_precision"):
            result[term] = agreement(
                [r[f"{term}_minicheck"] for r in group], [r[f"{term}_alignscore"] for r in group]
            )
        if per_backbone:
            for term in _TIE_TERMS:
                scores: dict[str, list[float]] = defaultdict(list)
                for r in group:
                    scores[r["example_id"]].append(r[term])
                result[f"ties_{term}"] = tie_rate(scores)
        return result

    return {
        "by_backbone": {b: block(g, True) for b, g in sorted(by_backbone.items())},
        "all": block(rows, False),
    }


def main(argv: list[str] | None, repo_root: Path) -> int:
    """Command-line wrapper for the check.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root.

    Returns:
        0 on success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="ASQA reward-signal check (D-136).")
    parser.add_argument("--config", required=True, type=Path, help="check config")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    parser.add_argument("--runs", required=True, nargs="+", type=Path, help="T0 ASQA run dirs")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides or [])
        configure_logging(cfg.logging)
        RewardCheck(cfg, repo_root).run(args.runs)
    except Exception:  # the one top-level catch (AGENTS.md §6)
        configure_logging(LoggingConfig())
        logger.exception("reward-signal check did not complete")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], Path(__file__).resolve().parents[4]))
