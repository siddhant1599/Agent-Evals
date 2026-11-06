"""Reproduce the published validation number of each metric port (D-018, D-023 to D-025, D-124).

    python3 src/validate_metrics.py --metric summac --metric alignscore

This checks the ports; it is not a run, so nothing goes to ``results/``. The summary and every
per-item score go to ``<metrics.cache_dir>/validation/`` so each reported number can be traced.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import ConfigError, LoggingConfig, MetricsConfig, ServingConfig, load_group
from ..env import EnvFileError, load_env_file
from ..logging_utils import configure_logging, get_logger
from .base import MetricError
from .validation import (
    balanced_accuracy,
    choose_best_threshold,
    load_aggrefact,
    load_xsumfaith,
    roc_auc,
)

__all__ = ["Reproduction", "main", "reproduce"]

logger = get_logger(__name__)

_XSUMFAITH = "SummaC benchmark XSumFaith, test split"


@dataclass(frozen=True, slots=True)
class Reproduction:
    """One achieved number next to its published target, both in percent.

    Attributes:
        score: Metric key the number is computed from.
        dataset: Evaluation data.
        statistic: What was computed, including how any threshold was set.
        achieved: Value this port produced.
        target: Published value.
        n_items: Items the value was computed over.
    """

    score: str
    dataset: str
    statistic: str
    achieved: float
    target: float
    n_items: int

    @property
    def gap(self) -> float:
        """Achieved minus target, in percentage points."""
        return self.achieved - self.target


Details = dict[str, object]
Reproducer = Callable[[MetricsConfig, dict[str, object]], tuple[list[Reproduction], Details]]


def _columns(items: list[dict[str, object]], summary_key: str) -> tuple[list, list, list]:
    return (
        [item["document"] for item in items],
        [item[summary_key] for item in items],
        [item["label"] for item in items],
    )


def _xsumfaith(cfg: MetricsConfig, cache: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    if "xsumfaith" not in cache:
        cache["xsumfaith"] = load_xsumfaith(cfg.validation, Path(cfg.cache_dir).expanduser())
    return cache["xsumfaith"]


def _reproduce_summac(
    cfg: MetricsConfig, cache: dict[str, object]
) -> tuple[list[Reproduction], Details]:
    from .summac import SummaC

    data = _xsumfaith(cfg, cache)
    val_docs, val_sums, val_labels = _columns(data["val"], "summary")
    test_docs, test_sums, test_labels = _columns(data["test"], "summary")
    metric = SummaC(cfg)
    val_scores = metric.pair_scores(val_docs, val_sums)
    test_scores = metric.pair_scores(test_docs, test_sums)
    targets = {
        "summac_zs": cfg.validation.target_summac_zs_bacc,
        "summac_conv": cfg.validation.target_summac_conv_bacc,
    }
    results, thresholds = [], {}
    for key, target in targets.items():
        threshold, _ = choose_best_threshold(
            val_labels, val_scores[key], cfg.validation.threshold_percentile_step
        )
        thresholds[key] = threshold
        predictions = [int(score > threshold) for score in test_scores[key]]
        results.append(
            Reproduction(
                score=key,
                dataset=_XSUMFAITH,
                statistic="balanced accuracy, threshold selected on the validation split",
                achieved=100.0 * balanced_accuracy(test_labels, predictions),
                target=target,
                n_items=len(test_labels),
            )
        )
    details = {
        "thresholds": thresholds,
        "val_labels": val_labels,
        "val_scores": val_scores,
        "test_labels": test_labels,
        "test_scores": test_scores,
    }
    return results, details


def _auc_reproducer(key: str, target_field: str) -> Reproducer:
    def run(cfg: MetricsConfig, cache: dict[str, object]) -> tuple[list[Reproduction], Details]:
        from . import AlignScore, FactCC

        data = _xsumfaith(cfg, cache)
        docs, sums, labels = _columns(data["test"], "summary")
        metric = {"alignscore": AlignScore, "factcc": FactCC}[key](cfg)
        scores = metric.pair_scores(docs, sums)
        result = Reproduction(
            score=key,
            dataset=_XSUMFAITH,
            statistic="ROC AUC",
            achieved=100.0 * roc_auc(labels, scores),
            target=getattr(cfg.validation, target_field),
            n_items=len(labels),
        )
        return [result], {"test_labels": labels, "test_scores": scores}

    return run


def _reproduce_minicheck(
    cfg: MetricsConfig, cache: dict[str, object]
) -> tuple[list[Reproduction], Details]:
    from .minicheck import MiniCheck

    items = load_aggrefact(cfg.validation)
    docs, claims, labels = _columns(items, "claim")
    metric = MiniCheck(cfg)
    scores = [metric.claim_support(doc, claim) for doc, claim in zip(docs, claims, strict=True)]
    predictions = [int(score > cfg.minicheck.threshold) for score in scores]
    result = Reproduction(
        score="minicheck",
        dataset=f"LLM-AggreFact {cfg.validation.aggrefact_dataset}, test split",
        statistic=f"balanced accuracy, fixed threshold {cfg.minicheck.threshold}",
        achieved=100.0 * balanced_accuracy(labels, predictions),
        target=cfg.validation.target_minicheck_bacc,
        n_items=len(labels),
    )
    return [result], {"labels": labels, "scores": scores}


def _qa_reproducer(key: str) -> Reproducer:
    """MiniCheck or AlignScore on LLM-AggreFact's long-form QA subsets, as ASQA's judges (D-124).

    Balanced accuracy at a fixed threshold, the setting of Tang et al. 2024, Table 2: MiniCheck's
    configured threshold, and AlignScore's ``validation.alignscore_threshold``.
    """

    def run(cfg: MetricsConfig, cache: dict[str, object]) -> tuple[list[Reproduction], Details]:
        from .alignscore import AlignScore
        from .minicheck import MiniCheck

        validation = cfg.validation
        if key == "minicheck":
            checker = MiniCheck(cfg)
            threshold = cfg.minicheck.threshold
            targets = validation.target_minicheck_qa_bacc

            def scores_of(docs: list, claims: list) -> list[float]:
                return [checker.claim_support(d, c) for d, c in zip(docs, claims, strict=True)]

        else:
            aligner = AlignScore(cfg)
            threshold = validation.alignscore_threshold
            targets = validation.target_alignscore_qa_bacc

            def scores_of(docs: list, claims: list) -> list[float]:
                return [float(s) for s in aligner.pair_scores(docs, claims)]

        results, details = [], {}
        for subset, target in zip(validation.aggrefact_qa_subsets, targets, strict=True):
            docs, claims, labels = _columns(load_aggrefact(validation, subset), "claim")
            scores = scores_of(docs, claims)
            predictions = [int(score > threshold) for score in scores]
            results.append(
                Reproduction(
                    score=key,
                    dataset=f"LLM-AggreFact {subset}, test split",
                    statistic=f"balanced accuracy, fixed threshold {threshold}",
                    achieved=100.0 * balanced_accuracy(labels, predictions),
                    target=target,
                    n_items=len(labels),
                )
            )
            details[subset] = {"labels": labels, "scores": scores}
        return results, details

    return run


REPRODUCERS: dict[str, Reproducer] = {
    "summac": _reproduce_summac,
    "alignscore": _auc_reproducer("alignscore", "target_alignscore_auc"),
    "factcc": _auc_reproducer("factcc", "target_factcc_auc"),
    "minicheck": _reproduce_minicheck,
    "minicheck_qa": _qa_reproducer("minicheck"),
    "alignscore_qa": _qa_reproducer("alignscore"),
}


def reproduce(cfg: MetricsConfig, names: list[str], out_dir: Path) -> list[Reproduction]:
    """Run the named reproductions one at a time, freeing each model before the next loads.

    Args:
        cfg: Metric configuration.
        names: Keys of :data:`REPRODUCERS`.
        out_dir: Where each reproduction's JSON record is written.

    Returns:
        Every achieved number with its target.

    Raises:
        MetricError: If a download, digest check or scorer fails.
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, object] = {}
    everything: list[Reproduction] = []
    for name in names:
        started = time.monotonic()
        results, details = REPRODUCERS[name](cfg, cache)
        elapsed = time.monotonic() - started
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        record = {
            "metric": name,
            "finished_utc": stamp,
            "elapsed_s": elapsed,
            "results": [{**dataclasses.asdict(r), "gap": r.gap} for r in results],
            "config": dataclasses.asdict(cfg),
            "details": details,
        }
        path = out_dir / f"{name}-{stamp}.json"
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        for r in results:
            logger.info(
                "%s | %s | %s: achieved %.1f, target %.1f, gap %+.1f (n=%d, %.0f s) -> %s",
                r.score,
                r.dataset,
                r.statistic,
                r.achieved,
                r.target,
                r.gap,
                r.n_items,
                elapsed,
                path,
            )
        everything.extend(results)
    return everything


def main(argv: list[str] | None, repo_root: Path, default_config: Path) -> int:
    """Command-line wrapper for :func:`reproduce`.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root, against which the ``.env`` path resolves.
        default_config: Config used when ``--config`` is not given.

    Returns:
        0 when every requested reproduction ran, 1 otherwise. A gap to the target is reported, not
        treated as a failure.
    """
    parser = argparse.ArgumentParser(
        description="Reproduce the published validation number of each metric port."
    )
    parser.add_argument(
        "--metric",
        dest="metrics",
        action="append",
        choices=sorted(REPRODUCERS),
        help="metric to validate; repeat for several; default all",
    )
    parser.add_argument("--config", type=Path, default=default_config, help="config file to use")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    try:
        configure_logging(load_group(args.config, "logging", LoggingConfig, args.overrides))
        cfg = load_group(args.config, "metrics", MetricsConfig, args.overrides)
        serving = load_group(args.config, "serving", ServingConfig, args.overrides)
        # LLM-AggreFact is gated; the HuggingFace token comes from the same .env serving uses.
        load_env_file(repo_root / serving.env_file)
        names = args.metrics or list(REPRODUCERS)
        reproduce(cfg, names, Path(cfg.cache_dir).expanduser() / "validation")
    except (ConfigError, EnvFileError, MetricError) as exc:
        logger.error("metric validation failed: %s", exc)
        return 1
    return 0
