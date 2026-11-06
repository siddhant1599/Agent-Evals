"""Data and statistics for validating the metric ports against published numbers.

The statistics reproduce what the source papers computed: balanced accuracy with SummaC's threshold
search (``summac/utils_scorer.py``, ``choose_best_threshold``) and ROC AUC. The data loaders rebuild
the SummaC benchmark's XSumFaith split exactly as ``summac/benchmark.py`` does, and read the
AggreFact-XSum subset of LLM-AggreFact.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..config import MetricValidationConfig
from .artifacts import fetch_pinned, sha256_file
from .base import MetricError

__all__ = [
    "balanced_accuracy",
    "build_xsumfaith",
    "choose_best_threshold",
    "load_aggrefact",
    "load_xsumfaith",
    "roc_auc",
]


def balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    """Mean recall over the classes present in ``labels``, as scikit-learn computes it.

    Args:
        labels: Gold binary labels.
        predictions: Predicted binary labels, aligned with ``labels``.

    Returns:
        Balanced accuracy in [0, 1].

    Raises:
        MetricError: If the inputs are empty or misaligned.
    """
    gold = np.asarray(labels)
    pred = np.asarray(predictions)
    if gold.size == 0 or gold.shape != pred.shape:
        raise MetricError("balanced accuracy needs non-empty, aligned labels and predictions")
    recalls = [float(np.mean(pred[gold == cls] == cls)) for cls in np.unique(gold)]
    return float(np.mean(recalls))


def choose_best_threshold(
    labels: Sequence[int], scores: Sequence[float], percentile_step: float
) -> tuple[float, float]:
    """SummaC's threshold search: the percentile threshold with the best balanced accuracy.

    Candidates are the score percentiles at ``0, step, 2*step, ...`` below 100; a score strictly
    above the threshold is predicted consistent. Ties keep the later threshold, as the original
    ``>=`` comparison does.

    Args:
        labels: Gold binary labels.
        scores: Metric scores, aligned with ``labels``.
        percentile_step: Spacing of candidate percentiles.

    Returns:
        ``(threshold, balanced_accuracy)`` for the best candidate.
    """
    values = np.asarray(scores, dtype=float)
    best_threshold, best_accuracy = 0.0, 0.0
    for percentile in np.arange(0, 100, percentile_step):
        threshold = float(np.percentile(values, percentile))
        accuracy = balanced_accuracy(labels, (values > threshold).astype(int))
        if accuracy >= best_accuracy:
            best_threshold, best_accuracy = threshold, accuracy
    return best_threshold, best_accuracy


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Area under the ROC curve, with tied scores counted as half, as scikit-learn computes it.

    Args:
        labels: Gold binary labels; 1 is the positive class.
        scores: Metric scores, aligned with ``labels``.

    Returns:
        ROC AUC in [0, 1].

    Raises:
        MetricError: If either class is absent or the inputs are misaligned.
    """
    gold = np.asarray(labels).astype(bool)
    values = np.asarray(scores, dtype=float)
    positives, negatives = int(gold.sum()), int((~gold).sum())
    if gold.shape != values.shape or positives == 0 or negatives == 0:
        raise MetricError("ROC AUC needs aligned inputs containing both classes")
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)
    ranks = (upper - (counts - 1) / 2.0)[inverse]
    return float((ranks[gold].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def build_xsumfaith(
    rows: Sequence[Mapping[str, str]], documents: Mapping[str, str]
) -> dict[str, list[dict[str, object]]]:
    """Group XSumFaith annotations into labelled pairs and split them as the SummaC benchmark does.

    Annotations are grouped by (article, system) in file order. The label is 1 when the most common
    annotation is ``NULL`` (no hallucination), else 0. Groups alternate between validation and test.

    Args:
        rows: Annotation rows with ``bbcid``, ``system``, ``summary`` and ``hallucination_type``.
        documents: Article text keyed by BBC id.

    Returns:
        ``{"val": [...], "test": [...]}``, each item holding ``document``, ``summary``, ``label``.

    Raises:
        MetricError: If an annotated article is missing from ``documents``.
    """
    groups: dict[tuple[str, str], list[Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["bbcid"], row["system"]), []).append(row)
    split: dict[str, list[dict[str, object]]] = {"val": [], "test": []}
    for index, ((bbcid, _system), members) in enumerate(groups.items()):
        if bbcid not in documents:
            raise MetricError(f"XSumFaith article {bbcid} is missing from the XSum test split")
        majority = Counter(member["hallucination_type"] for member in members).most_common(1)[0][0]
        item = {
            "document": documents[bbcid],
            "summary": members[0]["summary"],
            "label": 1 if majority == "NULL" else 0,
        }
        split["val" if index % 2 == 0 else "test"].append(item)
    return split


def load_xsumfaith(
    cfg: MetricValidationConfig, cache_dir: Path
) -> dict[str, list[dict[str, object]]]:
    """Download the pinned annotations and pinned XSum articles, and build the benchmark split.

    Args:
        cfg: Validation configuration.
        cache_dir: Where the annotation file is cached.

    Returns:
        As :func:`build_xsumfaith`.
    """
    from datasets import load_dataset

    path = fetch_pinned(cfg.xsumfaith_url, cfg.xsumfaith_sha256, cache_dir / "validation")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    xsum = load_dataset(cfg.xsum_path, revision=cfg.xsum_revision, split="test")
    documents = dict(zip(xsum["id"], xsum["document"], strict=True))
    return build_xsumfaith(rows, documents)


def load_aggrefact(
    cfg: MetricValidationConfig, subset: str | None = None
) -> list[dict[str, object]]:
    """Read the pinned LLM-AggreFact test file and keep one subset.

    The dataset is gated: the account behind the environment's HuggingFace token must have accepted
    its terms.

    Args:
        cfg: Validation configuration.
        subset: Subset to keep, as the file names it; defaults to ``cfg.aggrefact_dataset``.

    Returns:
        Items holding ``document``, ``claim`` and ``label``.

    Raises:
        MetricError: If the file does not match its pinned digest or the subset is empty.
    """
    from datasets import Dataset
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            cfg.aggrefact_repo,
            cfg.aggrefact_file,
            repo_type="dataset",
            revision=cfg.aggrefact_revision,
        )
    )
    actual = sha256_file(path)
    if actual != cfg.aggrefact_sha256:
        raise MetricError(f"{path} has sha256 {actual}, expected {cfg.aggrefact_sha256}")
    wanted = cfg.aggrefact_dataset if subset is None else subset
    table = Dataset.from_parquet(str(path)).filter(lambda row: row["dataset"] == wanted)
    if len(table) == 0:
        raise MetricError(f"LLM-AggreFact has no rows for {wanted}")
    return [
        {"document": str(doc), "claim": str(claim), "label": int(label)}
        for doc, claim, label in zip(table["doc"], table["claim"], table["label"], strict=True)
    ]
