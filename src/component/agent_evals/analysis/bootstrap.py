"""Paired gains with percentile bootstrap intervals (D-115).

The unit is one question: the topology's score on it, averaged over seeds, minus the baseline's,
averaged over seeds. A gain is the mean of those differences, and its interval comes from
resampling the questions with replacement. The length-adjusted change is the same per-question
difference regressed on the per-question change in answer length; its intercept, the change
expected at no change in length, is bootstrapped the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BootstrapConfig

__all__ = ["AnalysisError", "Interval", "LengthAdjusted", "PairedBootstrap"]


class AnalysisError(RuntimeError):
    """Finished runs that cannot be analysed as configured."""


@dataclass(frozen=True)
class Interval:
    """A point estimate with its percentile bootstrap interval."""

    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class LengthAdjusted:
    """A per-question change with answer length held fixed.

    Attributes:
        observed: Mean per-question change, with no adjustment.
        mean_length_change: Mean per-question change in answer length, in tokens.
        slope: Change in the score per token of added length, fitted over the questions.
        adjusted: The fitted change at no change in length (the intercept), with its interval.
        dropped_resamples: Resamples left out because their lengths did not vary.
    """

    observed: float
    mean_length_change: float
    slope: float
    adjusted: Interval
    dropped_resamples: int


class PairedBootstrap:
    """Bootstraps per-question differences over questions (D-115).

    Args:
        cfg: Resample count, seed and coverage. Each interval restarts the generator at the
            configured seed, so an interval does not depend on the order intervals are computed.
    """

    def __init__(self, cfg: BootstrapConfig) -> None:
        self._n = cfg.n_resamples
        self._seed = cfg.seed
        tail = (1.0 - cfg.ci_level) / 2.0 * 100.0
        self._percentiles = (tail, 100.0 - tail)

    def mean(self, diffs: np.ndarray) -> Interval:
        """The mean per-question difference and its interval.

        Args:
            diffs: One difference per question.

        Returns:
            The mean, with the percentile interval of the resampled means.

        Raises:
            AnalysisError: If there are no questions.
        """
        diffs = _per_question(diffs)
        means = diffs[self._indices(len(diffs))].mean(axis=1)
        return self._interval(float(diffs.mean()), means)

    def length_adjusted(self, diffs: np.ndarray, length_diffs: np.ndarray) -> LengthAdjusted:
        """The per-question difference regressed on the per-question change in length.

        Ordinary least squares, ``diff = intercept + slope * length_diff``, refitted on every
        resample of the questions. Assumes the effect of length is linear over the observed range.

        Args:
            diffs: One score difference per question.
            length_diffs: The same questions' differences in answer length, in tokens.

        Returns:
            The unadjusted mean, the mean length change, the fitted slope, and the intercept with
            its interval.

        A resample whose length differences are all equal has no slope; it is left out of the
        interval and counted in ``dropped_resamples``.

        Raises:
            AnalysisError: If the arrays differ in length or are empty, or the length differences
                do not vary, so no slope can be fitted.
        """
        diffs, length_diffs = _per_question(diffs), _per_question(length_diffs)
        if len(diffs) != len(length_diffs):
            raise AnalysisError("score and length differences must cover the same questions")
        intercept, slope = _fit(diffs[None, :], length_diffs[None, :])
        if np.isnan(slope[0]):
            raise AnalysisError("answer length does not vary across questions, so no slope fits")
        indices = self._indices(len(diffs))
        intercepts, _ = _fit(diffs[indices], length_diffs[indices])
        fitted = intercepts[~np.isnan(intercepts)]
        return LengthAdjusted(
            observed=float(diffs.mean()),
            mean_length_change=float(length_diffs.mean()),
            slope=float(slope[0]),
            adjusted=self._interval(float(intercept[0]), fitted),
            dropped_resamples=int(len(intercepts) - len(fitted)),
        )

    def _indices(self, n: int) -> np.ndarray:
        return np.random.default_rng(self._seed).integers(0, n, size=(self._n, n))

    def _interval(self, estimate: float, resampled: np.ndarray) -> Interval:
        low, high = np.percentile(resampled, self._percentiles)
        return Interval(estimate=estimate, low=float(low), high=float(high))


def _per_question(values: np.ndarray) -> np.ndarray:
    """``values`` as a 1-D float array with at least one question."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise AnalysisError("expected one value per question, and at least one question")
    return values


def _fit(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise least-squares intercepts and slopes of ``y`` on ``x``; NaN where ``x`` is flat."""
    x_centred = x - x.mean(axis=1, keepdims=True)
    spread = (x_centred**2).sum(axis=1)
    covariance = (x_centred * (y - y.mean(axis=1, keepdims=True))).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(spread > 0, covariance / spread, np.nan)
    return y.mean(axis=1) - slope * x.mean(axis=1), slope
