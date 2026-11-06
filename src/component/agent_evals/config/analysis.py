"""Analyses of finished runs: which runs are read and how gains are bootstrapped (D-115)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .validation import ConfigError


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """The percentile bootstrap behind every reported confidence interval (D-115).

    Attributes:
        n_resamples: Resamples of the per-question values. More only reduce the Monte Carlo noise
            in the interval ends. Safe range: >= 1; D-115 fixes 10,000.
        seed: Seed of the resampling generator, restarted for each interval so an interval does
            not depend on which others were computed before it. Safe range: >= 0; D-115 fixes 123.
        ci_level: Coverage of the interval. Safe range: (0, 1); D-115 fixes 0.95.
    """

    n_resamples: int = 10_000
    seed: int = 123
    ci_level: float = 0.95

    def __post_init__(self) -> None:
        if type(self.n_resamples) is not int or self.n_resamples < 1:
            raise ConfigError(
                f"analysis.bootstrap.n_resamples must be an integer >= 1, got {self.n_resamples!r}"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigError(f"analysis.bootstrap.seed must be an integer >= 0, got {self.seed!r}")
        if not 0.0 < self.ci_level < 1.0:
            raise ConfigError(
                f"analysis.bootstrap.ci_level must lie in (0, 1), got {self.ci_level!r}"
            )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Which finished runs an analysis reads, and what it reports about them.

    Every run is compared with the ``baseline`` runs of the same backbone, question by question,
    with seeds averaged within each question first (D-115).

    Attributes:
        task: The task every source run must have answered, e.g. ``tier2_asqa``. Required.
        runs: Run ids under ``run.output_root``, baseline runs included. Required; no repeats.
        metrics: ``metrics.csv`` keys whose paired gain over the baseline is reported. Required.
        length_metrics: Keys among ``metrics`` also reported with answer length held fixed: the
            per-question change regressed on the per-question change in answer length, in tokens.
            Empty reports none.
        baseline: Topology every other topology is compared with (D-115: T0).
        bootstrap: The confidence-interval settings (D-115).
    """

    task: str
    runs: tuple[str, ...]
    metrics: tuple[str, ...]
    length_metrics: tuple[str, ...] = ()
    baseline: str = "t0_single_agent"
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)

    def __post_init__(self) -> None:
        if not self.task:
            raise ConfigError("analysis.task must name a task")
        for name, values in (("runs", self.runs), ("metrics", self.metrics)):
            if not values or any(not isinstance(v, str) or not v for v in values):
                raise ConfigError(f"analysis.{name} must be a non-empty list of names")
            if len(set(values)) != len(values):
                raise ConfigError(f"analysis.{name} must not repeat a name")
        extra = sorted(set(self.length_metrics) - set(self.metrics))
        if extra:
            raise ConfigError(f"analysis.length_metrics must be among analysis.metrics: {extra}")
