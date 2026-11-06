"""The root config and the per-run groups: identity, logging and seeding."""

from __future__ import annotations

from dataclasses import dataclass, field

from .analysis import AnalysisConfig
from .data import DataConfig
from .measurement import MeasurementConfig
from .metrics import MetricsConfig
from .pool import PoolConfig
from .reward import RewardConfig
from .serving import SamplingConfig, ServingConfig
from .topology import TopologyConfig
from .training import TrainingConfig
from .validation import ConfigError

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Per-run identity and output location.

    Attributes:
        owner: Who is running this. Required, no default -- an unattributed run must not be
            possible to start. Validated against ``known_owners``.
        seed: This run's seed. The reporting protocol is three seeds (0, 1, 2) with mean and
            standard deviation; the sweep driver supplies one per run, so a single run holds a
            single seed here. Safe range: any non-negative int.
        output_root: Where run directories are created. Committed to git (D-031).
        known_owners: Accepted values for ``owner``. An unknown name raises rather than being
            recorded, so a typo or an agent-invented value fails loudly.
    """

    owner: str
    seed: int = 0
    output_root: str = "results"
    known_owners: tuple[str, ...] = ("sameer", "kartik", "siddhanth")

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ConfigError(f"run.seed must be non-negative, got {self.seed}")
        if self.owner not in self.known_owners:
            allowed = ", ".join(self.known_owners)
            raise ConfigError(
                f"run.owner {self.owner!r} is not a known owner. Allowed: {allowed}. "
                f"Add yourself to run.known_owners if this is a new team member."
            )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Log destinations and verbosity.

    Attributes:
        console_level: Level for the stream a human is watching. Safe range: any level name.
        file_level: Level for ``run.log``. Kept more verbose than the console: the log file is
            the forensic record for a run that failed unattended.
        flush_metrics_every: Flush ``metrics.csv`` every N records. 1 means every record, so a
            crashed or OOM run keeps its partial evidence -- and those are exactly the runs the
            evidence is needed for. Safe range: >= 1.
    """

    console_level: str = "INFO"
    file_level: str = "DEBUG"
    flush_metrics_every: int = 1

    def __post_init__(self) -> None:
        for name, level in (("console_level", self.console_level), ("file_level", self.file_level)):
            if level.upper() not in _LOG_LEVELS:
                raise ConfigError(
                    f"logging.{name} must be one of {', '.join(_LOG_LEVELS)}, got {level!r}"
                )
        if self.flush_metrics_every < 1:
            raise ConfigError(
                f"logging.flush_metrics_every must be >= 1, got {self.flush_metrics_every}"
            )


@dataclass(frozen=True, slots=True)
class SeedingConfig:
    """Determinism settings.

    Attributes:
        deterministic: Whether to request bitwise-deterministic kernels via
            ``torch.use_deterministic_algorithms``. Default off, pending a team decision: it
            does nothing for vLLM-served inference, which runs in another process, and may raise
            rather than merely slow down on 4-bit QLoRA kernels. Enabling it also needs
            ``CUBLAS_WORKSPACE_CONFIG`` set before the process starts, which is a shellscript
            concern, not a Python one.
    """

    deterministic: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """The root config. Everything a run needs, fully resolved.

    Attributes:
        name: Short config name, used in the run id.
        run: Per-run identity and output location.
        logging: Log destinations and verbosity.
        seeding: Determinism settings.
        sampling: Generation limits shared by every topology and backbone.
        serving: How backbones are served and reached.
        metrics: Every metric's configuration.
        topology: Topology parameters, from ``src/configs/topology/<name>.yaml`` when the
            topology has that file; None for a topology that binds only its name.
        data: Dataset configuration, or None for infrastructure-only runs.
        reward: GRPO reward configuration, or None for runs that do not train.
        training: GRPO training configuration, or None for runs that do not train.
        measurement: The GRPO memory measurement grid, or None outside that measurement.
        pool: The training pool's prompt-length filter, or None for runs that do not freeze one.
        analysis: The finished runs an analysis reads, or None for runs that are not analyses.
    """

    run: RunConfig
    name: str = "base"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seeding: SeedingConfig = field(default_factory=SeedingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    topology: TopologyConfig | None = None
    data: DataConfig | None = None
    reward: RewardConfig | None = None
    training: TrainingConfig | None = None
    measurement: MeasurementConfig | None = None
    pool: PoolConfig | None = None
    analysis: AnalysisConfig | None = None
