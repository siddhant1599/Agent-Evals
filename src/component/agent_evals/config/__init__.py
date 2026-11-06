"""Configuration: frozen dataclasses plus the only config parser in the package.

All experiment parameters live in ``src/configs/*.yaml`` and compose from ``base.yaml`` via a
``defaults:`` list. YAML loads into frozen dataclasses -- typed, validated, immutable -- so that
``config.resolved.yaml`` written into a run directory is enough to reproduce the run on its own.
Nothing mutates a config mid-run, because nothing can.

A misspelled key raises. Silently falling back to a default is how a sweep runs for six hours at
settings nobody chose.

Only the config groups whose modules exist are defined here. Backbone, dataset, topology, metric
and reward groups arrive with their modules, each carrying the parameter table that was approved
for it.
"""

from .analysis import AnalysisConfig, BootstrapConfig
from .data import DataConfig, SourceFile
from .experiment import ExperimentConfig, LoggingConfig, RunConfig, SeedingConfig
from .loading import apply_overrides, load_config, load_group, to_dict
from .measurement import MeasurementConfig
from .metrics import (
    AlignScoreConfig,
    BertScoreConfig,
    CitationConfig,
    FactCCConfig,
    MetricsConfig,
    MetricValidationConfig,
    MiniCheckConfig,
    RougeLConfig,
    SummaCConfig,
    WorkedExampleOverlapConfig,
)
from .pool import PoolConfig
from .reward import RewardConfig
from .serving import SamplingConfig, ServingConfig
from .topology import AgentConfig, TopologyConfig
from .training import TrainingConfig
from .validation import ConfigError

__all__ = [
    "AgentConfig",
    "AlignScoreConfig",
    "AnalysisConfig",
    "BertScoreConfig",
    "BootstrapConfig",
    "CitationConfig",
    "ConfigError",
    "DataConfig",
    "SourceFile",
    "FactCCConfig",
    "RunConfig",
    "LoggingConfig",
    "MeasurementConfig",
    "MetricValidationConfig",
    "MetricsConfig",
    "MiniCheckConfig",
    "PoolConfig",
    "RewardConfig",
    "RougeLConfig",
    "SamplingConfig",
    "SeedingConfig",
    "ServingConfig",
    "SummaCConfig",
    "TopologyConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "load_config",
    "load_group",
    "apply_overrides",
    "to_dict",
    "WorkedExampleOverlapConfig",
]
