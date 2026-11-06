"""Agent-Evals: an evaluation harness for agentic LLM systems (formerly FactAgent-Bench).

Evaluates agent architectures -- single-agent, Self-Refine, Critique-Revise with tool use, and
RL-trained agents -- on outcome, trajectory, efficiency and reliability, using factual
consistency on grounded generation tasks as the measured outcome.

The package is organised around six extension points, each an abstract base class with a
registry. All experiment parameters live in ``src/configs/``; every run writes a full
``results/<run_id>/``.
"""

from __future__ import annotations

from .agents import Agent, AgentError
from .config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    RunConfig,
    SamplingConfig,
    SeedingConfig,
    ServingConfig,
    load_config,
    load_group,
    to_dict,
)
from .datasets import DatasetAdapter, DatasetError
from .logging_utils import configure_logging, get_logger
from .metrics import Metric, MetricError
from .registry import (
    AGENTS,
    DATASETS,
    METRICS,
    REWARDS,
    SERVING,
    TOPOLOGIES,
    Registry,
    RegistryError,
)
from .rl import Reward, RewardError
from .run import MetricsWriter, RunDirectory, make_run_id
from .seeding import set_seed
from .serving import ChatClient, Completion, ServingError, get_client
from .topologies import Topology, TopologyError
from .types import AgentSet, AgentState, Example, Message, TokenCost, TopologyResult

__all__ = [
    "AGENTS",
    "Agent",
    "AgentError",
    "AgentSet",
    "AgentState",
    "ChatClient",
    "Completion",
    "ConfigError",
    "DATASETS",
    "DataConfig",
    "DatasetAdapter",
    "DatasetError",
    "Example",
    "ExperimentConfig",
    "LoggingConfig",
    "METRICS",
    "Message",
    "Metric",
    "MetricError",
    "MetricsWriter",
    "REWARDS",
    "Registry",
    "RegistryError",
    "Reward",
    "RewardError",
    "RunConfig",
    "RunDirectory",
    "SERVING",
    "SamplingConfig",
    "SeedingConfig",
    "ServingConfig",
    "ServingError",
    "TOPOLOGIES",
    "Topology",
    "TopologyError",
    "TokenCost",
    "TopologyResult",
    "configure_logging",
    "get_client",
    "get_logger",
    "load_config",
    "load_group",
    "make_run_id",
    "set_seed",
    "to_dict",
]
