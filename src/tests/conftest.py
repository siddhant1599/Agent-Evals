"""Shared fixtures. CPU-only, no network, no GPU, no vLLM server."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_evals import ExperimentConfig, RunConfig
from agent_evals.logging_utils import reset_logging
from agent_evals.types import Message, TokenCost, TopologyResult

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _clean_logging():
    """Detach log handlers around every test so tmp dirs can be removed and logs do not leak."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def base_config_path() -> Path:
    """The real src/configs/base.yaml, so the tests exercise the shipped defaults."""
    return CONFIGS_DIR / "base.yaml"


@pytest.fixture
def cfg() -> ExperimentConfig:
    """A minimal valid config with an owner supplied."""
    return ExperimentConfig(name="unittest", run=RunConfig(owner="sameer", seed=0))


@pytest.fixture
def result() -> TopologyResult:
    """A well-formed single-pass result, shaped as T0 will return one."""
    message = Message(
        role="generator",
        content="a summary",
        round_index=0,
        token_cost=TokenCost(prompt_tokens=800, completion_tokens=60),
        latency_s=1.25,
    )
    return TopologyResult(
        output="a summary",
        token_cost=message.token_cost,
        wall_clock_s=1.25,
        n_rounds=1,
        converged=True,
        trace=(message,),
        finish_reason=message.finish_reason,
    )
