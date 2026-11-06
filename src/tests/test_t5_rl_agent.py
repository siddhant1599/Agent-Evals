"""T5: the adapter config, and the topology that is T0's single call (D-111)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from agent_evals.agents import Generator
from agent_evals.config import (
    AgentConfig,
    ConfigError,
    SamplingConfig,
    ServingConfig,
    TopologyConfig,
    load_config,
)
from agent_evals.registry import TOPOLOGIES
from agent_evals.serving import ChatClient, Completion
from agent_evals.serving.backbones import BackboneSpec
from agent_evals.topologies import RLAgent, SingleAgent
from agent_evals.types import Example, TokenCost

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "src" / "configs"
SOURCE = "The council voted to close the library in March."
BACKBONE = BackboneSpec(
    name="b",
    hf_path="org/b",
    hf_revision="a" * 40,
    gated=False,
    quantization="bitsandbytes",
    max_model_len=2809,
)


class RecordingClient(ChatClient):
    def __init__(self):
        super().__init__(ServingConfig(), SamplingConfig(), BACKBONE)
        self.calls = []

    def complete(self, prompt, seed=None, system=None, turns=()):
        self.calls.append((prompt, seed, system, tuple(turns)))
        return Completion("A summary.", TokenCost(10, 2), 0.1, "stop")


def _example():
    return Example(
        id="x", tier=1, task="summarization", source=SOURCE, evidence=SOURCE, reference="r"
    )


def _run(topology_cls, name):
    client = RecordingClient()
    generator = Generator(AgentConfig(dataset="tier1_xsum", seed=4), client)
    result = topology_cls(TopologyConfig(name=name)).run(_example(), {"generator": generator})
    return client.calls, result


class TestAdapterConfig:
    def test_adapter_for_finds_the_backbone_entry(self):
        serving = ServingConfig(lora_adapters=("a=/x", " b = ~/y "))
        assert serving.adapter_for("a") == "/x"
        assert serving.adapter_for("b") == "~/y"
        assert serving.adapter_for("c") is None

    def test_no_adapters_by_default(self):
        assert ServingConfig().lora_adapters == ()
        assert ServingConfig().adapter_for("a") is None

    @pytest.mark.parametrize("entry", ["a", "=x", "a=", " = "])
    def test_malformed_entries_are_refused(self, entry):
        with pytest.raises(ConfigError, match="backbone=directory"):
            ServingConfig(lora_adapters=(entry,))

    def test_a_backbone_named_twice_is_refused(self):
        with pytest.raises(ConfigError, match="twice"):
            ServingConfig(lora_adapters=("a=/x", "a=/y"))

    @pytest.mark.parametrize("rank", [0, -1, 1.5, True])
    def test_max_lora_rank_must_be_a_positive_integer(self, rank):
        with pytest.raises(ConfigError, match="max_lora_rank"):
            ServingConfig(max_lora_rank=rank)

    def test_gpu_memory_utilization_defaults_to_vllm_and_is_validated(self):
        assert ServingConfig().gpu_memory_utilization is None
        assert ServingConfig(gpu_memory_utilization=0.85).gpu_memory_utilization == 0.85
        for bad in (0, -0.1, 1.5, True):
            with pytest.raises(ConfigError, match="gpu_memory_utilization"):
                ServingConfig(gpu_memory_utilization=bad)

    def test_requires_adapter_is_off_by_default_and_must_be_boolean(self):
        assert TopologyConfig(name="t0_single_agent").requires_adapter is False
        with pytest.raises(ConfigError, match="requires_adapter"):
            TopologyConfig(name="t5_rl_agent", requires_adapter="yes")


class TestT5Config:
    def test_t5_config_names_an_adapter_for_every_backbone_and_requires_one(self):
        cfg = load_config(
            CONFIGS / "data" / "tier1_xsum.yaml",
            ["run.owner=sameer"],
            extra_paths=[CONFIGS / "topology" / "t5_rl_agent.yaml"],
        )
        assert cfg.topology == TopologyConfig(name="t5_rl_agent", requires_adapter=True)
        assert cfg.serving.max_lora_rank == 16
        assert cfg.serving.gpu_memory_utilization == 0.85
        with (REPO_ROOT / "src" / "model_registry.csv").open() as handle:
            names = [row["name"] for row in csv.DictReader(handle)]
        for name in names:
            path = cfg.serving.adapter_for(name)
            assert path is not None and path.endswith("/checkpoint-625"), name


class TestRLAgent:
    def test_registered_under_its_own_name(self):
        assert TOPOLOGIES.get("t5_rl_agent") is RLAgent
        assert RLAgent.ROLES == ("generator",)

    def test_sends_exactly_t0_s_call(self):
        t5_calls, t5 = _run(RLAgent, "t5_rl_agent")
        t0_calls, _ = _run(SingleAgent, "t0_single_agent")
        assert t5_calls == t0_calls
        assert t5_calls == [
            (f"Summarize the following article in one sentence:\n\n{SOURCE}", 4, None, ())
        ]
        assert t5.output == "A summary." and t5.n_rounds == 1 and t5.converged is True
        assert t5.token_cost == TokenCost(10, 2) and t5.finish_reason == "stop"
