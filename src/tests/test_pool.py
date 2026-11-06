"""Freezing the T5 training pool: the prompt-length filter, the seeded draw, and the frozen file.

CPU only, no network: the fixture XSum split and tiny word-level tokenizers from test_memory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from agent_evals.agents.generator import TEMPLATES
from agent_evals.config import load_config
from agent_evals.datasets import DatasetError
from agent_evals.rl.pool import PoolError, TrainingPool, draw_pool, pool_path
from test_memory import FIXTURES, TOKENIZERS, FixtureXSum, tiny_tokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
SHIPPED = REPO_ROOT / "src" / "configs" / "rl" / "grpo_xsum.yaml"
INSTRUCTION = len(TEMPLATES["tier1_xsum"].instruction.split())


class OfflinePool(TrainingPool):
    def _build_adapter(self, data):
        return FixtureXSum(data)

    def _load_tokenizers(self, cfg):
        return {name: tiny_tokenizer(template) for name, template in TOKENIZERS.items()}


def _config(tmp_path: Path, **data) -> Path:
    """A training config of three articles, capped at 14 tokens under the longer template."""
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "defaults": ["train_pool"],
                "run": {"owner": "sameer"},
                "data": {
                    "split": "train",
                    "n_samples": 3,
                    "n_strata": 1,
                    "min_source_words": 1,
                    "sample_seed": 123,
                    **data,
                },
                "pool": {"prompt_cap": 14, "backbones": list(TOKENIZERS)},
            }
        )
    )
    return path


def _fixture_lengths() -> dict[str, int]:
    """Each fixture article's longest prompt: the instruction, the words, two template tokens."""
    rows = json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8"))
    return {r["id"]: INSTRUCTION + len(r["document"].split()) + 2 for r in rows if r["document"]}


class TestDrawPool:
    def test_same_seed_same_draw_within_the_cap(self):
        ids = [f"FAKE_EXAMPLE_{i}" for i in range(20)]
        lengths = list(range(20))
        first = draw_pool(ids, lengths, cap=14, count=5, seed=123)
        assert first == draw_pool(list(reversed(ids)), list(reversed(lengths)), 14, 5, 123)
        assert len(set(first)) == 5
        assert all(int(i.rsplit("_", 1)[1]) <= 14 for i in first)

    def test_too_few_fitting_raises(self):
        with pytest.raises(PoolError, match="only 2 prompts fit"):
            draw_pool(["a", "b", "c"], [1, 2, 9], cap=5, count=3, seed=123)


class TestFreeze:
    def test_writes_a_seeded_draw_of_articles_within_the_cap(self, tmp_path):
        config = _config(tmp_path)
        drawn = OfflinePool(config, REPO_ROOT).freeze()

        lengths = _fixture_lengths()
        assert drawn == draw_pool(list(lengths), list(lengths.values()), 14, 3, 123)
        assert all(lengths[i] <= 14 for i in drawn)
        assert load_config(config).data.sample_ids == drawn
        text = pool_path(config).read_text()
        fitting = sum(n <= 14 for n in lengths.values())
        assert f"{len(lengths)} eligible, {fitting} within the cap, 3 drawn" in text

    def test_refuses_to_redraw_a_frozen_pool(self, tmp_path):
        config = _config(tmp_path)
        OfflinePool(config, REPO_ROOT).freeze()
        with pytest.raises(PoolError, match="refusing to redraw"):
            OfflinePool(config, REPO_ROOT).freeze()

    def test_evaluation_split_is_refused(self, tmp_path):
        with pytest.raises(DatasetError, match="not trainable"):
            OfflinePool(_config(tmp_path, split="test"), REPO_ROOT).freeze()


class TestShippedConfig:
    def test_training_config_replays_the_frozen_pool_with_the_approved_settings(self):
        cfg = load_config(SHIPPED, ["run.owner=sameer"])
        assert len(cfg.data.sample_ids) == cfg.data.n_samples == 10_000
        assert len(set(cfg.data.sample_ids)) == 10_000
        assert (cfg.data.split, cfg.data.sample_seed) == ("train", 123)
        assert cfg.pool.prompt_cap == 1536
        training = cfg.training
        assert (training.num_generations, training.per_device_train_batch_size) == (8, 8)
        assert (training.gradient_accumulation_steps, training.max_steps) == (16, 625)
        assert training.save_steps == 125
        assert training.force_full_completions is False
