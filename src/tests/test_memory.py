"""The GRPO memory measurement: article selection by prompt length, and the sweep's control flow.

CPU only, no network. Tokenizers are tiny word-level ones built in memory; the sweep's training
processes are replaced by scripted statuses.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from agent_evals.agents.generator import TEMPLATES
from agent_evals.config import load_config
from agent_evals.datasets import DatasetError, XSumAdapter
from agent_evals.rl.memory import (
    MeasurementArticles,
    MeasurementError,
    MemorySweep,
    PromptLengths,
    cap_config_path,
    closest_to_median,
    longest_under_cap,
    typical_config_path,
)
from agent_evals.run import STATUS_OOM, STATUS_SUCCESS

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
SHIPPED = REPO_ROOT / "src" / "configs" / "rl" / "grpo_xsum_memory.yaml"
INSTRUCTION = TEMPLATES["tier1_xsum"].instruction


class FixtureXSum(XSumAdapter):
    """XSum's adapter over the eight-row offline fixture."""

    def _load_raw(self) -> Dataset:
        return Dataset.from_list(json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8")))


def tiny_tokenizer(chat_template: str):
    """A word-level tokenizer over the fixture text: one token per whitespace-separated word."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    rows = json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8"))
    words = {w for row in rows for w in row["document"].split()} | set(INSTRUCTION.split())
    vocab = {t: i for i, t in enumerate(["<unk>", "<sys>", "<gen>", *sorted(words)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    tokenizer.chat_template = chat_template
    return tokenizer


#: Two backbones whose templates differ: "short" adds nothing, "long" adds two tokens.
TOKENIZERS = {
    "short": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    "long": "<sys> {% for m in messages %}{{ m['content'] }}{% endfor %} <gen>",
}


class OfflineArticles(MeasurementArticles):
    def _build_adapter(self):
        return FixtureXSum(self._cfg.data)

    def _load_tokenizers(self):
        return {name: tiny_tokenizer(template) for name, template in TOKENIZERS.items()}


def _base(tmp_path: Path, **changes) -> Path:
    """A small base measurement config: three articles per cap, caps in fixture-sized tokens."""
    data = {
        "run": {"owner": "sameer"},
        "name": "memory",
        "data": {"split": "train", "n_samples": 3, "n_strata": 1, "min_source_words": 1},
        "training": {
            "num_generations": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 3,
            "save_steps": 3,
        },
        "measurement": {"backbones": list(TOKENIZERS), "prompt_caps": [12, 16], "group_sizes": [2]},
    }
    for key, value in changes.items():
        data[key] = {**data[key], **value} if isinstance(value, dict) else value
    path = tmp_path / "memory.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestLongestUnderCap:
    def test_longest_first_ties_by_id_and_cap_respected(self):
        ids = ["d", "a", "c", "b", "e"]
        lengths = [9, 7, 9, 12, 8]
        assert longest_under_cap(ids, lengths, cap=10, count=3) == ("c", "d", "e")

    def test_too_few_fitting_raises(self):
        with pytest.raises(MeasurementError, match="only 1 prompts fit 8 tokens"):
            longest_under_cap(["a", "b"], [8, 9], cap=8, count=2)


class TestClosestToMedian:
    def test_median_first_then_by_distance_length_and_id(self):
        ids = ["a", "b", "c", "d", "e", "f"]
        lengths = [5, 9, 7, 12, 8, 6]
        assert closest_to_median(ids, lengths, cap=10, count=3) == (("c", "f", "e"), 7)

    def test_too_few_fitting_raises(self):
        with pytest.raises(MeasurementError, match="only 1 prompts fit 6 tokens"):
            closest_to_median(["a", "b"], [5, 9], cap=6, count=2)


class TestPromptLengths:
    def test_counts_the_chat_template_and_takes_the_longest_backbone(self, tmp_path):
        adapter = FixtureXSum(load_config(_base(tmp_path)).data)
        examples = [adapter.as_example(row) for row in adapter.eligible_pool()]
        meter = PromptLengths(
            {name: tiny_tokenizer(t) for name, t in TOKENIZERS.items()}, TEMPLATES["tier1_xsum"]
        )
        instruction = len(INSTRUCTION.split())
        short = meter.lengths("short", examples)
        assert short == [instruction + len(e.source.split()) for e in examples]
        assert meter.lengths("long", examples) == [n + 2 for n in short]
        assert meter.longest(examples) == [n + 2 for n in short]


class TestSelect:
    def test_freezes_the_longest_fitting_articles_per_cap(self, tmp_path):
        base = _base(tmp_path)
        written = OfflineArticles(load_config(base), base, REPO_ROOT).select()
        assert written == [cap_config_path(base, 12), cap_config_path(base, 16)]

        instruction = len(INSTRUCTION.split())
        rows = json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8"))
        longest = {row["id"]: instruction + len(row["document"].split()) + 2 for row in rows}
        for cap, path in zip((12, 16), written, strict=True):
            cfg = load_config(path)
            fitting = sorted(
                (i for i, n in longest.items() if n <= cap and n > instruction + 2),
                key=lambda i: (-longest[i], i),
            )
            assert cfg.name == f"memory_{cap}"
            assert cfg.data.sample_ids == tuple(fitting[:3])
            assert cfg.measurement == load_config(base).measurement

    def test_evaluation_split_is_refused(self, tmp_path):
        base = _base(tmp_path, data={"split": "test"})
        with pytest.raises(DatasetError, match="not trainable"):
            OfflineArticles(load_config(base), base, REPO_ROOT).select()

    def test_base_config_must_not_freeze_articles(self, tmp_path):
        base = _base(tmp_path, data={"sample_ids": ["FAKE_EXAMPLE_a", "FAKE_EXAMPLE_b", "x"]})
        with pytest.raises(MeasurementError, match="freezes no articles"):
            OfflineArticles(load_config(base), base, REPO_ROOT)

    def test_freezes_the_articles_closest_to_the_median(self, tmp_path):
        base = _base(tmp_path, measurement={"typical_within_cap": 16, "typical_group_size": 2})
        path = OfflineArticles(load_config(base), base, REPO_ROOT).select_typical()
        assert path == typical_config_path(base)
        # Fixture prompts here are 10 to 16 tokens; the lower median of the seven is 13.
        cfg = load_config(path)
        assert cfg.name == "memory_typical"
        assert cfg.data.sample_ids == ("FAKE_EXAMPLE_d", "FAKE_EXAMPLE_c", "FAKE_EXAMPLE_e")
        assert "median prompt (13 tokens" in path.read_text()

    def test_typical_selection_needs_the_timing_settings(self, tmp_path):
        base = _base(tmp_path)
        with pytest.raises(MeasurementError, match="typical_within_cap"):
            OfflineArticles(load_config(base), base, REPO_ROOT).select_typical()

    def test_articles_must_cover_every_step(self, tmp_path):
        base = _base(tmp_path, training={"max_steps": 4})
        with pytest.raises(MeasurementError, match="no article repeats"):
            OfflineArticles(load_config(base), base, REPO_ROOT)


class ScriptedSweep(MemorySweep):
    """The sweep with each cell's training process replaced by a scripted status."""

    def __init__(self, *args, statuses, **kwargs):
        super().__init__(*args, **kwargs)
        self._statuses = statuses
        self.cells = []
        self.extras = []

    def _run_cell(self, config, backbone, group_size, extra=()):
        self.cells.append((config.name, backbone, group_size))
        self.extras.append(tuple(extra))
        return self._statuses.get((backbone, config.name, group_size), STATUS_SUCCESS), config


class TestSweep:
    def _sweep(self, tmp_path, statuses, group_sizes=(2, 4, 8)):
        base = _base(tmp_path, measurement={"group_sizes": list(group_sizes)})
        for cap in (12, 16):
            cap_config_path(base, cap).write_text("")
        return ScriptedSweep(load_config(base), base, tmp_path, statuses=statuses)

    def test_out_of_memory_skips_larger_groups_for_that_backbone_and_cap(self, tmp_path):
        sweep = self._sweep(tmp_path, {("short", "memory_12.yaml", 4): STATUS_OOM})
        results = sweep.run()
        assert sweep.cells[:3] == [
            ("memory_12.yaml", "short", 2),
            ("memory_12.yaml", "short", 4),
            ("memory_16.yaml", "short", 2),
        ]
        assert len(results) == 2 + 3 + 3 + 3

    def test_any_other_failure_stops_the_sweep(self, tmp_path):
        sweep = self._sweep(tmp_path, {("short", "memory_12.yaml", 2): "failed"})
        with pytest.raises(MeasurementError, match="without running out of memory"):
            sweep.run()
        assert len(sweep.cells) == 1

    def test_missing_article_list_raises_before_any_cell(self, tmp_path):
        base = _base(tmp_path)
        sweep = ScriptedSweep(load_config(base), base, tmp_path, statuses={})
        with pytest.raises(MeasurementError, match="run `select` first"):
            sweep.run()
        assert sweep.cells == []

    def _typical(self, tmp_path, statuses):
        base = _base(tmp_path, measurement={"typical_within_cap": 16, "typical_group_size": 8})
        typical_config_path(base).write_text("")
        return ScriptedSweep(load_config(base), base, tmp_path, statuses=statuses)

    def test_timing_runs_one_cell_per_backbone_at_the_typical_group(self, tmp_path):
        sweep = self._typical(tmp_path, {})
        results = sweep.run_typical()
        assert sweep.cells == [
            ("memory_typical.yaml", "short", 8),
            ("memory_typical.yaml", "long", 8),
        ]
        assert [(r.prompt_cap, r.status) for r in results] == [(16, STATUS_SUCCESS)] * 2

    def test_timing_cell_that_does_not_succeed_stops_the_timing(self, tmp_path):
        sweep = self._typical(tmp_path, {("short", "memory_typical.yaml", 8): STATUS_OOM})
        with pytest.raises(MeasurementError, match="did not succeed"):
            sweep.run_typical()
        assert len(sweep.cells) == 1

    def test_timing_needs_its_article_list_and_settings(self, tmp_path):
        base = _base(tmp_path, measurement={"typical_within_cap": 16, "typical_group_size": 8})
        with pytest.raises(MeasurementError, match="select-typical"):
            ScriptedSweep(load_config(base), base, tmp_path, statuses={}).run_typical()
        other = tmp_path / "plain"
        other.mkdir()
        unset = _base(other)
        with pytest.raises(MeasurementError, match="typical_group_size"):
            ScriptedSweep(load_config(unset), unset, other, statuses={}).run_typical()

    def _forced(self, tmp_path, statuses):
        grid = {"typical_within_cap": 16, "typical_group_size": 8, "forced_cap": 16}
        base = _base(tmp_path, measurement=grid)
        cap_config_path(base, 16).write_text("")
        return ScriptedSweep(load_config(base), base, tmp_path, statuses=statuses)

    def test_forced_cells_force_full_completions_and_name_the_runs(self, tmp_path):
        sweep = self._forced(tmp_path, {})
        sweep.run_forced()
        assert sweep.cells == [("memory_16.yaml", "short", 8), ("memory_16.yaml", "long", 8)]
        assert set(sweep.extras) == {
            ("training.force_full_completions=true", "name=memory_16_forced")
        }

    def test_forced_out_of_memory_is_a_result_not_a_stop(self, tmp_path):
        sweep = self._forced(tmp_path, {("short", "memory_16.yaml", 8): STATUS_OOM})
        results = sweep.run_forced()
        assert [r.status for r in results] == [STATUS_OOM, STATUS_SUCCESS]

    def test_forced_failure_stops_the_sweep(self, tmp_path):
        sweep = self._forced(tmp_path, {("short", "memory_16.yaml", 8): "failed"})
        with pytest.raises(MeasurementError, match="without running out of memory"):
            sweep.run_forced()

    def test_forced_cells_need_their_setting(self, tmp_path):
        base = _base(tmp_path)
        with pytest.raises(MeasurementError, match="forced_cap"):
            ScriptedSweep(load_config(base), base, tmp_path, statuses={}).run_forced()

    def test_cell_command_sets_the_group_and_passes_overrides(self, tmp_path):
        base = _base(tmp_path)
        sweep = MemorySweep(load_config(base), base, tmp_path, ["run.owner=sameer"])
        command = sweep._command(cap_config_path(base, 12), "short", 4, tmp_path / "adapters")
        assert command[1] == str(tmp_path / "src" / "run_training.py")
        assert command[2:6] == ["--config", str(cap_config_path(base, 12)), "--backbone", "short"]
        sets = [command[i + 1] for i, arg in enumerate(command) if arg == "--set"]
        assert sets == [
            "run.owner=sameer",
            "training.num_generations=4",
            "training.per_device_train_batch_size=4",
            f"training.adapter_root={tmp_path / 'adapters'}",
        ]


class TestShippedConfig:
    def test_base_config_resolves_with_the_approved_settings(self):
        cfg = load_config(SHIPPED, ["run.owner=sameer"])
        assert cfg.data.split == "train" and cfg.data.exclude_sources_from_split == "test"
        assert cfg.training.loss_type == "dr_grpo" and cfg.training.max_steps == 10
        assert cfg.measurement.prompt_caps == (1024, 1536, 2048)
        assert cfg.measurement.group_sizes == (2, 4, 8)

    def test_each_cap_freezes_one_article_per_step(self):
        base = load_config(SHIPPED, ["run.owner=sameer"])
        for cap in base.measurement.prompt_caps:
            cfg = load_config(cap_config_path(SHIPPED, cap), ["run.owner=sameer"])
            assert cfg.name == f"grpo_xsum_memory_{cap}"
            assert len(cfg.data.sample_ids) == cfg.training.max_steps

    def test_typical_list_freezes_one_article_per_step(self):
        cfg = load_config(typical_config_path(SHIPPED), ["run.owner=sameer"])
        assert cfg.name == "grpo_xsum_memory_typical"
        assert len(cfg.data.sample_ids) == cfg.training.max_steps
        assert (cfg.measurement.typical_within_cap, cfg.measurement.typical_group_size) == (1536, 8)
        assert cfg.measurement.forced_cap == 1536
