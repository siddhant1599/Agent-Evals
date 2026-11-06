"""T5 GRPO training: the tier gate, the TRL wiring, the step log, and two real GRPO steps.

CPU only, no network. The end-to-end test trains a tiny random Qwen2-shaped model built in memory,
with a word-level tokenizer built from the fixture text: it checks the plumbing, not quality.
The real 4-bit backbone is exercised only by the gpu-marked test at the bottom.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from datasets import Dataset
from agent_evals.agents.generator import TEMPLATES
from agent_evals.config import (
    DataConfig,
    ExperimentConfig,
    MetricsConfig,
    RewardConfig,
    RunConfig,
    SamplingConfig,
    ServingConfig,
    TrainingConfig,
    load_config,
)
from agent_evals.datasets import CnnDailyMailAdapter, DatasetError, XSumAdapter
from agent_evals.metrics import Metric
from agent_evals.rl import trainer as trainer_module
from agent_evals.rl.trainer import (
    STEP_COLUMNS,
    GRPOTrainingRun,
    StepMetricsCallback,
    TrainingError,
    grpo_args,
    latest_checkpoint,
    lora_config,
    main,
    training_dataset,
)
from agent_evals.run import MetricsWriter

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
REGISTRY = REPO_ROOT / "src" / "model_registry.csv"
BACKBONE = "qwen2.5-7b-instruct"

#: Smoke-test values for the five settings not decided yet (Q-004, Q-011): the smallest run that
#: still forms a group. Test values only, not proposals. Everything else is the approved default.
SMOKE_UNDECIDED = {
    "num_generations": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_steps": 2,
    "save_steps": 2,
}

#: The two approved defaults that need CUDA (paged AdamW is a bitsandbytes optimizer; the 4-bit
#: base computes in bf16), replaced so the CPU test runs every other approved setting (D-054).
CPU_ONLY = {"optim": "adamw_torch", "compute_dtype": "float32"}


class FixtureXSum(XSumAdapter):
    """XSum's adapter over the eight-row offline fixture."""

    def _load_raw(self) -> Dataset:
        return Dataset.from_list(json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8")))


class FixtureTier3(FixtureXSum):
    """Keeps XSum's trainable declarations, so only the Tier 3 veto can refuse it."""

    TIER = 3


class FixtureCnnDm(CnnDailyMailAdapter):
    def _load_raw(self) -> Dataset:
        return Dataset.from_list(json.loads((FIXTURES / "cnndm.json").read_text(encoding="utf-8")))


class StubAlignScore(Metric):
    """Emits the ``alignscore`` key without a model: a fixed score per non-blank completion."""

    KEYS = ("alignscore",)

    def __init__(self) -> None:
        super().__init__(MetricsConfig(device="cpu"))

    def _score_examples(self, predictions, references, evidence, gold):
        return [{"alignscore": min(len(p.split()), 10) / 10} for p in predictions]


def tiny_model_and_tokenizer():
    """A random two-layer Qwen2-shaped model and a word-level tokenizer over the fixture text."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    rows = json.loads((FIXTURES / "xsum.json").read_text(encoding="utf-8"))
    words = {w for row in rows for w in (row["document"] + " " + row["summary"]).split()}
    words.update(TEMPLATES["tier1_xsum"].instruction.split())
    vocab = {token: i for i, token in enumerate(["<pad>", "<eos>", "<unk>", *sorted(words)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
    config = Qwen2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config), tokenizer


class OfflineRun(GRPOTrainingRun):
    """The real run with fixture data, a stub metric and a tiny model swapped in."""

    adapter_class = FixtureXSum

    def _build_adapter(self):
        return self.adapter_class(self._cfg.data)

    def _build_metric(self, name):
        return StubAlignScore()

    def _load_model(self, backbone):
        self.model_loaded = True
        return tiny_model_and_tokenizer()


def _data(**changes) -> DataConfig:
    cfg = DataConfig(split="train", n_samples=4, n_strata=1, min_source_words=1)
    cfg = dataclasses.replace(cfg, **changes)
    if cfg.sample_ids is None and cfg.name == "tier1_xsum":
        cfg = dataclasses.replace(cfg, sample_ids=FixtureXSum(cfg).draw_sample_ids())
    return cfg


def _config(tmp_path: Path, data: DataConfig | None = None, **training) -> ExperimentConfig:
    return ExperimentConfig(
        name="unittest",
        run=RunConfig(owner="sameer", seed=0),
        sampling=SamplingConfig(max_tokens=8),
        serving=dataclasses.replace(ServingConfig(), model_registry=str(REGISTRY)),
        metrics=MetricsConfig(device="cpu"),
        data=data or _data(),
        reward=RewardConfig(),
        training=TrainingConfig(
            **{**SMOKE_UNDECIDED, **CPU_ONLY, **training}, adapter_root=str(tmp_path / "adapters")
        ),
    )


class TestTierGate:
    """AGENTS.md §8: the RL entrypoint refuses a Tier 3 split, before any model loads."""

    def _refused(self, tmp_path, run, match):
        run.model_loaded = False
        with pytest.raises(DatasetError, match=match):
            run.run()
        assert run.model_loaded is False
        assert not (tmp_path / "results").exists()

    def test_tier3_is_refused(self, tmp_path):
        run = OfflineRun(_config(tmp_path), BACKBONE, tmp_path)
        run.adapter_class = FixtureTier3
        self._refused(tmp_path, run, "Tier 3 is never trainable")

    def test_evaluation_split_is_refused(self, tmp_path):
        cfg = _config(tmp_path, _data(split="test"))
        self._refused(tmp_path, OfflineRun(cfg, BACKBONE, tmp_path), "not trainable")

    def test_evaluation_only_dataset_is_refused(self, tmp_path):
        data = DataConfig(name="tier1_cnndm", split="train", n_samples=1, n_strata=1)
        cfg = _config(tmp_path, data)
        run = OfflineRun(cfg, BACKBONE, tmp_path)
        run.adapter_class = FixtureCnnDm
        self._refused(tmp_path, run, "not trainable")

    def test_missing_training_group_is_refused(self, tmp_path):
        cfg = dataclasses.replace(_config(tmp_path), training=None)
        with pytest.raises(TrainingError, match="training"):
            GRPOTrainingRun(cfg, BACKBONE, tmp_path)


class TestTrlWiring:
    def test_dataset_uses_the_t0_prompt_and_aligned_columns(self, tmp_path):
        cfg = _config(tmp_path)
        adapter = FixtureXSum(cfg.data)
        examples = [adapter.as_example(row) for row in adapter.build()]
        dataset = training_dataset(examples, TEMPLATES["tier1_xsum"])
        assert dataset.column_names == ["prompt", "evidence", "reference", "gold", "id"]
        for row, example in zip(dataset, examples, strict=True):
            content = TEMPLATES["tier1_xsum"].render(example)
            assert row["prompt"] == [{"role": "user", "content": content}]
            assert row["evidence"] == example.evidence
            assert row["reference"] == example.reference
            assert row["gold"] is None
            assert row["id"] == example.id

    def test_completions_are_generated_one_forward_pass_at_a_time(self, tmp_path):
        cfg = _config(tmp_path)
        cfg = dataclasses.replace(
            cfg, training=dataclasses.replace(cfg.training, gradient_accumulation_steps=4)
        )
        args = grpo_args(cfg, tmp_path / "out")
        assert args.generation_batch_size == cfg.training.per_device_train_batch_size
        assert args.steps_per_generation == 1

    def test_grpo_args_take_every_value_from_the_config(self, tmp_path):
        cfg = dataclasses.replace(_config(tmp_path), run=RunConfig(owner="sameer", seed=2))
        args = grpo_args(cfg, tmp_path / "out")
        for name in (
            "beta",
            "loss_type",
            "scale_rewards",
            "epsilon",
            "num_iterations",
            "num_generations",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "warmup_steps",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_steps",
            "save_steps",
            "max_grad_norm",
            "optim",
            "gradient_checkpointing",
            "mask_truncated_completions",
        ):
            assert getattr(args, name) == getattr(cfg.training, name), name
        assert args.lr_scheduler_type == cfg.training.lr_scheduler_type
        assert args.seed == 2
        assert args.max_completion_length == cfg.sampling.max_tokens
        assert args.logging_steps == 1
        assert args.use_vllm is False
        assert args.bf16 is False and args.fp16 is False

    def test_lora_targets_all_linear_layers(self, tmp_path):
        lora = lora_config(_config(tmp_path).training)
        assert lora.target_modules == "all-linear"
        assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (16, 32, 0.0)


class TestStepMetricsCallback:
    def _log(self, step, **logs):
        return SimpleNamespace(global_step=step), logs

    def test_writes_one_row_per_step_with_fixed_columns(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        callback = StepMetricsCallback(writer)
        state, logs = self._log(1, reward=0.5, kl=0.01, loss=0.2, extra_key=3.0)
        callback.on_log(None, state, None, logs=logs)
        writer.close()
        with (tmp_path / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert tuple(rows[0]) == STEP_COLUMNS
        assert rows[0]["step"] == "1" and rows[0]["reward"] == "0.5" and rows[0]["kl"] == "0.01"
        assert callback.last_row["loss"] == 0.2

    def test_summary_log_without_reward_is_skipped(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        state, logs = self._log(2, train_runtime=1.0)
        StepMetricsCallback(writer).on_log(None, state, None, logs=logs)
        assert writer.count == 0

    def test_step_without_kl_raises(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        state, logs = self._log(1, reward=0.5)
        with pytest.raises(TrainingError, match="kl"):
            StepMetricsCallback(writer).on_log(None, state, None, logs=logs)


@pytest.fixture
def cpu_only(monkeypatch):
    """Train on the CPU even where a GPU exists, so the suite behaves the same here and in CI."""
    real = trainer_module.GRPOConfig
    monkeypatch.setattr(trainer_module, "GRPOConfig", lambda **kwargs: real(**kwargs, use_cpu=True))


class TestEndToEnd:
    """AGENTS.md §8: two GRPO steps on a stub model, with every run file written."""

    def test_two_grpo_steps_write_a_complete_run(self, tmp_path, cpu_only):
        path = OfflineRun(_config(tmp_path), BACKBONE, tmp_path).run()

        assert sorted(p.name for p in path.iterdir()) == [
            "config.resolved.yaml",
            "manifest.json",
            "metrics.csv",
            "run.log",
            "summary.json",
            "trace.jsonl",
        ]
        summary = json.loads((path / "summary.json").read_text())
        assert summary["status"] == "success", summary["error"]
        assert summary["steps_completed"] == 2
        assert summary["training_gpu_hours"] == 0
        adapter = Path(summary["adapter_path"])
        assert (adapter / "adapter_config.json").is_file()
        assert adapter.is_relative_to(tmp_path / "adapters")

        with (path / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["step"] for row in rows] == ["1", "2"]
        assert all(row["reward"] != "" and row["kl"] != "" for row in rows)

        assert path.name.endswith(f"_unittest_{BACKBONE}_seed0")

        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["prompt_templates"] == {
            "generator": {"xsum_one_sentence": "xsum_one_sentence@v1"}
        }
        assert manifest["reward"] == {"name": "alignscore", "metric": "alignscore"}
        assert manifest["grpo_args"]["beta"] == 0.04
        assert manifest["grpo_args"]["loss_type"] == "dr_grpo"
        assert manifest["split"] == "train" and manifest["n_examples"] == 4


class TestForcedCompletions:
    def test_forcing_maps_to_a_minimum_length_at_the_cap(self, tmp_path):
        cfg = _config(tmp_path, force_full_completions=True)
        assert grpo_args(cfg, tmp_path).generation_kwargs == {"min_new_tokens": 8}
        assert grpo_args(_config(tmp_path), tmp_path).generation_kwargs is None

    def test_every_completion_runs_to_the_cap(self, tmp_path, cpu_only):
        path = OfflineRun(_config(tmp_path, force_full_completions=True), BACKBONE, tmp_path).run()
        with (path / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        # Every completion is exactly the cap long. (TRL's clipped ratio is not checked: it also
        # counts a final padding token as an ending, and this tiny vocabulary can sample one.)
        assert {float(row["completions/min_length"]) for row in rows} == {8.0}
        assert {float(row["completions/max_length"]) for row in rows} == {8.0}


class TestResume:
    """D-059: an interrupted run continues from its last checkpoint in a new run directory."""

    def test_continues_from_the_last_checkpoint_without_touching_the_first_run(
        self, tmp_path, cpu_only, monkeypatch
    ):
        first = OfflineRun(_config(tmp_path, save_steps=1), BACKBONE, tmp_path).run()
        before = {f.name: f.read_bytes() for f in first.iterdir()}
        adapter_dir = Path(json.loads((first / "manifest.json").read_text())["adapter_dir"])
        shutil.rmtree(adapter_dir / "checkpoint-2")  # the run "stopped" after step 1

        # A continuation starts later than its run: give it a clock a minute on, so the two run
        # ids differ as they would in use.
        later = SimpleNamespace(now=lambda: datetime.now() + timedelta(minutes=1))
        monkeypatch.setattr(trainer_module, "datetime", later)
        resumed = OfflineRun.resuming(first, tmp_path)
        assert resumed.config.name == "unittest"
        path = resumed.run()

        assert path != first
        assert {f.name: f.read_bytes() for f in first.iterdir()} == before
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["resumed_from"] == first.name
        assert manifest["resumed_checkpoint"] == str(adapter_dir / "checkpoint-1")
        assert manifest["adapter_dir"] == str(adapter_dir)
        with (path / "metrics.csv").open() as handle:
            assert [row["step"] for row in csv.DictReader(handle)] == ["2"]
        summary = json.loads((path / "summary.json").read_text())
        assert summary["status"] == "success" and summary["steps_completed"] == 2
        assert latest_checkpoint(adapter_dir).name == "checkpoint-2"

        again = OfflineRun.resuming(path, tmp_path)
        assert again._resume.run_id == first.name

    def test_no_checkpoint_raises(self, tmp_path):
        with pytest.raises(TrainingError, match="no checkpoint"):
            latest_checkpoint(tmp_path)

    def test_resume_takes_no_other_option(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--resume", str(tmp_path), "--backbone", BACKBONE], tmp_path)


class TestMain:
    def test_config_without_training_group_exits_non_zero(self, tmp_path):
        config = tmp_path / "train.yaml"
        config.write_text(yaml.safe_dump({"run": {"owner": "sameer"}}))
        assert main(["--config", str(config), "--backbone", BACKBONE], tmp_path) == 1


@pytest.mark.gpu
def test_real_backbone_runs_two_steps(tmp_path):
    """Smoke test of the real path: the 4-bit backbone, AlignScore on the GPU, paged AdamW, bf16.

    Every approved default (D-054) with the smoke-test group, batch and step count, on the
    fixture rows, whose articles are a few words long. It shows the pieces work together; its
    memory and speed are not the Q-004 measurement, which needs real article lengths.
    """
    cfg = load_config(
        REPO_ROOT / "src" / "configs" / "base.yaml",
        ["run.owner=sameer", f"serving.model_registry={REGISTRY}"],
    )
    cfg = dataclasses.replace(
        cfg,
        data=_data(),
        reward=RewardConfig(),
        training=TrainingConfig(**SMOKE_UNDECIDED, adapter_root=str(tmp_path / "adapters")),
    )

    class FixtureDataRun(GRPOTrainingRun):
        def _build_adapter(self):
            return FixtureXSum(self._cfg.data)

    path = FixtureDataRun(cfg, BACKBONE, tmp_path).run()
    summary = json.loads((path / "summary.json").read_text())
    print("\nGPU smoke test summary:", json.dumps(summary, indent=2))
    assert summary["status"] == "success", summary["error"]
    assert summary["steps_completed"] == SMOKE_UNDECIDED["max_steps"]
    assert summary["training_gpu_hours"] > 0
    assert summary["peak_vram_gib"] is not None
