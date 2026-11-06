"""Config loading: composition, overrides, and the rule that a typo must raise."""

from __future__ import annotations

import dataclasses

import pytest
import yaml
from agent_evals.config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    MeasurementConfig,
    PoolConfig,
    RewardConfig,
    RunConfig,
    TopologyConfig,
    TrainingConfig,
    apply_overrides,
    load_config,
    to_dict,
)


def _write(tmp_path, name, data):
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


class TestShippedBaseConfig:
    def test_base_yaml_needs_an_owner_and_says_so(self, base_config_path):
        with pytest.raises(ConfigError, match="run.owner"):
            load_config(base_config_path)

    def test_base_yaml_loads_once_an_owner_is_supplied(self, base_config_path):
        cfg = load_config(base_config_path, ["run.owner=sameer"])
        assert cfg.run.owner == "sameer"
        assert cfg.run.seed == 0
        assert cfg.run.output_root == "results"
        assert cfg.logging.console_level == "INFO"
        assert cfg.logging.file_level == "DEBUG"
        assert cfg.logging.flush_metrics_every == 1
        assert cfg.seeding.deterministic is False


class TestUnknownKeys:
    def test_unknown_top_level_key_raises(self, tmp_path):
        path = _write(tmp_path, "bad", {"run": {"owner": "sameer"}, "tempreature": 0.7})
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(path)

    def test_unknown_nested_key_raises_with_its_path(self, tmp_path):
        path = _write(tmp_path, "bad", {"run": {"owner": "sameer", "seeed": 3}})
        with pytest.raises(ConfigError, match="run.seeed"):
            load_config(path)

    def test_error_lists_the_valid_keys(self, tmp_path):
        path = _write(tmp_path, "bad", {"run": {"owner": "sameer", "seeed": 3}})
        with pytest.raises(ConfigError, match="seed"):
            load_config(path)


class TestComposition:
    def test_defaults_compose_and_child_overlays_parent(self, tmp_path):
        _write(tmp_path, "base", {"name": "base", "run": {"owner": "sameer", "seed": 0}})
        child = _write(
            tmp_path, "child", {"defaults": ["base"], "name": "child", "run": {"seed": 7}}
        )
        cfg = load_config(child)
        assert cfg.name == "child"
        assert cfg.run.seed == 7
        assert cfg.run.owner == "sameer"

    def test_circular_defaults_raise(self, tmp_path):
        _write(tmp_path, "a", {"defaults": ["b"], "run": {"owner": "sameer"}})
        b = _write(tmp_path, "b", {"defaults": ["a"], "run": {"owner": "sameer"}})
        with pytest.raises(ConfigError, match="circular"):
            load_config(b)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")


class TestOverrides:
    def test_override_coerces_string_to_int(self, base_config_path):
        cfg = load_config(base_config_path, ["run.owner=sameer", "run.seed=2"])
        assert cfg.run.seed == 2 and isinstance(cfg.run.seed, int)

    def test_override_coerces_string_to_bool(self, base_config_path):
        cfg = load_config(base_config_path, ["run.owner=sameer", "seeding.deterministic=true"])
        assert cfg.seeding.deterministic is True

    def test_override_without_equals_raises(self):
        with pytest.raises(ConfigError, match="key.path=value"):
            apply_overrides({}, ["run.seed"])

    def test_uncoercible_value_raises(self, base_config_path):
        with pytest.raises(ConfigError, match="cannot read"):
            load_config(base_config_path, ["run.owner=sameer", "run.seed=abc"])


class TestValidation:
    def test_negative_seed_rejected(self, base_config_path):
        with pytest.raises(ConfigError, match="non-negative"):
            load_config(base_config_path, ["run.owner=sameer", "run.seed=-1"])

    def test_unknown_owner_rejected(self, base_config_path):
        with pytest.raises(ConfigError, match="not a known owner"):
            load_config(base_config_path, ["run.owner=sameerr"])

    def test_bad_log_level_rejected(self):
        with pytest.raises(ConfigError, match="console_level"):
            LoggingConfig(console_level="CHATTY")

    def test_zero_flush_interval_rejected(self):
        with pytest.raises(ConfigError, match="flush_metrics_every"):
            LoggingConfig(flush_metrics_every=0)


class TestImmutabilityAndRoundTrip:
    def test_config_is_frozen(self, cfg):
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.run.seed = 99

    def test_to_dict_round_trips_through_yaml(self, cfg, tmp_path):
        path = tmp_path / "resolved.yaml"
        path.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        reloaded = load_config(path)
        assert reloaded == cfg

    def test_resolved_config_alone_is_enough_to_rerun(self, base_config_path, tmp_path):
        original = load_config(base_config_path, ["run.owner=kartik", "run.seed=2"])
        path = tmp_path / "resolved.yaml"
        path.write_text(yaml.safe_dump(to_dict(original), sort_keys=False))
        assert load_config(path) == original


class TestDefaultsAreTheApprovedValues:
    """The infrastructure parameter table approved 2026-09-10."""

    def test_approved_defaults(self):
        run = RunConfig(owner="sameer")
        logging_cfg = LoggingConfig()
        cfg = ExperimentConfig(run=run)
        assert run.seed == 0
        assert run.output_root == "results"
        assert run.known_owners == ("sameer", "kartik", "siddhanth")
        assert logging_cfg.console_level == "INFO"
        assert logging_cfg.file_level == "DEBUG"
        assert logging_cfg.flush_metrics_every == 1
        assert cfg.seeding.deterministic is False


class TestTopologyConfig:
    def test_max_rounds_defaults_to_none(self):
        assert TopologyConfig(name="t0_single_agent").max_rounds is None

    @pytest.mark.parametrize("value", [0, -1, True, "x"])
    def test_invalid_max_rounds_raise(self, value):
        with pytest.raises(ConfigError, match="max_rounds"):
            TopologyConfig(name="t1_self_refine", max_rounds=value)

    def test_base_yaml_has_no_topology_group(self, base_config_path):
        assert load_config(base_config_path, ["run.owner=sameer"]).topology is None

    def test_extra_file_merges_after_the_main_file_and_before_overrides(self, tmp_path):
        main = _write(
            tmp_path,
            "main",
            {"run": {"owner": "sameer", "seed": 1}, "topology": {"name": "t", "max_rounds": 3}},
        )
        extra = _write(tmp_path, "extra", {"run": {"seed": 2}, "topology": {"max_rounds": 4}})
        merged = load_config(main, extra_paths=[extra])
        assert merged.run.seed == 2 and merged.topology == TopologyConfig("t", 4)
        overridden = load_config(main, ["topology.max_rounds=2"], extra_paths=[extra])
        assert overridden.topology.max_rounds == 2

    def test_shipped_t1_file_composes_with_the_xsum_config(self, base_config_path):
        xsum = base_config_path.parent / "data" / "tier1_xsum.yaml"
        t1 = base_config_path.parent / "topology" / "t1_self_refine.yaml"
        cfg = load_config(xsum, ["run.owner=sameer"], extra_paths=[t1])
        assert cfg.topology == TopologyConfig(name="t1_self_refine", max_rounds=4)
        assert cfg.data.name == "tier1_xsum"
        overridden = load_config(
            xsum, ["run.owner=sameer", "topology.max_rounds=2"], extra_paths=[t1]
        )
        assert overridden.topology.max_rounds == 2

    def test_unknown_key_under_topology_raises(self, tmp_path):
        path = _write(
            tmp_path,
            "c",
            {"run": {"owner": "sameer"}, "topology": {"name": "t", "max_round": 4}},
        )
        with pytest.raises(ConfigError, match="topology.max_round"):
            load_config(path)

    def test_topology_group_round_trips_through_the_resolved_config(self, tmp_path):
        cfg = ExperimentConfig(
            run=RunConfig(owner="sameer"), topology=TopologyConfig("t1_self_refine", 4)
        )
        resolved = tmp_path / "config.resolved.yaml"
        resolved.write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
        assert load_config(resolved) == cfg


class TestDataConfig:
    def test_infrastructure_config_does_not_select_a_dataset(self, cfg):
        assert cfg.data is None

    def test_xsum_config_composes_and_preserves_owner_requirement(self, base_config_path):
        path = base_config_path.parent / "data" / "tier1_xsum.yaml"
        with pytest.raises(ConfigError, match="run.owner"):
            load_config(path)
        cfg = load_config(path, ["run.owner=sameer"])
        assert isinstance(cfg.data, DataConfig)
        assert cfg.data.name == "tier1_xsum"
        assert cfg.data.n_samples == 500
        assert cfg.data.n_strata == 3
        assert cfg.data.min_source_words == 20
        assert cfg.data.max_source_words is None
        assert cfg.data.sample_seed == 123
        assert cfg.data.hf_config is None
        assert cfg.data.dedupe_sources is False
        assert cfg.run.seed == 0

    def test_cnndm_config_composes_the_approved_values(self, base_config_path):
        path = base_config_path.parent / "data" / "tier1_cnndm.yaml"
        cfg = load_config(path, ["run.owner=sameer"])
        assert cfg.data.name == "tier1_cnndm"
        assert cfg.data.hf_path == "abisee/cnn_dailymail"
        assert cfg.data.hf_revision == "96df5e686bee6baa90b8bee7c28b81fa3fa6223d"
        assert cfg.data.hf_config == "3.0.0"
        assert cfg.data.split == "test"
        assert cfg.data.dedupe_sources is True
        assert (cfg.data.n_samples, cfg.data.n_strata) == (500, 3)
        assert cfg.data.sample_seed == 123
        assert cfg.data.min_source_words == 20
        assert cfg.data.max_source_words is None

    def test_samsum_config_composes_the_approved_values(self, base_config_path):
        path = base_config_path.parent / "data" / "tier1_samsum.yaml"
        cfg = load_config(path, ["run.owner=sameer"])
        assert cfg.data.name == "tier1_samsum"
        assert cfg.data.hf_path == "knkarthick/samsum"
        assert cfg.data.hf_revision == "6b929ff10edec703164e3ddb2e94aae058c9ab5f"
        assert cfg.data.hf_config is None
        assert cfg.data.split == "test"
        assert cfg.data.dedupe_sources is True
        assert (cfg.data.n_samples, cfg.data.n_strata) == (500, 3)
        assert cfg.data.sample_seed == 123
        assert cfg.data.min_source_words == 20
        assert cfg.data.max_source_words is None
        assert cfg.data.sample_ids is not None and len(cfg.data.sample_ids) == 500

    def test_optional_data_group_loads_and_round_trips(self, tmp_path):
        path = _write(
            tmp_path,
            "data",
            {
                "run": {"owner": "sameer"},
                "data": {
                    "n_samples": 3,
                    "sample_ids": ["FAKE_EXAMPLE_a", "FAKE_EXAMPLE_b", "FAKE_EXAMPLE_c"],
                },
            },
        )
        cfg = load_config(path)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.data.sample_ids, tuple)
        path = _write(tmp_path, "resolved", to_dict(cfg))
        assert load_config(path) == cfg
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.data.sample_seed = 0

    @pytest.mark.parametrize("value", [None, "null", "none"])
    def test_optional_data_group_can_be_null(self, tmp_path, value):
        path = _write(tmp_path, "data", {"run": {"owner": "sameer"}, "data": value})
        assert load_config(path).data is None

    def test_data_overrides_and_explicit_null(self, tmp_path):
        path = _write(tmp_path, "data", {"run": {"owner": "sameer"}, "data": {}})
        cfg = load_config(path, ["data.max_source_words=100", "data.sample_ids=null"])
        assert cfg.data.max_source_words == 100
        assert cfg.data.sample_ids is None
        cfg = load_config(path, ["data.n_samples=3", "data.sample_ids=a,b,c"])
        assert cfg.data.sample_ids == ("a", "b", "c")

    @pytest.mark.parametrize(
        "values",
        [
            {"sample_sead": 123},
            {"tier": 1},
            {"task": "summarization"},
        ],
    )
    def test_unknown_data_keys_raise(self, tmp_path, values):
        path = _write(tmp_path, "data", {"run": {"owner": "sameer"}, "data": values})
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(path)

    @pytest.mark.parametrize(
        "values",
        [
            {"n_samples": 0},
            {"n_strata": 0},
            {"n_samples": 2},
            {"sample_seed": -1},
            {"sample_seed": 2**32},
            {"min_source_words": -1},
            {"max_source_words": 19},
            {"hf_revision": None},
            {"hf_revision": "main"},
            {"hf_path": ""},
            {"split": ""},
            {"stratify_by": "tokens"},
            {"n_samples": True},
            {"max_source_words": True},
            {"hf_config": ""},
            {"hf_config": 3},
            {"dedupe_sources": 1},
            {"dedupe_sources": None},
            {"hf_filename": "data.tar"},
            {"hf_file_sha256": "a" * 64},
            {"hf_filename": "data.tar", "hf_file_sha256": "A" * 64},
            {"hf_filename": "data.tar", "hf_file_sha256": "a" * 63},
            {"hf_filename": "", "hf_file_sha256": "a" * 64},
            {"archive_member": "x.json"},
            {"n_passages": 0},
            {"n_passages": True},
            {"strip_citations": "yes"},
            {"stratify_by": "sub_questions"},
            {"stratify_by": "sub_questions", "strata": ((2,), (3, 4))},
            {"stratify_by": "sub_questions", "strata": ((2,), (2, 3), (4,))},
            {"stratify_by": "sub_questions", "strata": ((2,), (), (4,))},
            {"stratify_by": "sub_questions", "strata": ((0,), (3,), (4,))},
            {"strata": ((2,), (3,), (4,))},
        ],
    )
    def test_invalid_parameters_raise(self, values):
        with pytest.raises(ConfigError):
            DataConfig(**values)

    @pytest.mark.parametrize(
        "ids",
        [
            ["a", "b", "c"],
            ("a", "a", "c"),
            ("a", "b"),
            ("a", "", "c"),
            ("a", 2, "c"),
        ],
    )
    def test_invalid_frozen_id_contract_raises(self, ids):
        with pytest.raises(ConfigError, match="sample_ids"):
            DataConfig(n_samples=3, sample_ids=ids)

    @pytest.mark.parametrize(
        "values",
        [
            {"n_samples": 3.5},
            {"sample_seed": True},
            {"sample_ids": 123},
            {"sample_ids": [1, 2, 3]},
        ],
    )
    def test_parser_rejects_invalid_types_before_coercion(self, tmp_path, values):
        path = _write(tmp_path, "data", {"run": {"owner": "sameer"}, "data": values})
        with pytest.raises(ConfigError):
            load_config(path)

    @pytest.mark.parametrize("value", [[], "xsum", 1])
    def test_data_group_must_be_a_mapping(self, tmp_path, value):
        path = _write(tmp_path, "data", {"run": {"owner": "sameer"}, "data": value})
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)


class TestRewardConfig:
    def test_runs_that_do_not_train_have_no_reward(self, cfg):
        assert cfg.reward is None

    def test_approved_defaults(self):
        reward = RewardConfig()
        assert reward.name == "alignscore"  # D-048
        assert reward.blank_reward == 0.0  # D-050

    def test_reward_group_loads_and_round_trips(self, tmp_path):
        path = _write(tmp_path, "reward", {"run": {"owner": "sameer"}, "reward": {}})
        cfg = load_config(path)
        assert cfg.reward == RewardConfig()
        path = _write(tmp_path, "resolved", to_dict(cfg))
        assert load_config(path) == cfg
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.reward.blank_reward = 0.5

    def test_override_coerces_blank_reward_to_float(self, tmp_path):
        path = _write(tmp_path, "reward", {"run": {"owner": "sameer"}, "reward": {}})
        cfg = load_config(path, ["reward.blank_reward=0"])
        assert cfg.reward.blank_reward == 0.0
        assert isinstance(cfg.reward.blank_reward, float)

    def test_unknown_reward_key_raises(self, tmp_path):
        path = _write(tmp_path, "reward", {"run": {"owner": "sameer"}, "reward": {"weight": 1}})
        with pytest.raises(ConfigError, match="reward.weight"):
            load_config(path)

    @pytest.mark.parametrize(
        "values",
        [{"name": ""}, {"blank_reward": float("nan")}, {"blank_reward": True}, {"name": 3}],
    )
    def test_invalid_values_raise(self, values):
        with pytest.raises(ConfigError):
            RewardConfig(**values)


#: A complete training group. Test values only: they check validation, they are not proposals.
_TRAINING = {
    "beta": 0.1,
    "loss_type": "grpo",
    "scale_rewards": "group",
    "epsilon": 0.2,
    "num_iterations": 1,
    "num_generations": 2,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 1,
    "max_steps": 2,
    "save_steps": 2,
    "learning_rate": 0.001,
    "lr_scheduler_type": "constant",
    "warmup_steps": 0,
    "optim": "adamw_torch",
    "gradient_checkpointing": False,
    "mask_truncated_completions": False,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "lora_rank": 2,
    "lora_alpha": 4,
    "lora_dropout": 0.0,
    "compute_dtype": "float32",
}


class TestTrainingConfig:
    def test_runs_that_do_not_train_have_no_training_group(self, cfg):
        assert cfg.training is None

    @pytest.mark.parametrize(
        "name",
        [
            "num_generations",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "save_steps",
        ],
    )
    def test_undecided_values_have_no_default(self, tmp_path, name):
        values = {key: value for key, value in _TRAINING.items() if key != name}
        path = _write(tmp_path, "train", {"run": {"owner": "sameer"}, "training": values})
        with pytest.raises(ConfigError, match=f"missing required config key 'training.{name}'"):
            load_config(path)

    def test_approved_defaults(self):
        undecided = (
            "num_generations",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "save_steps",
        )
        training = TrainingConfig(**{key: _TRAINING[key] for key in undecided})
        assert training.beta == 0.04  # D-054, Shao et al. 2024 §4.2
        assert (training.loss_type, training.scale_rewards) == ("dr_grpo", "none")  # D-054
        assert training.epsilon == 0.2 and training.num_iterations == 1  # D-054
        assert training.learning_rate == 1e-5  # D-054
        assert (training.lr_scheduler_type, training.warmup_steps) == ("constant", 0.0)  # D-054
        assert training.optim == "paged_adamw_32bit" and training.max_grad_norm == 0.3  # D-054
        assert training.gradient_checkpointing is True  # D-054
        assert training.mask_truncated_completions is False  # D-054
        sampling = (training.temperature, training.top_p, training.top_k)
        assert sampling == (1.0, 1.0, 0) and training.repetition_penalty == 1.0  # D-054
        lora = (training.lora_rank, training.lora_alpha, training.lora_dropout)
        assert lora == (16, 32, 0.0)  # D-054
        assert training.compute_dtype == "bfloat16"  # D-054
        assert training.lora_target_modules == "all-linear"  # AGENTS.md §12.1
        assert training.quant_type == "nf4" and training.double_quant is True  # D-008, D-041
        assert training.adapter_root == "~/.cache/agent_evals/adapters"  # D-053

    def test_training_group_loads_and_round_trips(self, tmp_path):
        path = _write(tmp_path, "train", {"run": {"owner": "sameer"}, "training": _TRAINING})
        cfg = load_config(path, ["training.learning_rate=0.0005"])
        assert cfg.training.learning_rate == 0.0005
        assert isinstance(cfg.training.warmup_steps, float)
        path = _write(tmp_path, "resolved", to_dict(cfg))
        assert load_config(path) == cfg

    def test_unknown_training_key_raises(self, tmp_path):
        data = {"run": {"owner": "sameer"}, "training": {**_TRAINING, "kl_coef": 0.1}}
        with pytest.raises(ConfigError, match="training.kl_coef"):
            load_config(_write(tmp_path, "train", data))

    @pytest.mark.parametrize(
        "changes",
        [
            {"beta": 0.0},
            {"num_generations": 1},
            {"learning_rate": -1.0},
            {"top_p": 1.5},
            {"top_k": -1},
            {"lora_dropout": 1.0},
            {"warmup_steps": -1},
            {"max_steps": 0},
            {"compute_dtype": "int8"},
            {"gradient_checkpointing": "yes"},
            {"loss_type": ""},
            {"temperature": float("nan")},
            {"max_grad_norm": 0.0},
        ],
    )
    def test_invalid_values_raise(self, changes):
        with pytest.raises(ConfigError):
            TrainingConfig(**{**_TRAINING, **changes})


class TestMeasurementConfig:
    _GRID = {"backbones": ("a", "b"), "prompt_caps": (1024, 2048), "group_sizes": (2, 4, 8)}

    def test_runs_outside_the_measurement_have_no_grid(self, cfg):
        assert cfg.measurement is None

    def test_grid_loads_from_lists(self, tmp_path):
        grid = {key: list(values) for key, values in self._GRID.items()}
        data = {"run": {"owner": "sameer"}, "measurement": grid}
        cfg = load_config(_write(tmp_path, "grid", data))
        assert cfg.measurement == MeasurementConfig(**self._GRID)

    @pytest.mark.parametrize(
        "changes",
        [
            {"backbones": ()},
            {"prompt_caps": (0,)},
            {"group_sizes": (1, 2)},
            {"group_sizes": (4, 2)},
            {"prompt_caps": (1024, 1024)},
            {"typical_within_cap": 1536},
            {"typical_group_size": 8},
            {"typical_within_cap": 0, "typical_group_size": 8},
            {"typical_within_cap": 1536, "typical_group_size": 1},
            {"forced_cap": 1024},
            {"typical_within_cap": 1536, "typical_group_size": 8, "forced_cap": 1536},
        ],
    )
    def test_invalid_grid_raises(self, changes):
        with pytest.raises(ConfigError):
            MeasurementConfig(**{**self._GRID, **changes})


class TestPoolConfig:
    def test_valid_filter(self):
        pool = PoolConfig(prompt_cap=1536, backbones=("a", "b"))
        assert pool.prompt_cap == 1536

    @pytest.mark.parametrize(
        "values",
        [
            {"prompt_cap": 0, "backbones": ("a",)},
            {"prompt_cap": 1536, "backbones": ()},
            {"prompt_cap": 1536, "backbones": ("a", "a")},
        ],
    )
    def test_invalid_filter_raises(self, values):
        with pytest.raises(ConfigError):
            PoolConfig(**values)
