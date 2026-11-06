"""T5 training on ASQA (D-132 to D-136): the training split, the three reward terms, GDPO wiring,
worked-example turns in training prompts, and the memory-test configs.

CPU only, no network. Source files are written into a temporary cache with their real checksums,
so the adapter reads them exactly as it reads the downloaded originals; any download attempt fails.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from agent_evals.agents.asqa_examples import ALCE_ASQA_EXAMPLES
from agent_evals.agents.generator import (
    ASQA_ALCE_VANILLA,
    ASQA_ALCE_ZERO_SHOT,
    TEMPLATES,
    TEMPLATES_BY_NAME,
)
from agent_evals.config import (
    CitationConfig,
    ConfigError,
    DataConfig,
    MetricsConfig,
    RewardConfig,
    SourceFile,
    TrainingConfig,
    load_config,
)
from agent_evals.config.metrics import AlignScoreConfig
from agent_evals.datasets import ASQATrainAdapter, DatasetError
from agent_evals.datasets.asqa import format_passages
from agent_evals.metrics import Metric
from agent_evals.metrics import artifacts as artifacts_module
from agent_evals.metrics import citation as citation_module
from agent_evals.metrics.alignscore import AlignScore
from agent_evals.metrics.str_em import StrEM
from agent_evals.registry import DATASETS, REWARDS
from agent_evals.rl import (
    CitationPrecisionReward,
    CitationRecallReward,
    ExactMatchRecallReward,
    RewardError,
    SharedMetric,
    training_template,
)
from agent_evals.rl.memory import PromptLengths
from agent_evals.rl.trainer import (
    STEP_COLUMNS,
    grpo_args,
    question_seed,
    step_columns,
    training_dataset,
)
from agent_evals.seeding import MAX_SEED
from test_trainer import (
    BACKBONE,
    CPU_ONLY,
    SMOKE_UNDECIDED,
    OfflineRun,
    _config,
    cpu_only,  # noqa: F401 - pytest fixture
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
ANNOTATIONS = "ASQA.json"
PASSAGE_FILES = ("passages_a.json", "passages_b.json")


def _official_item(index, n_sub=2):
    return {
        "ambiguous_question": f"Who won fixture contest {index}?",
        "qa_pairs": [
            {"question": f"Sub {index}.{k}?", "short_answers": [f"Winner{index}x{k}"]}
            for k in range(n_sub)
        ],
        "annotations": [{"long_answer": f"FAKE_EXAMPLE human answer {index}.", "knowledge": []}],
        "wikipages": [],
    }


def _official(n_train=6):
    train = {f"FAKE_EXAMPLE_t{i}": _official_item(i) for i in range(n_train)}
    dev = {"FAKE_EXAMPLE_d0": _official_item(100)}
    return {"train": train, "dev": dev}


def _passage_items(official, n_docs=6):
    return [
        {
            "question": item["ambiguous_question"],
            "qa_pairs": item["qa_pairs"],
            "docs": [
                {
                    "id": str(d),
                    "title": f"Title {sid} {d}",
                    "text": f"Passage {sid} {d}.",
                    "score": 1,
                }
                for d in range(n_docs)
            ],
            "output": "ignored",
        }
        for sid, item in official["train"].items()
    ]


def _write_sources(cache: Path, official, items) -> tuple[SourceFile, ...]:
    """Write the files into the cache and return their pinned entries."""
    cache.mkdir(parents=True, exist_ok=True)
    half = len(items) // 2
    contents = {
        ANNOTATIONS: official,
        PASSAGE_FILES[0]: items[:half],
        PASSAGE_FILES[1]: items[half:],
    }
    entries = []
    for name, value in contents.items():
        data = json.dumps(value).encode("utf-8")
        (cache / name).write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        entries.append(SourceFile(name=name, url=f"https://example.invalid/{name}", sha256=digest))
    return tuple(entries)


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    """Any download attempt fails, as with no network."""

    def refuse(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(artifacts_module.urllib.request, "urlopen", refuse)


def _data(tmp_path, official=None, items=None, **changes) -> DataConfig:
    official = official or _official()
    items = _passage_items(official) if items is None else items
    cache = tmp_path / "data_cache"
    values = {
        "name": "tier2_asqa_train",
        "hf_path": None,
        "hf_revision": None,
        "source_files": _write_sources(cache, official, items),
        "cache_dir": str(cache),
        "split": "train",
        "n_passages": 5,
        "strip_citations": True,
        "n_samples": 4,
        "n_strata": 1,
        "min_source_words": 0,
        "exclude_ids": ("FAKE_EXAMPLE_t5",),
    }
    values.update(changes)
    return DataConfig(**values)


class TestTrainingSplit:
    def test_joins_official_annotations_with_their_passages(self, tmp_path):
        official = _official()
        pool = ASQATrainAdapter(_data(tmp_path, official)).eligible_pool()
        assert sorted(pool["id"]) == [f"FAKE_EXAMPLE_t{i}" for i in range(5)]
        row = pool[0]
        item = official["train"][row["id"]]
        assert row["question"] == item["ambiguous_question"]
        docs = _passage_items({"train": {row["id"]: item}})[0]["docs"][:5]
        assert row["source"] == row["evidence"] == format_passages(docs)
        assert row["reference"] == item["annotations"][-1]["long_answer"]
        assert json.loads(row["gold"]) == {
            "short_answers": [p["short_answers"] for p in item["qa_pairs"]],
            "long_answers": [item["annotations"][0]["long_answer"]],
        }
        assert (row["tier"], row["task"]) == (2, "grounded_qa")

    def test_excluded_ids_are_dropped_and_must_exist(self, tmp_path):
        pool = ASQATrainAdapter(_data(tmp_path)).eligible_pool()
        assert "FAKE_EXAMPLE_t5" not in pool["id"]
        adapter = ASQATrainAdapter(_data(tmp_path, exclude_ids=("FAKE_EXAMPLE_nope",)))
        with pytest.raises(DatasetError, match="exclude_ids not in"):
            adapter.eligible_pool()

    def test_train_is_trainable_and_dev_is_not_its_split(self, tmp_path):
        ASQATrainAdapter(_data(tmp_path)).assert_trainable("train")
        with pytest.raises(DatasetError, match="holds split 'train' only"):
            ASQATrainAdapter(_data(tmp_path, split="dev"))

    def test_a_training_question_in_the_dev_split_is_refused(self, tmp_path):
        official = _official()
        official["dev"]["FAKE_EXAMPLE_d1"] = official["train"]["FAKE_EXAMPLE_t0"]
        with pytest.raises(DatasetError, match="also in ASQA's dev split"):
            ASQATrainAdapter(_data(tmp_path, official)).eligible_pool()

    def test_a_question_without_passages_is_refused(self, tmp_path):
        official = _official()
        items = _passage_items(official)[1:]
        with pytest.raises(DatasetError, match="in only one of"):
            ASQATrainAdapter(_data(tmp_path, official, items)).eligible_pool()

    def test_different_sub_questions_are_refused(self, tmp_path):
        official = _official()
        items = _passage_items(official)
        items[0]["qa_pairs"] = items[0]["qa_pairs"][:1]
        with pytest.raises(DatasetError, match="sub-questions differ"):
            ASQATrainAdapter(_data(tmp_path, official, items)).eligible_pool()

    def test_too_few_passages_are_refused(self, tmp_path):
        official = _official()
        items = _passage_items(official, n_docs=4)
        with pytest.raises(DatasetError, match="fewer than data.n_passages"):
            ASQATrainAdapter(_data(tmp_path, official, items)).eligible_pool()

    def test_a_changed_file_is_refused(self, tmp_path):
        cfg = _data(tmp_path)
        (Path(cfg.cache_dir) / ANNOTATIONS).write_text("{}", encoding="utf-8")
        with pytest.raises(DatasetError, match="could not read 'ASQA.json'"):
            ASQATrainAdapter(cfg).eligible_pool()

    def test_shipped_config_pins_the_sources_and_excludes_alces_examples(self):
        cfg = load_config(CONFIGS / "data" / "tier2_asqa_train.yaml", ["run.owner=sameer"])
        data = cfg.data
        assert DATASETS.get(data.name) is ASQATrainAdapter
        assert (data.hf_path, data.hf_revision, data.split, data.n_passages) == (
            None,
            None,
            "train",
            5,
        )
        assert {f.name: f.sha256 for f in data.source_files} == {
            "ASQA.json": "8de0c824824372243b73b10314727dac1a02d9ecc729c34c752c30f8a3c1e58a",
            "huang2024_asqa_train.json": (
                "62580fd30c64c907fe927c49f808b25c727a670cf64d674342bcbc0bb050e056"
            ),
            "huang2024_asqa_dev.json": (
                "4ea8ae12e06a765b43bf4f9733bb63f884a22eb2c88145712b138066361c28bd"
            ),
        }
        assert set(data.exclude_ids) == {e.source_id for e in ALCE_ASQA_EXAMPLES}
        assert data.strip_citations is True
        assert cfg.sampling.max_tokens == 300


class TestDataConfig:
    def test_source_files_replace_the_hub_source(self, tmp_path):
        entry = SourceFile(name="a.json", url="https://example.invalid/a", sha256="a" * 64)
        with pytest.raises(ConfigError, match="hf_path must be None"):
            DataConfig(source_files=(entry,), hf_revision=None)
        with pytest.raises(ConfigError, match="names must be unique"):
            DataConfig(source_files=(entry, entry), hf_path=None, hf_revision=None)
        with pytest.raises(ConfigError, match="hf_path must be a non-empty string"):
            DataConfig(hf_path=None)

    @pytest.mark.parametrize(
        "changes",
        [{"sha256": "A" * 64}, {"sha256": "a" * 63}, {"name": "dir/a.json"}, {"url": " "}],
    )
    def test_bad_source_file_entries_raise(self, changes):
        values = {"name": "a.json", "url": "https://example.invalid/a", "sha256": "a" * 64}
        with pytest.raises(ConfigError):
            SourceFile(**{**values, **changes})

    def test_excluded_ids_are_unique_and_never_sampled(self):
        with pytest.raises(ConfigError, match="exclude_ids"):
            DataConfig(exclude_ids=("x", "x"))
        with pytest.raises(ConfigError, match="contains excluded IDs"):
            DataConfig(exclude_ids=("x",), n_samples=1, n_strata=1, sample_ids=("x",))


class TestRewardAndTrainingConfig:
    def test_exactly_one_of_name_and_terms(self):
        with pytest.raises(ConfigError, match="exactly one"):
            RewardConfig(terms=("str_em", "citation_recall"), weights=(1.0, 1.0))
        with pytest.raises(ConfigError, match="exactly one"):
            RewardConfig(name=None)

    @pytest.mark.parametrize(
        "terms, weights",
        [(("str_em",), (1.0,)), (("str_em", "str_em"), (1.0, 1.0)), (("a", "b"), (1.0,))],
    )
    def test_terms_need_two_unique_names_and_a_weight_each(self, terms, weights):
        with pytest.raises(ConfigError):
            RewardConfig(name=None, terms=terms, weights=weights)

    @pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), True])
    def test_weights_are_positive_numbers(self, weight):
        with pytest.raises(ConfigError, match="reward.weights"):
            RewardConfig(name=None, terms=("a", "b"), weights=(1.0, weight))

    def test_each_term_is_built_as_a_single_reward(self):
        cfg = RewardConfig(name=None, terms=("a", "b"), weights=(1.0, 2.0), blank_reward=0.0)
        assert cfg.names == ("a", "b")
        assert cfg.single("b") == RewardConfig(name="b", blank_reward=0.0)
        assert RewardConfig().names == ("alignscore",)

    def test_gdpo_requires_group_scaling_stated(self):
        base = {**SMOKE_UNDECIDED, **CPU_ONLY}
        with pytest.raises(ConfigError, match="scale_rewards to group"):
            TrainingConfig(**base, multi_objective_aggregation="normalize_then_sum")
        TrainingConfig(
            **base, multi_objective_aggregation="normalize_then_sum", scale_rewards="group"
        )
        with pytest.raises(ConfigError, match="multi_objective_aggregation"):
            TrainingConfig(**base, multi_objective_aggregation="sum")

    def test_worked_examples_seed_and_prompt_template(self):
        base = {**SMOKE_UNDECIDED, **CPU_ONLY}
        with pytest.raises(ConfigError, match="worked_examples_seed"):
            TrainingConfig(**base, worked_examples_seed=-1)
        with pytest.raises(ConfigError, match="prompt_template"):
            TrainingConfig(**base, prompt_template=" ")

    def test_citation_judge_and_alignscore_threshold(self):
        assert CitationConfig().judge == "minicheck"
        with pytest.raises(ConfigError, match="judge"):
            CitationConfig(judge="true")
        assert AlignScoreConfig().threshold == 0.5
        for value in (0.0, 1.0):
            with pytest.raises(ConfigError, match="threshold"):
                AlignScoreConfig(threshold=value)


class StubCitation(Metric):
    """Both citation keys from the text, without a judge model. Counts calls."""

    KEYS = ("citation_recall", "citation_precision")
    READS_CITATIONS = True

    def __init__(self):
        super().__init__(MetricsConfig(device="cpu"))
        self.calls = 0

    def _score_examples(self, predictions, references, evidence, gold):
        self.calls += 1
        return [
            {"citation_recall": float("[1]" in p), "citation_precision": float("[2]" in p)}
            for p in predictions
        ]


def _gold(*answers):
    return json.dumps({"short_answers": [[a] for a in answers], "long_answers": ["x"]})


def _columns(n, gold=None):
    return {
        "evidence": ["Document [1](Title: t): FAKE_EXAMPLE passage."] * n,
        "reference": ["FAKE_EXAMPLE reference"] * n,
        "gold": [gold or _gold("Paris", "Rome")] * n,
    }


class TestASQARewards:
    def test_registered_with_their_metric_keys(self):
        for name, cls, metric, key in (
            ("str_em", ExactMatchRecallReward, "str_em", "str_em"),
            ("citation_recall", CitationRecallReward, "citation", "citation_recall"),
            ("citation_precision", CitationPrecisionReward, "citation", "citation_precision"),
        ):
            assert REWARDS.get(name) is cls
            assert metric == cls.METRIC and key == cls.KEY
            assert cls.OUTPUT_RANGE == (0.0, 1.0)

    def test_correctness_is_exact_match_recall_read_without_markers(self):
        reward = ExactMatchRecallReward(
            RewardConfig(name="str_em"), StrEM(MetricsConfig(device="cpu")), strip_citations=True
        )
        completions = ["Paris [1] and Rome [2].", "Only Paris [1].", "[1][2]", "Rome[3]."]
        assert reward.score(completions, **_columns(4)) == [1.0, 0.5, 0.0, 0.5]

    def test_correctness_needs_the_gold_column(self):
        reward = ExactMatchRecallReward(
            RewardConfig(name="str_em"), StrEM(MetricsConfig(device="cpu"))
        )
        columns = _columns(1)
        del columns["gold"]
        with pytest.raises(RewardError, match="gold"):
            reward.score(["Paris"], **columns)

    def test_citation_terms_read_the_markers_and_share_one_scoring(self):
        metric = StubCitation()
        shared = SharedMetric(metric)
        recall = CitationRecallReward(
            RewardConfig(name="citation_recall"), shared, strip_citations=True
        )
        precision = CitationPrecisionReward(
            RewardConfig(name="citation_precision"), shared, strip_citations=True
        )
        completions = ["A [1].", "B [1][2].", "C [2]."]
        columns = _columns(3)
        assert recall.score(completions, **columns) == [1.0, 1.0, 0.0]
        assert precision.score(completions, **columns) == [0.0, 1.0, 1.0]
        assert metric.calls == 1
        recall.score(["D [1]."], **_columns(1))
        assert metric.calls == 2

    def test_blank_answers_get_the_blank_reward_on_every_term(self):
        metric = SharedMetric(StubCitation())
        recall = CitationRecallReward(RewardConfig(name="citation_recall"), metric)
        assert recall.score(["  ", "A [1]."], **_columns(2)) == [0.0, 1.0]

    def test_a_several_term_config_cannot_build_a_reward_directly(self):
        cfg = RewardConfig(name=None, terms=("str_em", "citation_recall"), weights=(1.0, 1.0))
        with pytest.raises(RewardError, match="cfg.single"):
            ExactMatchRecallReward(cfg, StrEM(MetricsConfig(device="cpu")))


class FakeJudge:
    """Records the threshold path: support is 1 for claims containing 'yes'."""

    def __init__(self, cfg):
        self.calls = []

    def claim_support(self, document, claim):
        self.calls.append((document, claim))
        return 1.0 if "yes" in claim else 0.0


class TestCitationJudge:
    def test_alignscore_judges_when_configured_minicheck_otherwise(self, monkeypatch):
        monkeypatch.setattr(citation_module, "AlignScore", FakeJudge)
        monkeypatch.setattr(citation_module, "MiniCheck", lambda cfg: pytest.fail("MiniCheck"))
        cfg = MetricsConfig(device="cpu", citation=CitationConfig(judge="alignscore"))
        metric = citation_module.CitationQuality(cfg)
        assert isinstance(metric._judge, FakeJudge)
        assert metric._threshold == cfg.alignscore.threshold
        evidence = format_passages([{"title": "T", "text": "Some passage."}])
        rows = metric.score(["It is yes [1]. It is no [1]."], ["r"], [evidence])
        assert rows == [{"citation_recall": 0.5, "citation_precision": 0.5}]

    def test_default_judge_is_minicheck(self, monkeypatch):
        monkeypatch.setattr(citation_module, "MiniCheck", FakeJudge)
        monkeypatch.setattr(citation_module, "AlignScore", lambda cfg: pytest.fail("AlignScore"))
        cfg = MetricsConfig(device="cpu")
        metric = citation_module.CitationQuality(cfg)
        assert metric._threshold == cfg.minicheck.threshold


def _example(adapter_cfg):
    adapter = ASQATrainAdapter(adapter_cfg)
    return [adapter.as_example(row) for row in adapter.eligible_pool()]


class TestTrainingPrompt:
    def test_few_shot_prompt_sends_the_drawn_examples_as_earlier_turns(self, tmp_path):
        examples = _example(_data(tmp_path))
        dataset = training_dataset(examples, ASQA_ALCE_VANILLA, seed=2)
        turns = ASQA_ALCE_VANILLA.turns(2)
        assert len(turns) == 2
        for row, example in zip(dataset, examples, strict=True):
            expected = []
            for user, assistant in turns:
                expected += [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            expected.append({"role": "user", "content": ASQA_ALCE_VANILLA.render(example)})
            assert row["prompt"] == expected
            assert row["gold"] == example.gold

    def test_zero_shot_prompt_is_one_user_message_with_the_same_instruction(self, tmp_path):
        example = _example(_data(tmp_path))[0]
        assert ASQA_ALCE_ZERO_SHOT.instruction == ASQA_ALCE_VANILLA.instruction
        assert ASQA_ALCE_ZERO_SHOT.messages(example, 2) == [
            {"role": "user", "content": ASQA_ALCE_VANILLA.render(example)}
        ]

    def test_training_template_is_the_datasets_unless_named(self, tmp_path):
        cfg = _config(tmp_path, _data(tmp_path))
        assert training_template(cfg) is TEMPLATES["tier2_asqa_train"] is ASQA_ALCE_VANILLA
        named = dataclasses.replace(
            cfg, training=dataclasses.replace(cfg.training, prompt_template="asqa_alce_zero_shot")
        )
        assert training_template(named) is TEMPLATES_BY_NAME["asqa_alce_zero_shot"]

    def test_meter_counts_every_turn(self, tmp_path):
        from test_memory import tiny_tokenizer

        examples = _example(_data(tmp_path))
        template = "{% for m in messages %}{{ m['content'] }} {% endfor %}"
        meter = PromptLengths({"t": tiny_tokenizer(template)}, ASQA_ALCE_VANILLA, seed=2)
        expected = [
            sum(len(m["content"].split()) for m in ASQA_ALCE_VANILLA.messages(e, 2))
            for e in examples
        ]
        assert meter.lengths("t", examples) == expected
        zero = PromptLengths({"t": tiny_tokenizer(template)}, ASQA_ALCE_ZERO_SHOT)
        assert all(z < f for z, f in zip(zero.lengths("t", examples), expected, strict=True))


def _asqa_config(tmp_path, **training):
    data = _data(tmp_path)
    data = dataclasses.replace(data, sample_ids=ASQATrainAdapter(data).draw_sample_ids())
    cfg = _config(
        tmp_path,
        data,
        multi_objective_aggregation="normalize_then_sum",
        scale_rewards="group",
        worked_examples_seed=2,
        **training,
    )
    reward = RewardConfig(
        name=None, terms=("str_em", "citation_recall", "citation_precision"), weights=(1, 1, 1)
    )
    metrics = dataclasses.replace(cfg.metrics, citation=CitationConfig(judge="alignscore"))
    return dataclasses.replace(cfg, reward=reward, metrics=metrics)


class ASQAOfflineRun(OfflineRun):
    adapter_class = ASQATrainAdapter

    def _build_metric(self, name):
        self.built = [*getattr(self, "built", []), name]
        return StrEM(MetricsConfig(device="cpu")) if name == "str_em" else StubCitation()


class TestGRPOWiring:
    def test_gdpo_and_equal_weights_reach_trl(self, tmp_path):
        args = grpo_args(_asqa_config(tmp_path), tmp_path / "out")
        assert args.multi_objective_aggregation == "normalize_then_sum"
        assert args.reward_weights == [1.0, 1.0, 1.0]
        assert args.scale_rewards == "group"
        assert args.max_completion_length == 8

    def test_a_single_reward_keeps_trls_defaults(self, tmp_path):
        args = grpo_args(_config(tmp_path), tmp_path / "out")
        assert args.multi_objective_aggregation == "sum_then_normalize"
        assert args.reward_weights is None or list(args.reward_weights) == [1.0]

    def test_step_columns_add_each_terms_mean_and_spread(self):
        assert step_columns(["alignscore"]) == STEP_COLUMNS
        assert step_columns(["a", "b"])[len(STEP_COLUMNS) :] == (
            "rewards/a/mean",
            "rewards/a/std",
            "rewards/b/mean",
            "rewards/b/std",
        )

    def test_each_metric_is_built_once_and_shared_by_its_terms(self, tmp_path):
        run = ASQAOfflineRun(_asqa_config(tmp_path), BACKBONE, tmp_path)
        rewards = run._build_rewards()
        assert [r.name for r in rewards] == ["str_em", "citation_recall", "citation_precision"]
        assert run.built == ["str_em", "citation"]
        assert rewards[1]._metric is rewards[2]._metric

    def test_two_grpo_steps_with_three_terms(self, tmp_path, cpu_only):  # noqa: F811
        path = ASQAOfflineRun(_asqa_config(tmp_path), BACKBONE, tmp_path).run()
        summary = json.loads((path / "summary.json").read_text())
        assert summary["status"] == "success", summary["error"]
        assert summary["steps_completed"] == 2
        with (path / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert [row["step"] for row in rows] == ["1", "2"]
        for term in ("str_em", "citation_recall", "citation_precision"):
            assert all(row[f"rewards/{term}/mean"] != "" for row in rows)
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["reward"] == {
            "terms": [
                {"name": "str_em", "metric": "str_em"},
                {"name": "citation_recall", "metric": "citation"},
                {"name": "citation_precision", "metric": "citation"},
            ],
            "weights": [1.0, 1.0, 1.0],
            "aggregation": "normalize_then_sum",
            "citation_judge": "alignscore",
        }
        drawn = ",".join(e.source_id for e in ASQA_ALCE_VANILLA.drawn_examples(2))
        assert manifest["prompt_templates"] == {
            "generator": {
                "asqa_alce_vanilla": "asqa_alce_vanilla@v1",
                "asqa_alce_vanilla_examples": drawn,
            }
        }
        assert manifest["grpo_args"]["multi_objective_aggregation"] == "normalize_then_sum"


class TestShippedMemoryConfigs:
    @pytest.mark.parametrize(
        "setting, group, template, seed",
        [("fewshot", 4, None, 2), ("zeroshot", 8, "asqa_alce_zero_shot", None)],
    )
    def test_settings_resolve_to_the_approved_values(self, setting, group, template, seed):
        cfg = load_config(CONFIGS / "rl" / f"grpo_asqa_memory_{setting}.yaml", ["run.owner=sameer"])
        assert cfg.name == f"grpo_asqa_memory_{setting}"
        assert cfg.data.name == "tier2_asqa_train" and cfg.data.n_samples == 10
        assert cfg.reward.terms == ("str_em", "citation_recall", "citation_precision")
        assert cfg.reward.weights == (1.0, 1.0, 1.0)
        assert cfg.metrics.citation.judge == "alignscore"
        assert cfg.metrics.alignscore.threshold == 0.5
        t = cfg.training
        assert (t.num_generations, t.per_device_train_batch_size) == (group, group)
        assert (t.multi_objective_aggregation, t.scale_rewards) == ("normalize_then_sum", "group")
        assert (t.prompt_template, t.worked_examples_seed) == (template, seed)
        assert (t.gradient_accumulation_steps, t.max_steps, t.loss_type) == (1, 10, "dr_grpo")
        assert cfg.sampling.max_tokens == 300
        m = cfg.measurement
        assert m.group_sizes == (group,) and m.typical_group_size == group
        assert m.prompt_caps == (4096,) and m.forced_cap == 4096
        assert m.backbones == (
            "llama-3.1-8b-instruct",
            "mistral-7b-instruct-v0.3",
            "qwen2.5-7b-instruct",
        )

    def test_seed_two_draws_the_longest_pair(self):
        def words(seed):
            return sum(len(u.split()) + len(a.split()) for u, a in ASQA_ALCE_VANILLA.turns(seed))

        assert words(2) == max(words(seed) for seed in range(3))


class TestPerQuestionDraws:
    """D-139: each training question draws its own pair of worked examples."""

    def test_question_seed_is_stable_in_range_and_depends_on_both_inputs(self):
        assert question_seed(0, "q1") == question_seed(0, "q1")
        assert len({question_seed(0, "q1"), question_seed(1, "q1"), question_seed(0, "q2")}) == 3
        assert all(0 <= question_seed(s, str(i)) <= MAX_SEED for s in (0, 7) for i in range(50))

    def test_each_question_sends_its_own_draw_whatever_the_order(self, tmp_path):
        examples = _example(_data(tmp_path))
        dataset = training_dataset(examples, ASQA_ALCE_VANILLA, seed=0, per_question=True)
        for row, example in zip(dataset, examples, strict=True):
            assert row["prompt"] == ASQA_ALCE_VANILLA.messages(
                example, question_seed(0, example.id)
            )
        reversed_rows = training_dataset(
            examples[::-1], ASQA_ALCE_VANILLA, seed=0, per_question=True
        )
        assert list(reversed_rows["prompt"]) == list(dataset["prompt"])[::-1]

    def test_draws_vary_between_questions(self):
        pairs = {
            tuple(e.source_id for e in ASQA_ALCE_VANILLA.drawn_examples(question_seed(0, str(i))))
            for i in range(200)
        }
        assert len(pairs) == 12  # every ordered pair of the four examples

    def test_per_run_stays_the_default(self, tmp_path):
        examples = _example(_data(tmp_path))
        dataset = training_dataset(examples, ASQA_ALCE_VANILLA, seed=2)
        assert {len(p) for p in dataset["prompt"]} == {5}
        assert all(p[:4] == dataset["prompt"][0][:4] for p in dataset["prompt"])
        assert TrainingConfig(**SMOKE_UNDECIDED).worked_examples_draw == "per_run"

    def test_an_unknown_draw_is_refused(self):
        with pytest.raises(ConfigError, match="worked_examples_draw"):
            TrainingConfig(**SMOKE_UNDECIDED, worked_examples_draw="per_step")

    def test_the_manifest_counts_each_pairs_questions(self, tmp_path, cpu_only):  # noqa: F811
        cfg = _asqa_config(tmp_path, worked_examples_draw="per_question")
        path = ASQAOfflineRun(cfg, BACKBONE, tmp_path).run()
        manifest = json.loads((path / "manifest.json").read_text())
        prompt = manifest["prompt_templates"]["generator"]
        assert prompt["asqa_alce_vanilla_examples"] == "per_question"
        draws = prompt["asqa_alce_vanilla_example_draws"]
        assert sum(draws.values()) == manifest["n_examples"]
        assert all(len(pair.split(",")) == 2 for pair in draws)


class TestShippedTrainingConfig:
    def test_resolves_to_the_approved_values(self):
        cfg = load_config(CONFIGS / "rl" / "grpo_asqa.yaml", ["run.owner=sameer"])
        assert cfg.name == "grpo_asqa"
        assert cfg.data.name == "tier2_asqa_train"
        assert len(cfg.data.sample_ids) == len(set(cfg.data.sample_ids)) == 4349
        assert cfg.reward.terms == ("str_em", "citation_recall", "citation_precision")
        assert cfg.reward.weights == (1.0, 1.0, 1.0)
        assert cfg.metrics.citation.judge == "alignscore"
        t = cfg.training
        assert (t.num_generations, t.per_device_train_batch_size) == (4, 4)
        assert (t.multi_objective_aggregation, t.scale_rewards) == ("normalize_then_sum", "group")
        assert (t.prompt_template, t.worked_examples_draw) == (None, "per_question")
        assert (t.gradient_accumulation_steps, t.max_steps, t.save_steps) == (16, 271, 50)
        assert (t.learning_rate, t.loss_type) == (1e-5, "dr_grpo")
        assert cfg.sampling.max_tokens == 300
        # One pass: no question is used twice (D-058, D-139).
        assert t.max_steps * t.gradient_accumulation_steps <= len(cfg.data.sample_ids)


class PairScoreStub(AlignScore):
    """AlignScore's judge path without the model: records every pair it would score."""

    def __init__(self, cfg):
        self.pairs = []

    def _pair_score(self, document, summary):
        self.pairs.append((document, summary))
        return 1.0


class TestEmptyClaims:
    """D-137: a sentence that is only citation markers leaves an empty claim."""

    def test_alignscore_scores_an_empty_claim_zero_without_the_model(self):
        judge = PairScoreStub(None)
        assert judge.claim_support("Some passage.", "") == 0.0
        assert judge.claim_support("Some passage.", "  \n") == 0.0
        assert judge.pairs == []
        assert judge.claim_support("Some passage.", "A claim.") == 1.0

    def test_a_marker_only_sentence_counts_as_unsupported(self, monkeypatch):
        monkeypatch.setattr(citation_module, "AlignScore", PairScoreStub)
        cfg = MetricsConfig(device="cpu", citation=CitationConfig(judge="alignscore"))
        metric = citation_module.CitationQuality(cfg)
        evidence = format_passages([{"title": "T", "text": "P1."}, {"title": "U", "text": "P2."}])
        rows = metric.score(["Paris is the capital [1]. [2]"], ["r"], [evidence])
        assert rows == [{"citation_recall": 0.5, "citation_precision": 0.5}]
        assert [claim for _, claim in metric._judge.pairs] == ["Paris is the capital."]
