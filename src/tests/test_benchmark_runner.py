"""The benchmark runner end to end with a stub dataset, client, server and metric. CPU, offline."""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from agent_evals.agents import generator as generator_module
from agent_evals.agents import self_feedback as self_feedback_module
from agent_evals.benchmarks import BenchmarkCell, BenchmarkError, BenchmarkRunner
from agent_evals.benchmarks.runner import main
from agent_evals.config import (
    DataConfig,
    ExperimentConfig,
    MetricsConfig,
    RunConfig,
    ServingConfig,
    TopologyConfig,
)
from agent_evals.datasets import DatasetAdapter
from agent_evals.metrics import Metric
from agent_evals.registry import DATASETS, METRICS, SERVING, TOPOLOGIES
from agent_evals.run import RunDirectory
from agent_evals.serving import ChatClient, Completion, VLLMServer
from agent_evals.topologies import Topology, TopologyError
from agent_evals.types import Message, TokenCost

REPO_ROOT = Path(__file__).resolve().parents[2]
SIX_FILES = {
    "config.resolved.yaml",
    "manifest.json",
    "run.log",
    "metrics.csv",
    "trace.jsonl",
    "summary.json",
}
T1_TOPOLOGY = TopologyConfig(name="t1_self_refine", max_rounds=4)
ARTICLES = {
    "a": "The council voted to close the library in March after repairs were costed.",
    "b": "Blank me: the model returns nothing for this article about a football match.",
    "c": "Long one: the model runs into the completion cap on this article about rainfall.",
}


class StubAdapter(DatasetAdapter):
    TIER = 1
    TASK = "summarization"

    def _load_raw(self):
        return Dataset.from_list(
            [{"id": k, "document": v, "summary": f"Reference {k}."} for k, v in ARTICLES.items()]
        )

    def _map_row(self, row):
        return {
            "id": row["id"],
            "source": row["document"],
            "evidence": row["document"],
            "reference": row["summary"],
            "question": None,
        }


class StubClient(ChatClient):
    calls: list[tuple[str, int | None, str | None]] = []

    def complete(self, prompt, seed=None, system=None, turns=()):
        StubClient.calls.append((prompt, seed, system))
        text, finish = "A short summary, with a comma.", "stop"
        if "Blank me" in prompt:
            text = "   "
        if prompt.startswith(self_feedback_module.XSUM_SELF_FEEDBACK.instruction):
            text = "NONE"
        if "Long one" in prompt:
            finish = "length"
        return Completion(text, TokenCost(100, 10), 0.25, finish)


class NoServer(VLLMServer):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class CharCount(Metric):
    KEYS = ("stub_chars",)

    def _score_examples(self, predictions, references, evidence, gold):
        return [{"stub_chars": float(len(p))} for p in predictions]


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    StubClient.calls = []
    monkeypatch.setattr(BenchmarkRunner, "server_class", NoServer)
    monkeypatch.setitem(generator_module.TEMPLATES, "stub_task", generator_module.XSUM_ONE_SENTENCE)
    (tmp_path / "model_registry.csv").write_text(
        "name,hf_path,hf_revision,gated,quantization,max_model_len\n"
        f"stub-backbone,org/stub,{'a' * 40},false,bitsandbytes,2809\n",
        encoding="utf-8",
    )
    with (
        DATASETS.temporary("stub_task", StubAdapter),
        SERVING.temporary("stub_client", StubClient),
        METRICS.temporary("stub_metric", CharCount),
    ):
        yield tmp_path


@pytest.fixture
def t1_templates(monkeypatch):
    """Give the stub task T1's XSum feedback and refine templates."""
    monkeypatch.setitem(
        self_feedback_module.FEEDBACK_TEMPLATES,
        "stub_task",
        self_feedback_module.XSUM_SELF_FEEDBACK,
    )
    monkeypatch.setitem(
        generator_module.REFINE_TEMPLATES, "stub_task", generator_module.XSUM_SELF_REFINE
    )


class DraftThenCheck(Topology):
    """Ends on a check message whose finish reason differs from the output message's."""

    ROLES = ("generator",)

    def run(self, example, agents):
        started = time.perf_counter()
        draft = Message("generator", "the draft", 0, TokenCost(1, 1), 0.0, "length")
        check = Message("checker", "NONE", 0, TokenCost(1, 1), 0.0, "stop")
        return self._assemble(draft, (draft, check), n_rounds=1, converged=True, started=started)


def _cfg(seed=1, topology=None):
    return ExperimentConfig(
        topology=topology,
        name="unittest",
        run=RunConfig(owner="sameer", seed=seed),
        data=DataConfig(
            name="stub_task",
            n_samples=3,
            n_strata=1,
            min_source_words=0,
            sample_ids=("a", "b", "c"),
        ),
        metrics=MetricsConfig(device="cpu", suite=("stub_metric",)),
        serving=ServingConfig(client="stub_client", model_registry="model_registry.csv"),
    )


CELL = BenchmarkCell(topology="t0_single_agent", backbone="stub-backbone", task="stub_task")


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestRunnerEndToEnd:
    def test_writes_the_six_files_under_results(self, stubs):
        path = BenchmarkRunner(_cfg(), stubs).run(CELL)
        assert path.parent == stubs / "results"
        assert {p.name for p in path.iterdir()} == SIX_FILES
        assert path.name.endswith("_t0_single_agent_stub_task_stub-backbone_seed1")

    def test_one_row_per_example_in_frozen_order_with_every_column(self, stubs):
        rows = _rows(BenchmarkRunner(_cfg(), stubs).run(CELL) / "metrics.csv")
        assert [row["example_id"] for row in rows] == ["a", "b", "c"]
        for column in (
            "run_id",
            "output",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "wall_clock_s",
            "n_rounds",
            "converged",
            "source_words",
            "finish_reason",
            "hit_length_cap",
            "blank_output",
            "stub_chars",
        ):
            assert column in rows[0], column
        assert rows[0]["output"] == "A short summary, with a comma."

    def test_blank_output_is_recorded_but_not_scored(self, stubs):
        rows = {
            r["example_id"]: r
            for r in _rows(BenchmarkRunner(_cfg(), stubs).run(CELL) / "metrics.csv")
        }
        assert rows["b"]["blank_output"] == "True"
        assert rows["b"]["stub_chars"] == ""
        assert rows["a"]["stub_chars"] == str(float(len("A short summary, with a comma.")))

    def test_length_cap_is_recorded_per_row(self, stubs):
        rows = {
            r["example_id"]: r
            for r in _rows(BenchmarkRunner(_cfg(), stubs).run(CELL) / "metrics.csv")
        }
        assert rows["c"]["finish_reason"] == "length"
        assert rows["c"]["hit_length_cap"] == "True"
        assert rows["a"]["hit_length_cap"] == "False"

    def test_every_request_carries_the_run_seed_and_the_d016_prompt(self, stubs):
        BenchmarkRunner(_cfg(seed=2), stubs).run(CELL)
        assert len(StubClient.calls) == 3
        assert {seed for _, seed, _ in StubClient.calls} == {2}
        assert all(
            p.startswith("Summarize the following article in one sentence:\n\n")
            for p, _, _ in StubClient.calls
        )
        assert {system for _, _, system in StubClient.calls} == {
            None
        }, "T0 sends no system prompt (D-016, restored by D-045)"

    def test_manifest_records_cell_backbone_and_prompt_version(self, stubs):
        path = BenchmarkRunner(_cfg(), stubs).run(CELL)
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["cell"] == {
            "topology": "t0_single_agent",
            "backbone": "stub-backbone",
            "task": "stub_task",
        }
        assert manifest["backbone"]["hf_revision"] == "a" * 40
        assert manifest["prompt_templates"] == {
            "generator": {"xsum_one_sentence": "xsum_one_sentence@v1"}
        }
        assert manifest["metric_keys"] == ["stub_chars"]

    def test_summary_carries_totals_rates_and_metric_means(self, stubs):
        summary = json.loads(
            (BenchmarkRunner(_cfg(), stubs).run(CELL) / "summary.json").read_text()
        )
        assert summary["status"] == "success"
        assert summary["n_examples"] == 3
        assert summary["n_blank_outputs"] == 1
        assert summary["length_cap_rate"] == pytest.approx(1 / 3)
        assert summary["total_tokens"] == 330
        assert summary["metric_means"]["stub_chars"] == pytest.approx(
            float(len("A short summary, with a comma."))
        )

    def test_config_for_another_task_is_refused(self, stubs):
        other = BenchmarkCell(topology="t0_single_agent", backbone="stub-backbone", task="other")
        with pytest.raises(BenchmarkError, match="task"):
            BenchmarkRunner(_cfg(), stubs).run(other)

    def test_topology_group_for_another_topology_is_refused(self, stubs):
        with pytest.raises(BenchmarkError, match="topology"):
            BenchmarkRunner(_cfg(topology=T1_TOPOLOGY), stubs).run(CELL)

    def test_a_topology_without_a_group_records_none(self, stubs):
        path = BenchmarkRunner(_cfg(), stubs).run(CELL)
        assert yaml.safe_load((path / "config.resolved.yaml").read_text())["topology"] is None

    def test_trace_has_one_line_per_message_labelled_with_the_example(self, stubs):
        path = BenchmarkRunner(_cfg(), stubs).run(CELL)
        lines = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
        assert [(r["example_id"], r["message_index"], r["role"]) for r in lines] == [
            ("a", 0, "generator"),
            ("b", 0, "generator"),
            ("c", 0, "generator"),
        ]
        assert {r["run_id"] for r in lines} == {path.name}

    def test_length_cap_comes_from_the_output_message_not_the_last_message(self, stubs):
        cell = BenchmarkCell(
            topology="draft_then_check", backbone="stub-backbone", task="stub_task"
        )
        with TOPOLOGIES.temporary("draft_then_check", DraftThenCheck):
            path = BenchmarkRunner(_cfg(), stubs).run(cell)
        rows = _rows(path / "metrics.csv")
        assert {r["finish_reason"] for r in rows} == {"length"}
        assert {r["hit_length_cap"] for r in rows} == {"True"}
        assert json.loads((path / "summary.json").read_text())["length_cap_rate"] == 1.0


CITED_OUTPUTS = {
    # id: (output, completion tokens of the call that wrote it)
    "a": ("Paris is the capital [1][2].\nIt is on the Seine [3].", 7),
    "b": ("[1][2]", 3),
    "c": ("A plain answer.", 5),
}


class GoldAdapter(StubAdapter):
    """A grounded-QA stub whose examples carry gold annotations (D-126)."""

    TIER = 2
    TASK = "grounded_qa"

    def _map_row(self, row):
        position = sorted(ARTICLES).index(row["id"]) + 1
        return {
            **super()._map_row(row),
            "question": f"Question {row['id']}?",
            "gold": json.dumps({"n": position}),
        }


class CitedAnswer(Topology):
    """Answers with ALCE-style citation markers; one output is nothing but markers."""

    ROLES = ()

    def run(self, example, agents):
        text, completion = CITED_OUTPUTS[example.id]
        started = time.perf_counter()
        message = Message("generator", text, 0, TokenCost(100, completion), 0.0, "stop")
        return self._assemble(message, (message,), n_rounds=1, converged=True, started=started)


class RawCharCount(Metric):
    KEYS = ("raw_chars",)
    READS_CITATIONS = True

    def _score_examples(self, predictions, references, evidence, gold):
        return [{"raw_chars": float(len(p))} for p in predictions]


class GoldNumber(Metric):
    KEYS = ("gold_n",)
    NEEDS_GOLD = True

    def _score_examples(self, predictions, references, evidence, gold):
        return [{"gold_n": float(json.loads(g)["n"])} for g in gold]


class TestCitationsGoldAndOutputStatistics:
    """D-126: citation stripping per metric, gold passed to metrics, two new summary fields."""

    CELL = BenchmarkCell(topology="cited_answer", backbone="stub-backbone", task="cite_task")

    @pytest.fixture
    def cited(self, stubs):
        with (
            DATASETS.temporary("cite_task", GoldAdapter),
            TOPOLOGIES.temporary("cited_answer", CitedAnswer),
            METRICS.temporary("raw_metric", RawCharCount),
            METRICS.temporary("gold_metric", GoldNumber),
        ):
            yield stubs

    @staticmethod
    def _run(root, strip_citations):
        cfg = _cfg()
        cfg = dataclasses.replace(
            cfg,
            data=dataclasses.replace(cfg.data, name="cite_task", strip_citations=strip_citations),
            metrics=MetricsConfig(device="cpu", suite=("stub_metric", "raw_metric", "gold_metric")),
        )
        path = BenchmarkRunner(cfg, root).run(TestCitationsGoldAndOutputStatistics.CELL)
        rows = {r["example_id"]: r for r in _rows(path / "metrics.csv")}
        return rows, json.loads((path / "summary.json").read_text())

    def test_metrics_score_the_answer_without_markers_unless_they_read_citations(self, cited):
        rows, _ = self._run(cited, strip_citations=True)
        answer = "Paris is the capital.\nIt is on the Seine."
        assert rows["a"]["stub_chars"] == str(float(len(answer)))
        assert rows["a"]["raw_chars"] == str(float(len(CITED_OUTPUTS["a"][0])))

    def test_an_output_of_markers_only_is_a_blank_output_when_stripping(self, cited):
        rows, summary = self._run(cited, strip_citations=True)
        assert rows["b"]["blank_output"] == "True"
        assert rows["b"]["stub_chars"] == rows["b"]["raw_chars"] == ""
        assert summary["n_blank_outputs"] == 1

    def test_without_stripping_every_metric_scores_the_output_as_written(self, cited):
        rows, summary = self._run(cited, strip_citations=False)
        assert rows["a"]["stub_chars"] == rows["a"]["raw_chars"]
        assert rows["b"]["blank_output"] == "False"
        assert summary["n_blank_outputs"] == 0

    def test_gold_reaches_the_metric_for_each_scored_example(self, cited):
        rows, _ = self._run(cited, strip_citations=True)
        assert (rows["a"]["gold_n"], rows["b"]["gold_n"], rows["c"]["gold_n"]) == ("1.0", "", "3.0")

    def test_summary_reports_newline_rate_and_mean_output_tokens(self, cited):
        _, summary = self._run(cited, strip_citations=True)
        assert summary["newline_rate"] == pytest.approx(1 / 3)
        assert summary["mean_output_tokens"] == pytest.approx(5.0)

    def test_output_tokens_come_from_the_output_message_not_the_last(self, stubs):
        cell = BenchmarkCell(
            topology="draft_then_check", backbone="stub-backbone", task="stub_task"
        )
        with TOPOLOGIES.temporary("draft_then_check", DraftThenCheck):
            path = BenchmarkRunner(_cfg(), stubs).run(cell)
        summary = json.loads((path / "summary.json").read_text())
        assert summary["mean_output_tokens"] == 1.0
        assert summary["newline_rate"] == 0.0


@pytest.mark.usefixtures("t1_templates")
class TestSelfRefineCell:
    T1 = BenchmarkCell(topology="t1_self_refine", backbone="stub-backbone", task="stub_task")

    def test_max_rounds_and_every_template_are_recorded(self, stubs):
        path = BenchmarkRunner(_cfg(topology=T1_TOPOLOGY), stubs).run(self.T1)
        resolved = yaml.safe_load((path / "config.resolved.yaml").read_text())
        assert resolved["topology"] == {
            "name": "t1_self_refine",
            "max_rounds": 4,
            "requires_adapter": False,
        }
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["prompt_templates"] == {
            "generator": {
                "xsum_one_sentence": "xsum_one_sentence@v1",
                "xsum_self_refine": "xsum_self_refine@v2",
            },
            "self_feedback": {"xsum_self_feedback": "xsum_self_feedback@v3"},
        }

    def test_trace_lines_match_each_examples_trace(self, stubs):
        path = BenchmarkRunner(_cfg(topology=T1_TOPOLOGY), stubs).run(self.T1)
        lines = [json.loads(line) for line in (path / "trace.jsonl").read_text().splitlines()]
        assert [
            (r["example_id"], r["message_index"], r["role"], r["round_index"]) for r in lines
        ] == [
            (example_id, index, role, 0)
            for example_id in ("a", "b", "c")
            for index, role in enumerate(("generator", "self_feedback"))
        ]
        assert [r["content"] for r in lines if r["example_id"] == "a"] == [
            "A short summary, with a comma.",
            "NONE",
        ]
        assert {r["total_tokens"] for r in lines} == {110}

    def test_stopping_on_none_keeps_the_draft_as_the_output(self, stubs):
        rows = _rows(
            BenchmarkRunner(_cfg(topology=T1_TOPOLOGY), stubs).run(self.T1) / "metrics.csv"
        )
        assert len(StubClient.calls) == 6
        assert {r["n_rounds"] for r in rows} == {"1"}
        assert {r["converged"] for r in rows} == {"True"}
        assert rows[0]["output"] == "A short summary, with a comma."

    def test_without_max_rounds_the_cell_is_refused(self, stubs):
        with pytest.raises(TopologyError, match="max_rounds"):
            BenchmarkRunner(_cfg(), stubs).run(self.T1)


class TestRunDirectoryExtras:
    def test_manifest_extra_and_summary_fields_are_written(self, cfg, tmp_path):
        run = RunDirectory(cfg, tmp_path, datetime(2026, 9, 13), manifest_extra={"cell": "x"})
        with run:
            run.add_summary(n_examples=3)
        assert json.loads((run.path / "manifest.json").read_text())["cell"] == "x"
        assert json.loads((run.path / "summary.json").read_text())["n_examples"] == 3

    def test_built_in_keys_cannot_be_replaced(self, cfg, tmp_path):
        with pytest.raises(ValueError, match="manifest"):
            RunDirectory(cfg, tmp_path, manifest_extra={"git": "fake"})
        with RunDirectory(cfg, tmp_path) as run, pytest.raises(ValueError, match="summary"):
            run.add_summary(status="success")


class TestRLAgentCell:
    T5 = BenchmarkCell(topology="t5_rl_agent", backbone="stub-backbone", task="stub_task")
    TOPOLOGY = TopologyConfig(name="t5_rl_agent", requires_adapter=True)

    def _cfg(self, adapters):
        cfg = _cfg(topology=self.TOPOLOGY)
        serving = ServingConfig(
            client="stub_client", model_registry="model_registry.csv", lora_adapters=adapters
        )
        return dataclasses.replace(cfg, serving=serving)

    def test_refused_without_an_adapter_before_anything_runs(self, stubs):
        with pytest.raises(BenchmarkError, match="requires a LoRA adapter"):
            BenchmarkRunner(self._cfg(()), stubs).run(self.T5)
        assert StubClient.calls == []

    def test_refused_when_the_adapter_has_no_weights(self, stubs, tmp_path):
        with pytest.raises(BenchmarkError, match="no LoRA adapter weights"):
            BenchmarkRunner(self._cfg((f"stub-backbone={tmp_path}",)), stubs).run(self.T5)

    def test_adapter_path_and_weights_hash_are_recorded(self, stubs, tmp_path):
        (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
        path = BenchmarkRunner(self._cfg((f"stub-backbone={tmp_path}",)), stubs).run(self.T5)
        manifest = json.loads((path / "manifest.json").read_text())
        assert manifest["lora_adapter"] == {
            "path": str(tmp_path),
            "sha256": hashlib.sha256(b"weights").hexdigest(),
        }
        assert manifest["cell"]["topology"] == "t5_rl_agent"
        assert len(StubClient.calls) == 3

    def test_runs_without_an_adapter_record_none(self, stubs):
        path = BenchmarkRunner(_cfg(), stubs).run(CELL)
        assert json.loads((path / "manifest.json").read_text())["lora_adapter"] is None


def _configs_dir(root):
    """A configs tree for the stub task, with a topology group file for T1 only."""
    configs = root / "configs"
    (configs / "data").mkdir(parents=True)
    (configs / "topology").mkdir()
    base = {
        "run": {"owner": "sameer", "seed": 1},
        "metrics": {"device": "cpu", "suite": ["stub_metric"]},
        "serving": {"client": "stub_client", "model_registry": "model_registry.csv"},
    }
    (configs / "base.yaml").write_text(yaml.safe_dump(base))
    data = {
        "defaults": ["../base"],
        "data": {
            "name": "stub_task",
            "n_samples": 3,
            "n_strata": 1,
            "min_source_words": 0,
            "sample_ids": ["a", "b", "c"],
        },
    }
    (configs / "data" / "stub_task.yaml").write_text(yaml.safe_dump(data))
    topology = {"topology": {"name": "t1_self_refine", "max_rounds": 3}}
    (configs / "topology" / "t1_self_refine.yaml").write_text(yaml.safe_dump(topology))
    return configs


@pytest.mark.usefixtures("t1_templates")
def test_main_merges_a_topology_group_file_only_into_its_own_cells(stubs):
    configs = _configs_dir(stubs)
    for topology in ("t1_self_refine", "t0_single_agent"):
        argv = ["--topology", topology, "--backbone", "stub-backbone", "--task", "stub_task"]
        assert main(argv, stubs, configs) == 0
    groups = {}
    for path in (stubs / "results").iterdir():
        cell = json.loads((path / "manifest.json").read_text())["cell"]
        groups[cell["topology"]] = yaml.safe_load((path / "config.resolved.yaml").read_text())[
            "topology"
        ]
    assert groups == {
        "t1_self_refine": {"name": "t1_self_refine", "max_rounds": 3, "requires_adapter": False},
        "t0_single_agent": None,
    }


def test_unknown_selector_fails_before_anything_runs():
    configs = REPO_ROOT / "src" / "configs"
    assert (
        main(["--topology", "nope", "--backbone", "all", "--task", "all"], REPO_ROOT, configs) == 1
    )
