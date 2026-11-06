"""The reproducibility contract: six files, every exit path, and the cost/latency gate."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime

import pytest
import yaml
from agent_evals.config import load_config
from agent_evals.run import (
    STATUS_FAILED,
    STATUS_OOM,
    STATUS_SUCCESS,
    MetricsWriter,
    RunDirectory,
    git_info,
    gpu_info,
    make_run_id,
)
from agent_evals.types import Message, TokenCost, TopologyResult

SIX_FILES = {
    "config.resolved.yaml",
    "manifest.json",
    "run.log",
    "metrics.csv",
    "trace.jsonl",
    "summary.json",
}
TRACE_KEYS = [
    "run_id",
    "example_id",
    "message_index",
    "role",
    "round_index",
    "content",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "latency_s",
    "finish_reason",
]


def _summary(run):
    return json.loads((run.path / "summary.json").read_text())


def _rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestRunId:
    def test_format_is_timestamp_name_seed(self):
        run_id = make_run_id("t0_baseline", 2, now=datetime(2026, 9, 10, 14, 30, 5))
        assert run_id == "20260910_143005_t0_baseline_seed2"

    def test_generated_ids_match_the_documented_shape(self, cfg):
        assert re.fullmatch(r"\d{8}_\d{6}_unittest_seed0", make_run_id(cfg.name, cfg.run.seed))


class TestRunDirectoryOnSuccess:
    def test_writes_exactly_the_six_contract_files(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            run.metrics.write({"example_id": "x", "score": 1.0})
        assert {p.name for p in run.path.iterdir()} == SIX_FILES

    def test_directory_lands_under_the_configured_output_root(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        assert run.path.parent == tmp_path / "results"

    def test_status_is_success(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        assert _summary(run)["status"] == STATUS_SUCCESS

    def test_resolved_config_alone_can_rebuild_the_config(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        assert load_config(run.path / "config.resolved.yaml") == cfg

    def test_summary_carries_owner_seed_and_wall_clock(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        summary = _summary(run)
        assert summary["owner"] == "sameer"
        assert summary["seed"] == 0
        assert summary["wall_clock_s"] >= 0


class TestManifest:
    def test_carries_every_provenance_field(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        manifest = json.loads((run.path / "manifest.json").read_text())
        for key in (
            "run_id",
            "owner",
            "git",
            "seeding",
            "gpu",
            "libraries",
            "python",
            "platform",
            "hostname",
            "started_at",
        ):
            assert key in manifest, f"manifest is missing {key}"

    def test_records_the_seed_that_was_actually_set(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        manifest = json.loads((run.path / "manifest.json").read_text())
        assert manifest["seeding"]["seed"] == cfg.run.seed

    def test_git_info_reports_dirty_and_untracked_separately(self):
        info = git_info()
        assert set(info) == {"sha", "branch", "dirty", "untracked_files"}
        assert isinstance(info["dirty"], bool)

    def test_gpu_info_is_populated_even_with_no_gpu(self):
        info = gpu_info()
        assert set(info) == {"available", "name", "total_vram_gib", "driver", "cuda"}
        assert isinstance(info["available"], bool)


class TestFailurePaths:
    def test_failure_still_writes_all_six_files(self, cfg, tmp_path):
        with pytest.raises(RuntimeError), RunDirectory(cfg, repo_root=tmp_path) as run:
            raise RuntimeError("boom")
        assert {p.name for p in run.path.iterdir()} == SIX_FILES

    def test_failure_records_status_and_traceback(self, cfg, tmp_path):
        with pytest.raises(RuntimeError), RunDirectory(cfg, repo_root=tmp_path) as run:
            raise RuntimeError("boom")
        summary = _summary(run)
        assert summary["status"] == STATUS_FAILED
        assert summary["error"]["type"] == "RuntimeError"
        assert "boom" in summary["error"]["traceback"]

    def test_exception_is_not_swallowed(self, cfg, tmp_path):
        with pytest.raises(ValueError, match="propagate"), RunDirectory(cfg, repo_root=tmp_path):
            raise ValueError("must propagate")

    def test_oom_is_a_status_not_a_crash(self, cfg, tmp_path):
        with pytest.raises(RuntimeError), RunDirectory(cfg, repo_root=tmp_path) as run:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        summary = _summary(run)
        assert summary["status"] == STATUS_OOM
        assert summary["error"]["type"] == "RuntimeError"

    def test_metrics_written_before_a_crash_survive_it(self, cfg, tmp_path):
        with pytest.raises(RuntimeError), RunDirectory(cfg, repo_root=tmp_path) as run:
            run.metrics.write({"example_id": "a"})
            run.metrics.write({"example_id": "b"})
            raise RuntimeError("boom")
        rows = _rows(run.path / "metrics.csv")
        assert [row["example_id"] for row in rows] == ["a", "b"]
        assert _summary(run)["records_written"] == 2


class TestResultGate:
    def test_records_a_valid_result(self, cfg, tmp_path, result):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            run.record_result(result, example_id="x")
        (record,) = _rows(run.path / "metrics.csv")
        assert record["example_id"] == "x"
        assert record["prompt_tokens"] == "800"
        assert record["completion_tokens"] == "60"
        assert record["total_tokens"] == "860"
        assert record["wall_clock_s"] == "1.25"

    def test_metrics_columns_keep_their_order_with_example_id_after_n_messages(
        self, cfg, tmp_path, result
    ):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            run.record_result(result, example_id="x", source_words=3)
        header = (run.path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == [
            "run_id",
            "output",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "wall_clock_s",
            "n_rounds",
            "converged",
            "n_messages",
            "example_id",
            "source_words",
        ]

    def test_trace_holds_one_json_line_per_message_in_order(self, cfg, tmp_path):
        draft = Message("generator", "a draft – Zürich", 0, TokenCost(100, 10), 0.5, "stop")
        check = Message("self_feedback", '- "draft"', 0, TokenCost(200, 5), 0.25, "stop")
        final = Message("generator", "a summary", 1, TokenCost(300, 12), 0.75, "length")
        multi = TopologyResult(
            output="a summary",
            token_cost=TokenCost(600, 27),
            wall_clock_s=1.5,
            n_rounds=2,
            converged=False,
            trace=(draft, check, final),
            finish_reason="length",
        )
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            run.record_result(multi, example_id="x")
            run.record_result(multi, example_id="y")
        text = (run.path / "trace.jsonl").read_text(encoding="utf-8")
        assert "Zürich" in text
        records = [json.loads(line) for line in text.splitlines()]
        assert len(records) == 6
        assert all(list(r) == TRACE_KEYS for r in records)
        assert [(r["example_id"], r["message_index"]) for r in records] == [
            ("x", 0),
            ("x", 1),
            ("x", 2),
            ("y", 0),
            ("y", 1),
            ("y", 2),
        ]
        assert records[2] == {
            "run_id": run.run_id,
            "example_id": "x",
            "message_index": 2,
            "role": "generator",
            "round_index": 1,
            "content": "a summary",
            "prompt_tokens": 300,
            "completion_tokens": 12,
            "total_tokens": 312,
            "latency_s": 0.75,
            "finish_reason": "length",
        }

    def test_rejects_anything_that_is_not_a_topology_result(self, cfg, tmp_path):
        with (
            pytest.raises(TypeError, match="TopologyResult"),
            RunDirectory(cfg, repo_root=tmp_path) as run,
        ):
            run.record_result({"output": "not a result"}, example_id="x")

    def test_metrics_unavailable_before_entering(self, cfg, tmp_path):
        run = RunDirectory(cfg, repo_root=tmp_path)
        with pytest.raises(RuntimeError, match="must be entered"):
            run.metrics.write({"x": 1})


class TestMetricsWriter:
    def test_first_record_fixes_the_header_and_later_key_order_is_free(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        writer.write({"example_id": "a", "score": 0.5})
        writer.write({"score": 0.25, "example_id": "b"})
        writer.close()
        lines = (tmp_path / "metrics.csv").read_text().splitlines()
        assert lines == ["example_id,score", "a,0.5", "b,0.25"]

    def test_counts_records(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        for i in range(3):
            writer.write({"i": i})
        assert writer.count == 3
        writer.close()

    def test_flush_every_record_means_data_is_on_disk_immediately(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv", flush_every=1)
        writer.write({"a": 1})
        assert (tmp_path / "metrics.csv").read_text().strip() != ""
        writer.close()

    def test_changed_columns_are_rejected(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        writer.write({"a": 1})
        with pytest.raises(ValueError, match="columns"):
            writer.write({"a": 1, "b": 2})
        writer.close()

    def test_nested_values_are_rejected_rather_than_stringified(self, tmp_path):
        writer = MetricsWriter(tmp_path / "metrics.csv")
        with pytest.raises(TypeError, match="cost"):
            writer.write({"cost": TokenCost(1, 2)})
        writer.close()

    def test_commas_quotes_newlines_unicode_and_none_survive(self, tmp_path):
        text = 'Zürich, "quoted"\nsecond line – €'
        writer = MetricsWriter(tmp_path / "metrics.csv")
        writer.write({"output": text, "missing": None})
        writer.close()
        assert _rows(tmp_path / "metrics.csv") == [{"output": text, "missing": ""}]

    def test_never_overwrites_an_existing_file(self, tmp_path):
        (tmp_path / "metrics.csv").write_text("evidence")
        with pytest.raises(FileExistsError):
            MetricsWriter(tmp_path / "metrics.csv")


class TestLogging:
    def test_run_log_captures_the_start_line(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        assert "started" in (run.path / "run.log").read_text()

    def test_resolved_yaml_is_valid_yaml(self, cfg, tmp_path):
        with RunDirectory(cfg, repo_root=tmp_path) as run:
            pass
        assert isinstance(yaml.safe_load((run.path / "config.resolved.yaml").read_text()), dict)
