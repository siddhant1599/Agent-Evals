"""Analyses of finished runs: the config group, D-115's bootstrap, and the diagnostics run.

CPU only, no network: source runs are small directories written here, and the sentence splitter is
a stub, so NLTK's data is not needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from agent_evals.agents.self_feedback import is_stop_reply
from agent_evals.analysis import (
    AnalysisError,
    PairedBootstrap,
    TopologyDiagnostics,
    read_source_run,
    reply_kind,
    uncited_share,
)
from agent_evals.config import (
    AnalysisConfig,
    BootstrapConfig,
    ConfigError,
    ExperimentConfig,
    RunConfig,
    load_config,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "analysis" / "asqa_diagnostics.yaml"
METRICS = ("minicheck", "str_em")


def _split(text):
    """Stub sentence splitter: one sentence per full stop."""
    return [s.strip() + "." for s in text.split(".") if s.strip()]


def _write_run(results, run_id, topology, seed, questions, *, status="success", dirty=False):
    """A finished run. ``questions``: id -> (scores, [(role, content, tokens, finish_reason)])."""
    path = results / run_id
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps({"run_id": run_id, "status": status, "seed": seed}), encoding="utf-8"
    )
    cell = {"topology": topology, "backbone": "qwen2.5-7b-instruct", "task": "tier2_asqa"}
    manifest = {"cell": cell, "git": {"dirty": dirty}}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "output", "example_id", *METRICS])
        for qid, (scores, calls) in questions.items():
            writer.writerow([run_id, f"{calls[-1][1]}\nsecond line", qid, *scores])
    with (path / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for qid, (_, calls) in questions.items():
            for index, (role, content, tokens, finish) in reversed(list(enumerate(calls))):
                record = {
                    "run_id": run_id,
                    "example_id": qid,
                    "message_index": index,
                    "role": role,
                    "round_index": index // 2,
                    "content": content,
                    "completion_tokens": tokens,
                    "finish_reason": finish,
                }
                handle.write(json.dumps(record) + "\n")
    return path


def _answer(text, tokens, finish="stop"):
    return ("generator", text, tokens, finish)


def _feedback(text, finish="stop"):
    return ("self_feedback", text, 5, finish)


def _baseline_questions(minicheck):
    return {q: ((minicheck, 0.4), [_answer("A [1].", 10)]) for q in ("q1", "q2", "q3")}


def _loop_questions():
    """q1 is rewritten after bullets; q2 stops at once; q3 is rewritten after 'NONE' + text."""
    return {
        "q1": (
            (0.8, 0.6),
            [
                _answer("A [1]. B [2].", 12),
                _feedback('- "B": no search result supports it'),
                _answer("A [1]. Sources differ. C [2].", 20, "length"),
                _feedback("NONE"),
            ],
        ),
        "q2": ((0.6, 0.4), [_answer("A [1].", 10), _feedback("NONE.")]),
        "q3": (
            (0.6, 0.5),
            [
                _answer("A [1].", 10),
                _feedback("NONE. The search results do not say."),
                _answer("A [1]. B.", 15),
                _feedback("none"),
            ],
        ),
    }


def _sources(results):
    """T0 seeds 0-1 (MiniCheck 0.5, 0.7) and T1 seeds 0-1 on the same three questions."""
    ids = []
    for seed, minicheck in ((0, 0.5), (1, 0.7)):
        run_id = f"20261101_00000{seed}_t0_single_agent_tier2_asqa_qwen2.5-7b-instruct_seed{seed}"
        _write_run(results, run_id, "t0_single_agent", seed, _baseline_questions(minicheck))
        ids.append(run_id)
    for seed in (0, 1):
        run_id = f"20261101_10000{seed}_t1_self_refine_tier2_asqa_qwen2.5-7b-instruct_seed{seed}"
        _write_run(results, run_id, "t1_self_refine", seed, _loop_questions())
        ids.append(run_id)
    return ids


def _config(runs, **analysis):
    fields = {"task": "tier2_asqa", "runs": tuple(runs), "metrics": METRICS}
    fields.update(analysis)
    return ExperimentConfig(
        run=RunConfig(owner="sameer"),
        name="test_analysis",
        analysis=AnalysisConfig(**fields),
    )


class TestConfig:
    def test_the_asqa_config_loads_with_d115_settings(self):
        cfg = load_config(CONFIG, ["run.owner=sameer"])
        assert cfg.analysis.bootstrap == BootstrapConfig(
            n_resamples=10_000, seed=123, ci_level=0.95
        )
        assert cfg.analysis.baseline == "t0_single_agent"
        assert len(cfg.analysis.runs) == 27
        assert cfg.analysis.length_metrics == ("str_em",)

    def test_defaults_are_d115(self):
        assert BootstrapConfig() == BootstrapConfig(n_resamples=10_000, seed=123, ci_level=0.95)

    def test_unknown_key_raises(self):
        with pytest.raises(ConfigError, match="unknown config key"):
            load_config(CONFIG, ["run.owner=sameer", "analysis.bootstrap.resamples=5"])

    @pytest.mark.parametrize(
        "fields, match",
        [
            ({"runs": ()}, "analysis.runs"),
            ({"runs": ("a", "a")}, "repeat"),
            ({"metrics": ()}, "analysis.metrics"),
            ({"length_metrics": ("rouge_l_f1",)}, "length_metrics"),
            ({"task": ""}, "analysis.task"),
        ],
    )
    def test_invalid_analysis_raises(self, fields, match):
        values = {"task": "tier2_asqa", "runs": ("a",), "metrics": ("str_em",), **fields}
        with pytest.raises(ConfigError, match=match):
            AnalysisConfig(**values)

    @pytest.mark.parametrize(
        "fields", [{"n_resamples": 0}, {"seed": -1}, {"ci_level": 1.0}, {"ci_level": 0.0}]
    )
    def test_invalid_bootstrap_raises(self, fields):
        with pytest.raises(ConfigError, match="analysis.bootstrap"):
            BootstrapConfig(**fields)


class TestBootstrap:
    def test_constant_differences_give_a_point_interval(self):
        interval = PairedBootstrap(BootstrapConfig(n_resamples=50)).mean(np.full(4, 0.25))
        assert (interval.estimate, interval.low, interval.high) == pytest.approx((0.25, 0.25, 0.25))

    def test_mean_and_interval_lie_within_the_data(self):
        interval = PairedBootstrap(BootstrapConfig(n_resamples=2000)).mean(
            np.array([1.0, 2.0, 3.0])
        )
        assert interval.estimate == pytest.approx(2.0)
        assert 1.0 <= interval.low < 2.0 < interval.high <= 3.0

    def test_the_same_seed_gives_the_same_interval(self):
        data = np.array([0.1, -0.3, 0.5, 0.0, 0.2])
        first = PairedBootstrap(BootstrapConfig(n_resamples=500)).mean(data)
        second = PairedBootstrap(BootstrapConfig(n_resamples=500)).mean(data)
        assert first == second

    def test_length_adjustment_recovers_an_exact_line(self):
        lengths = np.array([-4.0, 0.0, 3.0, 10.0])
        fit = PairedBootstrap(BootstrapConfig(n_resamples=200)).length_adjusted(
            0.5 + 0.02 * lengths, lengths
        )
        assert fit.slope == pytest.approx(0.02)
        assert fit.mean_length_change == pytest.approx(2.25)
        assert fit.observed == pytest.approx(0.545)
        assert (fit.adjusted.estimate, fit.adjusted.low, fit.adjusted.high) == pytest.approx(
            (0.5, 0.5, 0.5)
        )

    def test_resamples_with_constant_lengths_are_left_out_and_counted(self):
        # Two questions: about half of all resamples draw the same question twice.
        fit = PairedBootstrap(BootstrapConfig(n_resamples=400)).length_adjusted(
            np.array([0.0, 1.0]), np.array([0.0, 10.0])
        )
        assert 100 < fit.dropped_resamples < 300
        assert fit.adjusted.estimate == pytest.approx(0.0)

    def test_length_adjustment_needs_varying_lengths(self):
        with pytest.raises(AnalysisError, match="does not vary"):
            PairedBootstrap(BootstrapConfig(n_resamples=10)).length_adjusted(
                np.array([0.1, 0.2]), np.array([3.0, 3.0])
            )

    def test_empty_and_mismatched_inputs_raise(self):
        bootstrap = PairedBootstrap(BootstrapConfig(n_resamples=10))
        with pytest.raises(AnalysisError, match="at least one question"):
            bootstrap.mean(np.array([]))
        with pytest.raises(AnalysisError, match="same questions"):
            bootstrap.length_adjusted(np.array([0.1, 0.2]), np.array([1.0, 2.0, 3.0]))


class TestReplies:
    @pytest.mark.parametrize(
        "text, kind",
        [
            ("NONE", "stop"),
            (" none. ", "stop"),
            ('- "A": no search result supports it\n\n- "B": [2] does not', "bullets"),
            ("NONE. The search results do not say.", "stop_word_then_text"),
            ("None (as no search result discusses it)", "stop_word_then_text"),
            ("All sentences are supported. NONE.", "prose"),
            ('- "A": unsupported\nCorrected answer: A [1].', "prose"),
            ("NONEXISTENT claims", "prose"),
            ("", "prose"),
        ],
    )
    def test_reply_kind(self, text, kind):
        assert reply_kind(text) == kind

    @pytest.mark.parametrize(
        "text, stop", [("NONE", True), (" None. ", True), ("NONE. More", False), ("- x", False)]
    )
    def test_is_stop_reply_follows_d101(self, text, stop):
        assert is_stop_reply(text) is stop

    def test_uncited_share(self):
        assert uncited_share("A [1]. B. C [2][3].", _split) == pytest.approx(1 / 3)
        assert uncited_share("A [1].", _split) == 0.0
        assert uncited_share("", _split) is None


class TestSourceRun:
    def test_reads_scores_and_calls_in_message_order(self, tmp_path):
        path = _write_run(tmp_path, "r1", "t1_self_refine", 0, _loop_questions())
        run = read_source_run(path, METRICS)
        assert run.topology == "t1_self_refine" and run.seed == 0
        assert run.scores["q1"] == {"minicheck": 0.8, "str_em": 0.6}
        assert [c.role for c in run.calls["q1"]] == [
            "generator",
            "self_feedback",
            "generator",
            "self_feedback",
        ]
        assert run.calls["q1"][2].hit_cap and not run.calls["q1"][0].hit_cap
        assert len(run.metrics_sha256) == len(run.trace_sha256) == 64

    @pytest.mark.parametrize(
        "kwargs, keys, match",
        [
            ({"status": "failed"}, METRICS, "not success"),
            ({"dirty": True}, METRICS, "clean tree"),
            ({}, ("minicheck", "rouge_l_f1"), "no column"),
        ],
    )
    def test_refuses_uncitable_or_incomplete_runs(self, tmp_path, kwargs, keys, match):
        path = _write_run(tmp_path, "r1", "t1_self_refine", 0, _loop_questions(), **kwargs)
        with pytest.raises(AnalysisError, match=match):
            read_source_run(path, keys)

    def test_refuses_a_missing_directory(self, tmp_path):
        with pytest.raises(AnalysisError, match="no run directory"):
            read_source_run(tmp_path / "absent", METRICS)


class TestDiagnostics:
    def _run(self, tmp_path, **analysis):
        runs = _sources(tmp_path / "results")
        cfg = _config(runs, length_metrics=("str_em",), **analysis)
        path = TopologyDiagnostics(cfg, tmp_path, split_sentences=_split).run()
        return path, json.loads((path / "summary.json").read_text(encoding="utf-8"))

    def test_writes_every_run_file(self, tmp_path):
        path, summary = self._run(tmp_path)
        for name in ("config.resolved.yaml", "manifest.json", "run.log", "metrics.csv"):
            assert (path / name).exists()
        assert summary["status"] == "success" and summary["records_written"] == 12
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["source_runs"]) == 4

    def test_paired_gain_averages_seeds_within_questions(self, tmp_path):
        _, summary = self._run(tmp_path)
        assert summary["n_seeds"]["t0_single_agent"]["qwen2.5-7b-instruct"] == 2
        paired = summary["paired"]["t1_self_refine"]["qwen2.5-7b-instruct"]["minicheck"]
        # T0 averages 0.6 on every question; T1 scores 0.8, 0.6, 0.6.
        assert paired["baseline"] == pytest.approx(0.6)
        assert paired["topology"] == pytest.approx(2.0 / 3.0)
        assert paired["gain"] == pytest.approx(0.2 / 3.0)
        assert paired["low"] <= paired["gain"] <= paired["high"]
        assert "t0_single_agent" not in summary["paired"]

    def test_length_adjusted_change(self, tmp_path):
        _, summary = self._run(tmp_path)
        fit = summary["length_adjusted"]["t1_self_refine"]["qwen2.5-7b-instruct"]["str_em"]
        # Final answers are 10, 5 and 0 tokens longer than T0's; str_em rises 0.02 per token.
        assert fit["mean_length_change_tokens"] == pytest.approx(5.0)
        assert fit["slope_per_token"] == pytest.approx(0.02)
        assert fit["observed"] == pytest.approx(0.1)
        assert fit["adjusted"] == pytest.approx(0.0, abs=1e-12)

    def test_feedback_replies_and_first_feedback(self, tmp_path):
        _, summary = self._run(tmp_path)
        replies = summary["feedback_replies"]["t1_self_refine"]["qwen2.5-7b-instruct"]
        assert replies == {
            "n": 10,
            "stop": 6,
            "bullets": 2,
            "stop_word_then_text": 2,
            "prose": 0,
            "hit_cap": 0,
        }
        first = summary["first_feedback"]["t1_self_refine"]["qwen2.5-7b-instruct"]
        assert set(first) == {"bullets", "stop", "stop_word_then_text"}
        assert first["bullets"]["n"] == 2 and first["bullets"]["mean_answers"] == 2.0
        assert first["bullets"]["minicheck_change"] == pytest.approx(0.2)
        assert first["stop"]["minicheck_change"] == pytest.approx(0.0)

    def test_rewrites_compare_draft_and_final(self, tmp_path):
        _, summary = self._run(tmp_path)
        rewrites = summary["rewrites"]["t1_self_refine"]["qwen2.5-7b-instruct"]
        assert rewrites["n_rewritten"] == 4
        assert rewrites["draft_tokens"] == pytest.approx(11.0)
        assert rewrites["final_tokens"] == pytest.approx(17.5)
        assert rewrites["draft_uncited_share"] == pytest.approx(0.0)
        assert rewrites["final_uncited_share"] == pytest.approx((1 / 3 + 1 / 2) / 2)
        assert rewrites["draft_hit_cap"] == 0.0 and rewrites["final_hit_cap"] == 0.5
        assert "t0_single_agent" not in summary.get("rewrites", {})

    def test_refuses_a_topology_without_its_baseline(self, tmp_path):
        runs = [r for r in _sources(tmp_path / "results") if "t0_single" not in r]
        with pytest.raises(AnalysisError, match="has no t0_single_agent runs"):
            TopologyDiagnostics(_config(runs), tmp_path, split_sentences=_split).run()

    def test_refuses_another_task(self, tmp_path):
        runs = _sources(tmp_path / "results")
        cfg = _config(runs, task="tier1_xsum")
        with pytest.raises(AnalysisError, match="not the configured one"):
            TopologyDiagnostics(cfg, tmp_path, split_sentences=_split).run()

    def test_refuses_a_repeated_cell(self, tmp_path):
        runs = _sources(tmp_path / "results")
        copy = tmp_path / "results" / "copy_of_t0_seed0"
        _write_run(tmp_path / "results", copy.name, "t0_single_agent", 0, _baseline_questions(0.5))
        cfg = _config([*runs, copy.name])
        with pytest.raises(AnalysisError, match="appear twice"):
            TopologyDiagnostics(cfg, tmp_path, split_sentences=_split).run()

    def test_needs_the_analysis_group(self, tmp_path):
        cfg = ExperimentConfig(run=RunConfig(owner="sameer"))
        with pytest.raises(AnalysisError, match="analysis group"):
            TopologyDiagnostics(cfg, tmp_path)
