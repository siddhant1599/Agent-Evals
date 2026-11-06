"""The ASQA reward-signal check (D-136): reading T0 runs, judge agreement and tie counts.

CPU only, no network: the citation judge is a stub and the dataset adapter is replaced by one that
returns fixed examples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from agent_evals.config import load_config
from agent_evals.datasets.asqa import format_passages
from agent_evals.metrics import citation as citation_module
from agent_evals.rl import reward_check as check_module
from agent_evals.rl.reward_check import (
    RewardCheck,
    RewardCheckError,
    agreement,
    pearson,
    read_source_run,
    spearman,
    summarize,
    tie_rate,
)
from agent_evals.types import Example

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rl" / "asqa_reward_check.yaml"
COLUMNS = (
    "run_id",
    "output",
    "example_id",
    "blank_output",
    "str_em",
    "citation_recall",
    "citation_precision",
)


def _run_dir(tmp_path, name, outputs, backbone="qwen2.5-7b-instruct", seed=0, status="success"):
    """A T0 ASQA run directory: (example_id, output, blank, str_em, recall, precision) rows."""
    path = tmp_path / name
    path.mkdir()
    (path / "summary.json").write_text(
        json.dumps({"run_id": name, "status": status, "seed": seed}), encoding="utf-8"
    )
    cell = {"topology": "t0_single_agent", "backbone": backbone, "task": "tier2_asqa"}
    (path / "manifest.json").write_text(json.dumps({"cell": cell}), encoding="utf-8")
    with (path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for example_id, output, blank, em, recall, precision in outputs:
            writer.writerow([name, output, example_id, blank, em, recall, precision])
    return path


def _example(example_id):
    evidence = format_passages([{"title": "T", "text": "P1."}, {"title": "U", "text": "P2."}])
    return Example(
        id=example_id, tier=2, task="grounded_qa", source=evidence, evidence=evidence, reference="r"
    )


class FixedAdapter:
    def __init__(self, ids):
        self._rows = [_example(i) for i in ids]

    def build(self):
        return self._rows

    def as_example(self, row):
        return row


class YesJudge:
    """Support 1 for claims containing 'yes', else 0."""

    def __init__(self, cfg):
        pass

    def claim_support(self, document, claim):
        return 1.0 if "yes" in claim else 0.0


class TestReadSourceRun:
    def test_reads_non_blank_outputs_and_hashes_the_file(self, tmp_path):
        path = _run_dir(
            tmp_path,
            "r0",
            [("a", "It is yes [1].", "False", 1.0, 1.0, 1.0), ("b", "", "True", "", "", "")],
        )
        run = read_source_run(path)
        assert (run.run_id, run.backbone, run.seed) == ("r0", "qwen2.5-7b-instruct", 0)
        assert [o.example_id for o in run.outputs] == ["a"]
        assert len(run.metrics_sha256) == 64

    @pytest.mark.parametrize(
        ("status", "task", "match"),
        [("failed", "tier2_asqa", "not success"), ("success", "tier1_xsum", "not a t0")],
    )
    def test_refuses_failed_or_other_runs(self, tmp_path, status, task, match):
        path = _run_dir(tmp_path, "r0", [("a", "x [1].", "False", 0, 0, 0)], status=status)
        manifest = json.loads((path / "manifest.json").read_text())
        manifest["cell"]["task"] = task
        (path / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(RewardCheckError, match=match):
            read_source_run(path)


class TestStatistics:
    def test_pearson_and_spearman(self):
        assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
        assert spearman([1, 2, 3], [1, 4, 9]) == pytest.approx(1.0)
        assert pearson([1, 1], [1, 2]) is None

    def test_spearman_averages_tied_ranks(self):
        assert spearman([0, 0, 1], [0, 0, 1]) == pytest.approx(1.0)
        assert check_module._ranks([5, 1, 5]) == [2.5, 1.0, 2.5]

    def test_agreement(self):
        result = agreement([0.0, 1.0, 0.5], [0.0, 1.0, 1.0])
        assert result["exact_agreement"] == pytest.approx(2 / 3)
        assert result["mean_difference"] == pytest.approx(0.5 / 3)

    def test_tie_rate_leaves_out_single_outputs(self):
        result = tie_rate({"a": [1.0, 1.0, 1.0], "b": [0.0, 1.0], "c": [1.0]})
        assert result == {"questions": 2, "tied": 0.5}


class TestRewardCheck:
    def _check(self, monkeypatch, tmp_path, ids):
        monkeypatch.setattr(citation_module, "AlignScore", YesJudge)
        monkeypatch.setattr(check_module.DATASETS, "build", lambda name, cfg: FixedAdapter(ids))
        cfg = load_config(CONFIG, ["run.owner=sameer", f"run.output_root={tmp_path / 'out'}"])
        return RewardCheck(cfg, tmp_path)

    def test_the_config_sets_the_rewards_judge(self):
        cfg = load_config(CONFIG, ["run.owner=sameer"])
        assert cfg.metrics.citation.judge == "alignscore"
        assert cfg.data.name == "tier2_asqa"

    def test_rescores_with_alignscore_and_keeps_the_minicheck_scores(self, monkeypatch, tmp_path):
        runs = [
            _run_dir(
                tmp_path,
                f"r{seed}",
                [
                    ("a", "It is yes [1].", "False", 1.0, 0.0, 0.0),
                    ("b", "No [2].", "False", 0, 0, 0),
                ],
                seed=seed,
            )
            for seed in (0, 1)
        ]
        out = self._check(monkeypatch, tmp_path, ["a", "b"]).run(runs)
        rows = list(csv.DictReader((out / "metrics.csv").open()))
        assert [r["citation_recall_alignscore"] for r in rows] == ["1.0", "0.0"] * 2
        assert [r["citation_recall_minicheck"] for r in rows] == ["0.0", "0.0"] * 2
        summary = json.loads((out / "summary.json").read_text())
        block = summary["by_backbone"]["qwen2.5-7b-instruct"]
        assert block["citation_recall"]["exact_agreement"] == 0.5
        assert block["ties_str_em"] == {"questions": 2, "tied": 1.0}
        manifest = json.loads((out / "manifest.json").read_text())
        assert [s["run_id"] for s in manifest["source_runs"]] == ["r0", "r1"]

    def test_refuses_a_question_outside_the_sample(self, monkeypatch, tmp_path):
        run = _run_dir(tmp_path, "r0", [("zzz", "x [1].", "False", 0, 0, 0)])
        with pytest.raises(RewardCheckError, match="outside the configured sample"):
            self._check(monkeypatch, tmp_path, ["a"]).run([run])

    def test_refuses_a_repeated_backbone_and_seed(self, monkeypatch, tmp_path):
        runs = [_run_dir(tmp_path, n, [("a", "x [1].", "False", 0, 0, 0)]) for n in ("r0", "r1")]
        with pytest.raises(RewardCheckError, match="appear twice"):
            self._check(monkeypatch, tmp_path, ["a"]).run(runs)

    def test_refuses_the_minicheck_judge(self, tmp_path):
        cfg = load_config(CONFIG, ["run.owner=sameer", "metrics.citation.judge=minicheck"])
        with pytest.raises(RewardCheckError, match="alignscore"):
            RewardCheck(cfg, tmp_path)


def test_summarize_reports_every_backbone_and_all():
    row = {
        "backbone": "b",
        "example_id": "a",
        "str_em": 1.0,
        "citation_recall_minicheck": 1.0,
        "citation_precision_minicheck": 1.0,
        "citation_recall_alignscore": 1.0,
        "citation_precision_alignscore": 0.0,
    }
    result = summarize([row, {**row, "backbone": "c"}])
    assert sorted(result["by_backbone"]) == ["b", "c"]
    assert result["all"]["citation_precision"]["mean_difference"] == -1.0
