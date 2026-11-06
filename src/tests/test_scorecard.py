"""The agentic eval scorecard, on small synthetic results/ folders. CPU only, no network."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from agent_evals.config import BootstrapConfig
from agent_evals.scorecard import (
    ScorecardConfig,
    build_scorecard,
    canonical_runs,
    scan_results,
    write_outputs,
)
from agent_evals.scorecard.compute import _task_family

FIELDS = [
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
    "finish_reason",
    "hit_length_cap",
    "blank_output",
    "minicheck",
    "alignscore",
]
CFG = ScorecardConfig(pass_threshold=0.5, bootstrap=BootstrapConfig(n_resamples=200, seed=1))


def _eval_run(
    root: Path,
    run_id: str,
    topology: str,
    seed: int,
    scores: list[float],
    *,
    task: str = "tier1_xsum",
    backbone: str = "bb",
    status: str = "success",
    tokens: int = 100,
    calls: int = 1,
    converged: bool = True,
    latency: float = 1.0,
    adapter: str | None = None,
) -> None:
    d = root / run_id
    d.mkdir(parents=True)
    manifest = {
        "run_id": run_id,
        "started_at": run_id,
        "config_name": topology,
        "git": {"dirty": False},
        "seeding": {"seed": seed},
        "cell": {"topology": topology, "backbone": backbone, "task": task},
    }
    if adapter:
        manifest["lora_adapter"] = {"path": f"~/.cache/agent_evals/adapters/{adapter}/final"}
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "summary.json").write_text(json.dumps({"status": status}))
    with (d / "metrics.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=FIELDS)
        w.writeheader()
        if status != "success":
            return
        for i, s in enumerate(scores):
            w.writerow(
                {
                    "run_id": run_id,
                    "output": "x",
                    "prompt_tokens": tokens - 10,
                    "completion_tokens": 10,
                    "total_tokens": tokens,
                    "wall_clock_s": latency,
                    "n_rounds": calls if not converged or calls == 1 else calls - 1,
                    "converged": converged,
                    "n_messages": calls,
                    "example_id": str(i),
                    "source_words": 50,
                    "finish_reason": "stop",
                    "hit_length_cap": False,
                    "blank_output": False,
                    "minicheck": s,
                    "alignscore": s,
                }
            )


def _training_run(root: Path, run_id: str, task: str, hours: float, backbone: str = "bb") -> None:
    d = root / run_id
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "started_at": run_id,
                "task": task,
                "backbone": {"name": backbone},
                "grpo_args": {},
                "git": {"dirty": False},
                "seeding": {"seed": 0},
            }
        )
    )
    (d / "summary.json").write_text(json.dumps({"status": "success", "training_gpu_hours": hours}))
    (d / "metrics.csv").write_text("step,reward\n1,0.5\n")


@pytest.fixture
def results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    base = [0.4, 0.6, 0.6, 0.4]
    for seed in range(2):
        _eval_run(root, f"2026010{seed}_t0", "t0_single_agent", seed, base)
        # A superseded run of the loop (lower scores), then the re-run that replaces it.
        _eval_run(
            root, f"2026011{seed}_t1_old", "t1_self_refine", seed, [0.0] * 4, tokens=400, calls=4
        )
        _eval_run(
            root,
            f"2026012{seed}_t1",
            "t1_self_refine",
            seed,
            [0.6, 0.7, 0.7, 0.6],
            tokens=400,
            calls=4,
            converged=seed == 0,
            latency=4.0,
        )
        _eval_run(
            root,
            f"2026013{seed}_t5",
            "t5_rl_agent",
            seed,
            [0.5, 0.9, 0.9, 0.5],
            latency=1.0,
            adapter="20260001_grpo_xsum_bb_seed0",
        )
    _eval_run(root, "20260139_t5_failed", "t5_rl_agent", 2, [], status="failed")
    _training_run(root, "20260001_grpo_xsum_bb_seed0", "tier1_xsum", hours=2.0)
    return root


def test_scan_classifies_runs_and_keeps_failures(results: Path) -> None:
    records = scan_results(results)
    kinds = sorted({r.kind for r in records})
    assert kinds == ["agent_eval", "training"]
    assert any(r.status == "failed" for r in records)
    t5 = [r for r in records if r.topology == "t5_rl_agent" and r.succeeded]
    assert {r.adapter_run_id for r in t5} == {"20260001_grpo_xsum_bb_seed0"}


def test_canonical_runs_take_the_latest_success_per_seed(results: Path) -> None:
    canonical = canonical_runs(scan_results(results))
    t1 = [r.run_id for r in canonical if r.topology == "t1_self_refine"]
    assert t1 == ["20260120_t1", "20260121_t1"]
    assert "20260139_t5_failed" not in {r.run_id for r in canonical}


def test_outcome_reliability_trajectory_and_efficiency(results: Path) -> None:
    card = build_scorecard(scan_results(results), CFG)
    cells = {c["topology"]: c for c in card["cells"]}
    t0, t1, t5 = cells["t0_single_agent"], cells["t1_self_refine"], cells["t5_rl_agent"]

    assert t0["outcome"]["headline"] == pytest.approx(0.5)
    assert t0["outcome"]["headline_seed_sd"] == pytest.approx(0.0)
    # Two of four items reach 0.5 on every seed: 0.6, 0.6 (0.4 fails).
    assert t0["reliability"]["pass^k"] == pytest.approx(0.5)
    assert t1["reliability"]["pass^k"] == pytest.approx(1.0)

    assert t0["trajectory"]["multi_step"] is False
    assert t0["trajectory"]["self_termination_rate"] is None
    assert t1["trajectory"]["llm_calls_per_task"] == pytest.approx(4.0)
    assert t1["trajectory"]["self_termination_rate"] == pytest.approx(0.5)

    assert t5["reliability"]["runs_launched"] == 3
    assert t5["reliability"]["runs_failed"] == 1
    assert t5["reliability"]["run_completion_rate"] == pytest.approx(2 / 3)
    assert t5["efficiency"]["training_gpu_hours"] == pytest.approx(2.0)
    assert t5["efficiency"]["transfer"] is False


def test_value_over_baseline_is_paired_with_a_cost_multiplier(results: Path) -> None:
    card = build_scorecard(scan_results(results), CFG)
    t1 = next(c for c in card["cells"] if c["topology"] == "t1_self_refine")
    v = t1["value_over_baseline"]
    assert v["headline"]["gain"] == pytest.approx(0.15)
    assert v["headline"]["verdict"] == "better"
    assert v["token_multiplier"] == pytest.approx(4.0)
    assert v["gain_per_1k_extra_tokens"] == pytest.approx(0.15 / 0.3)
    t0 = next(c for c in card["cells"] if c["topology"] == "t0_single_agent")
    assert t0["value_over_baseline"] is None


def test_pareto_and_break_even(results: Path) -> None:
    card = build_scorecard(scan_results(results), CFG)
    pareto = {c["topology"]: c["pareto_optimal"] for c in card["cells"]}
    # T5 is as cheap as T0 and better, so T0 is dominated; T1 costs more than T5 for a lower score.
    assert pareto == {"t0_single_agent": False, "t1_self_refine": False, "t5_rl_agent": True}
    (row,) = card["break_even"]
    assert row["inference_loop"] == "t1_self_refine"
    assert row["break_even_requests"] == pytest.approx(2.0 * 3600 / 3.0)
    assert row["rl_matches_quality"] is True


def test_fixed_length_loops_have_no_self_termination_rate(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _eval_run(root, "1_t2", "t2_critique_revise", 0, [0.9, 0.8], calls=3, converged=False)
    (cell,) = build_scorecard(scan_results(root), CFG)["cells"]
    assert cell["trajectory"]["multi_step"] is True
    assert cell["trajectory"]["self_termination_rate"] is None


def test_tool_ablation_pairs_t2_with_its_no_tool_variant(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _eval_run(root, "1_t2", "t2_critique_revise", 0, [0.9, 0.9, 0.8], latency=9.0)
    _eval_run(root, "2_t2n", "t2_critique_revise_no_nli", 0, [0.5, 0.5, 0.4], latency=3.0)
    card = build_scorecard(scan_results(root), CFG)
    (row,) = card["tool_value"]
    assert row["gain"] == pytest.approx(0.4)
    assert row["latency_multiplier"] == pytest.approx(3.0)


def test_task_family_matches_training_and_evaluation_names() -> None:
    assert _task_family("tier2_asqa_train") == _task_family("tier2_asqa") == "asqa"
    assert _task_family("tier1_xsum") != _task_family("tier1_cnndm")


def test_write_outputs(results: Path, tmp_path: Path) -> None:
    card = build_scorecard(scan_results(results), CFG)
    paths = write_outputs(card, tmp_path / "evals", figures=False)
    assert json.loads(paths["json"].read_text())["counts"]["cells"] == 3
    rows = list(csv.DictReader(paths["csv"].open()))
    assert [r["agent"] for r in rows] == ["T0", "T1", "T5"]
    md = paths["markdown"].read_text()
    assert "# Agentic Eval Scorecard" in md and "RL break-even" in md
