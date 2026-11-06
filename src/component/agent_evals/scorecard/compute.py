"""The agentic evaluation scorecard: five dimensions per agent configuration, plus comparisons.

Every agent configuration (a *cell*: agent architecture x backbone x task, over its seeds) is
scored on five dimensions, the same ones agent-evaluation work separates an agent's final answer
from the way it got there:

1. **Outcome** -- how good the final output is: the task's headline judge (MiniCheck, D-115 and
   D-120) plus task-specific secondary metrics, mean over seeds with the seed-to-seed spread.
2. **Reliability** -- whether the agent succeeds *consistently*: ``pass@1``, ``pass@k`` and
   ``pass^k`` over seeds (an item passes when its headline score clears a threshold; ``pass^k``
   requires every seed to pass, as in tau-bench), and the share of launched runs that completed.
3. **Trajectory** -- what the agent did: LLM calls and rounds per task, how often a refinement
   loop stopped on its own before its budget, how often a call hit the completion cap, and how
   often the final output was blank.
4. **Efficiency** -- what the agent cost: tokens and wall-clock seconds per task, and (for an
   RL-trained agent) the one-time GPU-hours of the training run behind it.
5. **Value over baseline** -- whether the extra machinery paid off: the paired, per-item gain over
   the single-agent baseline on the same backbone and task, with a bootstrap interval, the cost
   multiplier it took, and the gain per thousand extra tokens.

On top of the cells it reports a per-architecture roll-up across backbones, the value of the
Critic's NLI tool (Critique-Revise with the tool against the same loop without it), the
inference-time quality/cost Pareto frontier, and the request volume at which a one-time RL
training cost breaks even with an inference-time loop.

Inputs are read with :mod:`agent_evals.scorecard.collect`; nothing is generated or re-scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..analysis.bootstrap import PairedBootstrap
from ..config import BootstrapConfig
from .collect import Example, RunRecord, canonical_runs, group_by_cell, read_examples

__all__ = [
    "AGENT_PROFILES",
    "TASKS",
    "AgentProfile",
    "ScorecardConfig",
    "TaskSpec",
    "build_scorecard",
    "profile_for",
    "task_spec",
]

BASELINE = "t0_single_agent"


@dataclass(frozen=True)
class AgentProfile:
    """What kind of agent a topology is.

    Attributes:
        code: Short label used in tables (T0, T1, ...).
        name: Human-readable name.
        pattern: The agentic design pattern it instantiates.
        roles: The agent roles that take part in one task.
        uses_tool: Whether an agent calls an external tool during generation.
        adaptation: Where its improvement is paid for: ``none``, ``inference-time`` (every
            request), or ``training-time`` (once).
    """

    code: str
    name: str
    pattern: str
    roles: tuple[str, ...]
    uses_tool: bool
    adaptation: str


AGENT_PROFILES: dict[str, AgentProfile] = {
    "t0_single_agent": AgentProfile(
        "T0", "Single agent", "single call (baseline)", ("generator",), False, "none"
    ),
    "t1_self_refine": AgentProfile(
        "T1",
        "Self-Refine",
        "reflection loop: draft -> self-feedback -> refine",
        ("generator", "self_feedback", "refiner"),
        False,
        "inference-time",
    ),
    "t2_critique_revise": AgentProfile(
        "T2",
        "Critique-Revise + NLI tool",
        "multi-agent critique with a tool-using Critic",
        ("generator", "critic"),
        True,
        "inference-time",
    ),
    "t2_critique_revise_no_nli": AgentProfile(
        "T2-noTool",
        "Critique-Revise (no tool)",
        "multi-agent critique, tool removed (ablation)",
        ("generator", "critic"),
        False,
        "inference-time",
    ),
    "t5_rl_agent": AgentProfile(
        "T5",
        "RL-trained agent (GRPO)",
        "single call by a policy trained with a judge-based reward",
        ("generator",),
        False,
        "training-time",
    ),
}


@dataclass(frozen=True)
class TaskSpec:
    """How a task is scored.

    Attributes:
        name: Display name.
        kind: What the agent is asked to do.
        headline: The metric key every comparison is made on.
        secondary: Further outcome metrics reported alongside it.
    """

    name: str
    kind: str
    headline: str
    secondary: tuple[str, ...]


_SUMMARY_SECONDARY = ("alignscore", "rouge_l_f1", "bertscore_f1")
TASKS: dict[str, TaskSpec] = {
    "tier1_xsum": TaskSpec("XSum", "one-sentence news summary", "minicheck", _SUMMARY_SECONDARY),
    "tier1_cnndm": TaskSpec(
        "CNN/DM", "multi-sentence news summary", "minicheck", _SUMMARY_SECONDARY
    ),
    "tier1_samsum": TaskSpec("SAMSum", "dialogue summary", "minicheck", _SUMMARY_SECONDARY),
    "tier2_asqa": TaskSpec(
        "ASQA",
        "grounded long-form QA with citations",
        "minicheck",
        ("alignscore", "str_em", "citation_recall", "citation_precision"),
    ),
}


def profile_for(topology: str) -> AgentProfile:
    """The topology's profile; an unknown topology gets a generic one rather than an error."""
    return AGENT_PROFILES.get(
        topology, AgentProfile(topology, topology, "unknown", (), False, "inference-time")
    )


def task_spec(task: str) -> TaskSpec:
    """The task's scoring spec; an unknown task is scored on MiniCheck with no secondaries."""
    return TASKS.get(task, TaskSpec(task, "unknown", "minicheck", ()))


@dataclass(frozen=True)
class ScorecardConfig:
    """Scorecard settings.

    Attributes:
        pass_threshold: An item *passes* when its headline score is at least this.
        bootstrap: Resamples, seed and coverage of every interval (D-115's defaults).
    """

    pass_threshold: float = 0.5
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


# --------------------------------------------------------------------------------------------
# helpers


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _sd(values: Sequence[float]) -> float | None:
    return float(np.std(values, ddof=1)) if len(values) >= 2 else (0.0 if values else None)


def _rate(flags: Sequence[bool]) -> float | None:
    return float(np.mean(flags)) if len(flags) else None


def _ratio(a: float | None, b: float | None) -> float | None:
    return a / b if a is not None and b else None


def _task_family(task: str | None) -> str:
    """The dataset behind a task name: ``tier2_asqa_train`` and ``tier2_asqa`` are both ``asqa``."""
    name = (task or "").removesuffix("_train").removesuffix("_sample")
    return name.split("_", 1)[1] if name.startswith("tier") and "_" in name else name


def _per_item_mean(seeds: Sequence[Sequence[Example]], key: str) -> dict[str, float]:
    """Item id -> score on ``key`` averaged over the seeds that scored it (D-115)."""
    acc: dict[str, list[float]] = {}
    for examples in seeds:
        for ex in examples:
            if key in ex.scores:
                acc.setdefault(ex.example_id, []).append(ex.scores[key])
    return {item: float(np.mean(v)) for item, v in acc.items()}


# --------------------------------------------------------------------------------------------
# cells


@dataclass
class CellScore:
    """The five-dimension scorecard of one agent configuration over its seeds."""

    task: str
    topology: str
    backbone: str
    agent: dict[str, Any]
    n_seeds: int
    n_items: int
    outcome: dict[str, Any]
    reliability: dict[str, Any]
    trajectory: dict[str, Any]
    efficiency: dict[str, Any]
    provenance: dict[str, Any]
    value_over_baseline: dict[str, Any] | None = None
    pareto_optimal: bool | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.task, self.topology, self.backbone)


def _score_cell(
    cell: tuple[str, str, str],
    runs: Sequence[RunRecord],
    attempts: Sequence[RunRecord],
    training: Mapping[str, RunRecord],
    cfg: ScorecardConfig,
) -> tuple[CellScore, list[list[Example]]]:
    task, topology, backbone = cell
    spec = task_spec(task)
    seeds: list[list[Example]] = []
    shas: list[str] = []
    for run in runs:
        examples, sha = read_examples(run)
        seeds.append(examples)
        shas.append(sha)

    # 1. outcome
    def seed_means(key: str) -> list[float]:
        return [
            m
            for ex in seeds
            if (m := _mean([e.scores[key] for e in ex if key in e.scores])) is not None
        ]

    headline = seed_means(spec.headline)
    outcome = {
        "headline_metric": spec.headline,
        "headline": _mean(headline),
        "headline_seed_sd": _sd(headline),
        "secondary": {key: _mean(seed_means(key)) for key in spec.secondary if seed_means(key)},
    }

    # 2. reliability (pass@k over seeds on items every seed answered)
    per_seed_pass = [
        {
            e.example_id: e.scores[spec.headline] >= cfg.pass_threshold
            for e in ex
            if spec.headline in e.scores
        }
        for ex in seeds
    ]
    common = set.intersection(*(set(p) for p in per_seed_pass)) if per_seed_pass else set()
    k = len(per_seed_pass)
    if common:
        grid = np.array([[p[i] for i in sorted(common)] for p in per_seed_pass], dtype=bool)
        pass_1, pass_k, pass_hat_k = (
            float(grid.mean()),
            float(grid.any(axis=0).mean()),
            float(grid.all(axis=0).mean()),
        )
    else:
        pass_1 = pass_k = pass_hat_k = None
    failed = [a for a in attempts if not a.succeeded]
    reliability = {
        "pass_threshold": cfg.pass_threshold,
        "k": k,
        "pass@1": pass_1,
        "pass@k": pass_k,
        "pass^k": pass_hat_k,
        "consistency_gap": (pass_k - pass_hat_k) if common else None,
        "runs_launched": len(attempts),
        "runs_failed": len(failed),
        "run_completion_rate": _ratio(len(attempts) - len(failed), len(attempts)),
        "failure_statuses": sorted({a.status for a in failed}),
    }

    # 3. trajectory
    flat = [e for ex in seeds for e in ex]
    calls = [e.trajectory["n_messages"] for e in flat if "n_messages" in e.trajectory]
    multi_step = bool(calls) and max(calls) > 1
    rounds = [e.trajectory["n_rounds"] for e in flat if "n_rounds" in e.trajectory]
    # Only a loop whose length varies can stop early. For a single call, or a loop that always
    # runs the same number of rounds (T2), the rate is undefined, not 100 %.
    adaptive = multi_step and len(set(rounds)) > 1
    trajectory = {
        "multi_step": multi_step,
        "adaptive_length": adaptive,
        "llm_calls_per_task": _mean(calls),
        "max_llm_calls": max(calls) if calls else None,
        "rounds_per_task": _mean(rounds),
        "self_termination_rate": (
            _rate([e.flags["converged"] for e in flat if "converged" in e.flags])
            if adaptive
            else None
        ),
        "length_cap_rate": _rate(
            [e.flags["hit_length_cap"] for e in flat if "hit_length_cap" in e.flags]
        ),
        "blank_output_rate": _rate(
            [e.flags["blank_output"] for e in flat if "blank_output" in e.flags]
        ),
    }

    # 4. efficiency
    adapters = sorted({r.adapter_run_id for r in runs if r.adapter_run_id})
    trained = [training[a] for a in adapters if a in training]
    efficiency = {
        "tokens_per_task": _mean(
            [e.trajectory["total_tokens"] for e in flat if "total_tokens" in e.trajectory]
        ),
        "completion_tokens_per_task": _mean(
            [e.trajectory["completion_tokens"] for e in flat if "completion_tokens" in e.trajectory]
        ),
        "latency_s_per_task": _mean(
            [e.trajectory["wall_clock_s"] for e in flat if "wall_clock_s" in e.trajectory]
        ),
        "training_gpu_hours": (
            sum(float(t.summary.get("training_gpu_hours") or 0.0) for t in trained)
            if trained
            else None
        ),
        "trained_on": sorted({t.task for t in trained if t.task}) or None,
        "transfer": bool(trained)
        and all(_task_family(t.task) != _task_family(task) for t in trained),
    }

    score = CellScore(
        task=task,
        topology=topology,
        backbone=backbone,
        agent=asdict(profile_for(topology)),
        n_seeds=k,
        n_items=len(common),
        outcome=outcome,
        reliability=reliability,
        trajectory=trajectory,
        efficiency=efficiency,
        provenance={
            "runs": [r.run_id for r in runs],
            "metrics_sha256": shas,
            "seeds": [r.seed for r in runs],
            "all_clean_tree": all(r.clean_tree for r in runs),
            "adapters": adapters,
            "failed_attempts": [a.run_id for a in failed],
        },
    )
    return score, seeds


# --------------------------------------------------------------------------------------------
# comparisons


def _paired(
    boot: PairedBootstrap,
    treated: Sequence[Sequence[Example]],
    control: Sequence[Sequence[Example]],
    key: str,
) -> dict[str, Any] | None:
    a, b = _per_item_mean(treated, key), _per_item_mean(control, key)
    items = sorted(set(a) & set(b))
    if not items:
        return None
    diffs = np.array([a[i] - b[i] for i in items])
    interval = boot.mean(diffs)
    if interval.low > 0:
        verdict = "better"
    elif interval.high < 0:
        verdict = "worse"
    else:
        verdict = "no clear difference"
    return {
        "metric": key,
        "n_items": len(items),
        "gain": interval.estimate,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "verdict": verdict,
        "items_improved": float(np.mean(diffs > 0)),
        "items_hurt": float(np.mean(diffs < 0)),
    }


def _value_over(
    boot: PairedBootstrap,
    cell: CellScore,
    seeds: Sequence[Sequence[Example]],
    base: CellScore,
    base_seeds: Sequence[Sequence[Example]],
) -> dict[str, Any] | None:
    spec = task_spec(cell.task)
    headline = _paired(boot, seeds, base_seeds, spec.headline)
    if headline is None:
        return None
    tok, base_tok = cell.efficiency["tokens_per_task"], base.efficiency["tokens_per_task"]
    lat, base_lat = cell.efficiency["latency_s_per_task"], base.efficiency["latency_s_per_task"]
    extra_k = (tok - base_tok) / 1000.0 if tok is not None and base_tok is not None else None
    secondary = {}
    for key in spec.secondary:
        a, b = cell.outcome["secondary"].get(key), base.outcome["secondary"].get(key)
        if a is not None and b is not None:
            secondary[key] = a - b
    return {
        "baseline": base.topology,
        "headline": headline,
        "secondary_delta": secondary,
        "pass^k_delta": (
            (cell.reliability["pass^k"] - base.reliability["pass^k"])
            if cell.reliability["pass^k"] is not None and base.reliability["pass^k"] is not None
            else None
        ),
        "token_multiplier": _ratio(tok, base_tok),
        "latency_multiplier": _ratio(lat, base_lat),
        "extra_tokens_per_task": extra_k * 1000.0 if extra_k is not None else None,
        # Gain per 1k extra tokens is only defined when the configuration spends more than T0.
        "gain_per_1k_extra_tokens": (
            headline["gain"] / extra_k if extra_k and extra_k > 0.05 else None
        ),
    }


def _pareto(cells: Sequence[CellScore]) -> None:
    """Mark the cells no other cell on the same task and backbone dominates on (quality, tokens).

    Inference-time cost only: an RL-trained agent's one-time training cost is reported separately
    (``efficiency.training_gpu_hours`` and the break-even table), not folded in here.
    """
    groups: dict[tuple[str, str], list[CellScore]] = {}
    for c in cells:
        groups.setdefault((c.task, c.backbone), []).append(c)
    for group in groups.values():
        for c in group:
            q, t = c.outcome["headline"], c.efficiency["tokens_per_task"]
            if q is None or t is None:
                c.pareto_optimal = None
                continue
            c.pareto_optimal = not any(
                o is not c
                and o.outcome["headline"] is not None
                and o.efficiency["tokens_per_task"] is not None
                and o.outcome["headline"] >= q
                and o.efficiency["tokens_per_task"] <= t
                and (o.outcome["headline"] > q or o.efficiency["tokens_per_task"] < t)
                for o in group
            )


def _rollup(cells: Sequence[CellScore]) -> list[dict[str, Any]]:
    """Per (task, architecture): how it fares across backbones."""
    groups: dict[tuple[str, str], list[CellScore]] = {}
    for c in cells:
        groups.setdefault((c.task, c.topology), []).append(c)
    rows = []
    for (task, topology), group in sorted(groups.items()):
        vob = [c.value_over_baseline for c in group if c.value_over_baseline]
        verdicts = [v["headline"]["verdict"] for v in vob]

        def avg(get, group=group):
            vals = [x for c in group if (x := get(c)) is not None]
            return _mean(vals)

        rows.append(
            {
                "task": task,
                "topology": topology,
                "code": profile_for(topology).code,
                "name": profile_for(topology).name,
                "backbones": len(group),
                "headline": avg(lambda c: c.outcome["headline"]),
                "pass^k": avg(lambda c: c.reliability["pass^k"]),
                "run_completion_rate": avg(lambda c: c.reliability["run_completion_rate"]),
                "llm_calls_per_task": avg(lambda c: c.trajectory["llm_calls_per_task"]),
                "tokens_per_task": avg(lambda c: c.efficiency["tokens_per_task"]),
                "latency_s_per_task": avg(lambda c: c.efficiency["latency_s_per_task"]),
                "mean_gain": _mean([v["headline"]["gain"] for v in vob]) if vob else None,
                "mean_token_multiplier": (
                    _mean([v["token_multiplier"] for v in vob if v["token_multiplier"]])
                    if vob
                    else None
                ),
                "backbones_better": verdicts.count("better"),
                "backbones_no_difference": verdicts.count("no clear difference"),
                "backbones_worse": verdicts.count("worse"),
            }
        )
    return rows


def _tool_value(
    boot: PairedBootstrap, by_key: Mapping[tuple[str, str, str], tuple[CellScore, list]]
) -> list[dict[str, Any]]:
    """Critique-Revise with the Critic's NLI tool against the same loop without it."""
    rows = []
    for (task, topology, backbone), (cell, seeds) in sorted(by_key.items()):
        if topology != "t2_critique_revise":
            continue
        other = by_key.get((task, "t2_critique_revise_no_nli", backbone))
        if other is None:
            continue
        paired = _paired(boot, seeds, other[1], task_spec(task).headline)
        if paired is None:
            continue
        rows.append(
            {
                "task": task,
                "backbone": backbone,
                **paired,
                "latency_multiplier": _ratio(
                    cell.efficiency["latency_s_per_task"], other[0].efficiency["latency_s_per_task"]
                ),
                "token_multiplier": _ratio(
                    cell.efficiency["tokens_per_task"], other[0].efficiency["tokens_per_task"]
                ),
            }
        )
    return rows


def _break_even(cells: Sequence[CellScore]) -> list[dict[str, Any]]:
    """Requests after which a one-time RL training cost is cheaper than an inference-time loop.

    Assumes one GPU processes one request at a time, so a request's wall clock is its GPU time
    (the benchmark runs sequentially, D-100). ``rl_matches_quality`` says whether the RL agent's
    headline is at least the loop's, i.e. whether the comparison is like for like.
    """
    rows = []
    for c in cells:
        hours = c.efficiency["training_gpu_hours"]
        if c.agent["adaptation"] != "training-time" or not hours:
            continue
        for o in cells:
            if (o.task, o.backbone) != (c.task, c.backbone) or o.agent[
                "adaptation"
            ] != "inference-time":
                continue
            lat_loop, lat_rl = (
                o.efficiency["latency_s_per_task"],
                c.efficiency["latency_s_per_task"],
            )
            if lat_loop is None or lat_rl is None:
                continue
            saving = lat_loop - lat_rl
            rows.append(
                {
                    "task": c.task,
                    "backbone": c.backbone,
                    "rl_agent": c.topology,
                    "inference_loop": o.topology,
                    "training_gpu_hours": hours,
                    "transfer": c.efficiency["transfer"],
                    "loop_extra_s_per_request": saving,
                    "break_even_requests": (hours * 3600.0 / saving) if saving > 0 else None,
                    "rl_headline": c.outcome["headline"],
                    "loop_headline": o.outcome["headline"],
                    "rl_matches_quality": (c.outcome["headline"] or 0)
                    >= (o.outcome["headline"] or 0),
                }
            )
    return rows


# --------------------------------------------------------------------------------------------
# entry point


def build_scorecard(
    records: Sequence[RunRecord], cfg: ScorecardConfig | None = None
) -> dict[str, Any]:
    """The full scorecard from every recorded attempt.

    Args:
        records: Every run from :func:`agent_evals.scorecard.collect.scan_results`.
        cfg: Settings; defaults to :class:`ScorecardConfig`.

    Returns:
        A JSON-serialisable dict with ``cells``, ``architectures``, ``tool_value``,
        ``break_even`` and ``settings``.
    """
    cfg = cfg or ScorecardConfig()
    boot = PairedBootstrap(cfg.bootstrap)
    canonical = group_by_cell(canonical_runs(records))
    attempts = group_by_cell([r for r in records if r.kind == "agent_eval"])
    training = {r.run_id: r for r in records if r.kind == "training"}

    by_key: dict[tuple[str, str, str], tuple[CellScore, list[list[Example]]]] = {}
    for cell, runs in sorted(canonical.items()):
        by_key[cell] = _score_cell(cell, runs, attempts.get(cell, []), training, cfg)

    for (task, topology, backbone), (cell, seeds) in by_key.items():
        base = by_key.get((task, BASELINE, backbone))
        if base is not None and topology != BASELINE:
            cell.value_over_baseline = _value_over(boot, cell, seeds, base[0], base[1])

    cells = [c for c, _ in by_key.values()]
    _pareto(cells)
    return {
        "settings": {
            "baseline": BASELINE,
            "pass_threshold": cfg.pass_threshold,
            "bootstrap": asdict(cfg.bootstrap),
            "headline_metrics": {t: task_spec(t).headline for t in sorted({c.task for c in cells})},
        },
        "agents": {k: asdict(v) for k, v in AGENT_PROFILES.items()},
        "tasks": {k: asdict(v) for k, v in TASKS.items()},
        "cells": [asdict(c) for c in cells],
        "architectures": _rollup(cells),
        "tool_value": _tool_value(boot, by_key),
        "break_even": _break_even(cells),
        "counts": {
            "agent_eval_attempts": sum(len(v) for v in attempts.values()),
            "agent_eval_scored_runs": sum(len(v) for v in canonical.values()),
            "training_runs": len(training),
            "cells": len(cells),
        },
    }
