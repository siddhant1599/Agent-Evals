"""Write the scorecard: ``scorecard.json``, ``leaderboard.csv``, ``SCORECARD.md`` and SVG figures.

    python3 src/run_evals.py                      # reads results/, writes evals/
    python3 src/run_evals.py --no-figures         # skip matplotlib

Figures need matplotlib, which is a reporting dependency only (the benchmark does not import it);
without it the tables are still written and the figures are skipped with a warning.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import BootstrapConfig
from ..logging_utils import get_logger
from .collect import scan_results
from .compute import AGENT_PROFILES, ScorecardConfig, build_scorecard, profile_for, task_spec

__all__ = ["main", "write_outputs"]

logger = get_logger(__name__)

#: Fixed categorical slot per architecture (validated reference palette, light mode), so an
#: architecture keeps its colour in every figure regardless of which others are present.
COLORS = {
    "t0_single_agent": "#2a78d6",
    "t1_self_refine": "#eb6834",
    "t2_critique_revise": "#1baf7a",
    "t2_critique_revise_no_nli": "#eda100",
    "t5_rl_agent": "#e87ba4",
}
#: Secondary encoding, so identity never rests on colour alone.
MARKERS = {
    "t0_single_agent": "o",
    "t1_self_refine": "s",
    "t2_critique_revise": "^",
    "t2_critique_revise_no_nli": "v",
    "t5_rl_agent": "D",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOPOLOGY_ORDER = list(AGENT_PROFILES)
TASK_ORDER = ["tier1_xsum", "tier1_cnndm", "tier1_samsum", "tier2_asqa"]

LEADERBOARD_FIELDS = [
    "task",
    "agent",
    "topology",
    "backbone",
    "seeds",
    "items",
    "headline_metric",
    "headline",
    "headline_seed_sd",
    "pass@1",
    "pass@k",
    "pass^k",
    "run_completion_rate",
    "llm_calls_per_task",
    "rounds_per_task",
    "self_termination_rate",
    "length_cap_rate",
    "tokens_per_task",
    "latency_s_per_task",
    "training_gpu_hours",
    "transfer",
    "gain_vs_t0",
    "gain_ci_low",
    "gain_ci_high",
    "verdict_vs_t0",
    "token_multiplier",
    "gain_per_1k_extra_tokens",
    "pareto_optimal",
    "all_clean_tree",
]


def _order(cell: dict[str, Any]) -> tuple:
    t = cell["task"]
    return (
        TASK_ORDER.index(t) if t in TASK_ORDER else 99,
        t,
        TOPOLOGY_ORDER.index(cell["topology"]) if cell["topology"] in TOPOLOGY_ORDER else 99,
        cell["backbone"],
    )


def _row(cell: dict[str, Any]) -> dict[str, Any]:
    vob = cell.get("value_over_baseline") or {}
    head = vob.get("headline") or {}
    return {
        "task": cell["task"],
        "agent": cell["agent"]["code"],
        "topology": cell["topology"],
        "backbone": cell["backbone"],
        "seeds": cell["n_seeds"],
        "items": cell["n_items"],
        "headline_metric": cell["outcome"]["headline_metric"],
        "headline": cell["outcome"]["headline"],
        "headline_seed_sd": cell["outcome"]["headline_seed_sd"],
        "pass@1": cell["reliability"]["pass@1"],
        "pass@k": cell["reliability"]["pass@k"],
        "pass^k": cell["reliability"]["pass^k"],
        "run_completion_rate": cell["reliability"]["run_completion_rate"],
        "llm_calls_per_task": cell["trajectory"]["llm_calls_per_task"],
        "rounds_per_task": cell["trajectory"]["rounds_per_task"],
        "self_termination_rate": cell["trajectory"]["self_termination_rate"],
        "length_cap_rate": cell["trajectory"]["length_cap_rate"],
        "tokens_per_task": cell["efficiency"]["tokens_per_task"],
        "latency_s_per_task": cell["efficiency"]["latency_s_per_task"],
        "training_gpu_hours": cell["efficiency"]["training_gpu_hours"],
        "transfer": cell["efficiency"]["transfer"],
        "gain_vs_t0": head.get("gain"),
        "gain_ci_low": head.get("ci_low"),
        "gain_ci_high": head.get("ci_high"),
        "verdict_vs_t0": head.get("verdict"),
        "token_multiplier": vob.get("token_multiplier"),
        "gain_per_1k_extra_tokens": vob.get("gain_per_1k_extra_tokens"),
        "pareto_optimal": cell["pareto_optimal"],
        "all_clean_tree": cell["provenance"]["all_clean_tree"],
    }


# --------------------------------------------------------------------------------------------
# markdown


def _f(value: Any, digits: int = 3, pct: bool = False, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if pct:
        return f"{value * 100:.0f}%"
    if signed:
        return f"{value:+.{digits}f}"
    if isinstance(value, float):
        return f"{value:,.{digits}f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _verdict_counts(r: dict[str, Any]) -> str:
    return f"{r['backbones_better']} / {r['backbones_no_difference']} / {r['backbones_worse']}"


def _markdown(card: dict[str, Any], figures: Sequence[str]) -> str:
    s = card["settings"]
    cells = sorted(card["cells"], key=_order)
    tasks = sorted(
        {c["task"] for c in cells}, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99
    )
    out = [
        "# Agentic Eval Scorecard",
        "",
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `src/run_evals.py` from "
        f"{card['counts']['agent_eval_scored_runs']} scored agent runs "
        f"({card['counts']['agent_eval_attempts']} launched) and "
        f"{card['counts']['training_runs']} training runs in `results/`. Do not edit by hand._",
        "",
        "Every agent configuration (architecture × backbone × task, over seeds) is scored on "
        "five dimensions: **outcome**, **reliability**, **trajectory**, **efficiency**, and "
        "**value over "
        f"the single-agent baseline** (`{s['baseline']}`). The headline judge is MiniCheck "
        "(faithfulness of the output to its grounding evidence). An item *passes* when its "
        f"headline score is ≥ {s['pass_threshold']}; `pass^k` requires all k seeds to pass. "
        f"Intervals are {s['bootstrap']['ci_level']:.0%} paired bootstrap over items "
        f"({s['bootstrap']['n_resamples']:,} resamples, seed {s['bootstrap']['seed']}).",
        "",
        "## 1. Architecture roll-up (across backbones)",
        "",
    ]
    for task in tasks:
        spec = task_spec(task)
        rows = [r for r in card["architectures"] if r["task"] == task]
        rows.sort(
            key=lambda r: (
                TOPOLOGY_ORDER.index(r["topology"]) if r["topology"] in TOPOLOGY_ORDER else 99
            )
        )
        out += [
            f"### {spec.name} — {spec.kind}",
            "",
            _table(
                [
                    "Agent",
                    "Architecture",
                    "Backbones",
                    "MiniCheck",
                    "pass^k",
                    "Calls/task",
                    "Tokens/task",
                    "Latency (s)",
                    "Gain vs T0",
                    "Cost ×",
                    "Better / same / worse",
                ],
                [
                    [
                        r["code"],
                        r["name"],
                        str(r["backbones"]),
                        _f(r["headline"]),
                        _f(r["pass^k"], pct=True),
                        _f(r["llm_calls_per_task"], 1),
                        _f(r["tokens_per_task"], 0),
                        _f(r["latency_s_per_task"], 2),
                        _f(r["mean_gain"], signed=True),
                        _f(r["mean_token_multiplier"], 1)
                        + ("×" if r["mean_token_multiplier"] else ""),
                        ("–" if r["topology"] == s["baseline"] else _verdict_counts(r)),
                    ]
                    for r in rows
                ],
            ),
            "",
        ]

    out += ["## 2. Value over the single-agent baseline (per backbone)", ""]
    rows = []
    for c in cells:
        v = c.get("value_over_baseline")
        if not v:
            continue
        h = v["headline"]
        rows.append(
            [
                task_spec(c["task"]).name,
                c["agent"]["code"],
                c["backbone"],
                f"{h['gain']:+.3f} [{h['ci_low']:+.3f}, {h['ci_high']:+.3f}]",
                h["verdict"],
                _f(h["items_improved"], pct=True) + " / " + _f(h["items_hurt"], pct=True),
                _f(v["pass^k_delta"], signed=True),
                _f(v["token_multiplier"], 1) + "×" if v["token_multiplier"] else "–",
                (
                    _f(v["gain_per_1k_extra_tokens"], 4, signed=True)
                    if v["gain_per_1k_extra_tokens"] is not None
                    else "–"
                ),
            ]
        )
    out += [
        _table(
            [
                "Task",
                "Agent",
                "Backbone",
                "MiniCheck gain [95% CI]",
                "Verdict",
                "Items up / down",
                "Δ pass^k",
                "Token ×",
                "Gain per 1k extra tokens",
            ],
            rows,
        ),
        "",
    ]

    out += ["## 3. Trajectory and reliability (per configuration)", ""]
    out += [
        _table(
            [
                "Task",
                "Agent",
                "Backbone",
                "Seeds",
                "MiniCheck ± seed SD",
                "pass@1",
                "pass@k",
                "pass^k",
                "Calls/task",
                "Self-stop",
                "Len-cap",
                "Runs completed",
                "Pareto",
            ],
            [
                [
                    task_spec(c["task"]).name,
                    c["agent"]["code"],
                    c["backbone"],
                    str(c["n_seeds"]),
                    f"{_f(c['outcome']['headline'])} ± {_f(c['outcome']['headline_seed_sd'])}",
                    _f(c["reliability"]["pass@1"], pct=True),
                    _f(c["reliability"]["pass@k"], pct=True),
                    _f(c["reliability"]["pass^k"], pct=True),
                    _f(c["trajectory"]["llm_calls_per_task"], 2),
                    _f(c["trajectory"]["self_termination_rate"], pct=True),
                    _f(c["trajectory"]["length_cap_rate"], pct=True),
                    f"{c['reliability']['runs_launched'] - c['reliability']['runs_failed']}"
                    f"/{c['reliability']['runs_launched']}",
                    "★" if c["pareto_optimal"] else "",
                ]
                for c in cells
            ],
        ),
        "",
        "_Self-stop_: share of tasks where a refinement loop ended on its own stop signal "
        "rather than its round budget (undefined for single calls and fixed-length loops). "
        "_Len-cap_: share of "
        "tasks where a call hit the completion cap. _Pareto_ ★: no other configuration on the "
        "same task and backbone is at least as faithful for fewer tokens (inference cost only).",
        "",
    ]

    if card["tool_value"]:
        out += [
            "## 4. Tool-use ablation: does the Critic's NLI tool help?",
            "",
            "Critique-Revise with the NLI fact-checking tool against the identical loop with the "
            "tool removed.",
            "",
            _table(
                [
                    "Task",
                    "Backbone",
                    "MiniCheck gain from tool [95% CI]",
                    "Verdict",
                    "Latency ×",
                    "Token ×",
                ],
                [
                    [
                        task_spec(r["task"]).name,
                        r["backbone"],
                        f"{r['gain']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]",
                        r["verdict"],
                        _f(r["latency_multiplier"], 2) + "×",
                        _f(r["token_multiplier"], 2) + "×",
                    ]
                    for r in card["tool_value"]
                ],
            ),
            "",
        ]

    if card["break_even"]:
        out += [
            "## 5. Train once or orchestrate every time? (RL break-even)",
            "",
            "Requests after which the one-time GPU cost of training the RL agent is repaid by not "
            "running an inference-time loop (wall clock per request taken as GPU time; sequential "
            "serving). _Transfer_: the adapter was trained on a different task.",
            "",
            _table(
                [
                    "Task",
                    "Backbone",
                    "vs loop",
                    "Training GPU-h",
                    "Transfer",
                    "Loop extra s/request",
                    "Break-even requests",
                    "RL ≥ loop quality?",
                ],
                [
                    [
                        task_spec(r["task"]).name,
                        r["backbone"],
                        profile_for(r["inference_loop"]).code,
                        _f(r["training_gpu_hours"], 1),
                        _f(r["transfer"]),
                        _f(r["loop_extra_s_per_request"], 2),
                        _f(r["break_even_requests"], 0) if r["break_even_requests"] else "never",
                        _f(r["rl_matches_quality"]),
                    ]
                    for r in sorted(
                        card["break_even"],
                        key=lambda r: (r["task"], r["backbone"], r["inference_loop"]),
                    )
                ],
            ),
            "",
        ]

    if figures:
        out += (
            ["## Figures", ""] + [f"![{Path(f).stem}](fig/{Path(f).name})" for f in figures] + [""]
        )

    out += [
        "## Provenance",
        "",
        "Each configuration is scored from the latest successful run per seed; superseded and "
        "failed attempts are counted under reliability but never scored. Run ids and the SHA-256 "
        "of every "
        "`metrics.csv` read are in `scorecard.json` under `cells[].provenance`.",
        "",
    ]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# figures


def _label(c: dict[str, Any]) -> str:
    return f"{task_spec(c['task']).name} · {c['agent']['code']} · {c['backbone']}"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def _figures(card: dict[str, Any], fig_dir: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; figures skipped")
        return []
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "svg.fonttype": "none",
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
        }
    )
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    cells = sorted(card["cells"], key=_order)
    tasks = sorted(
        {c["task"] for c in cells}, key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99
    )

    # 1. forest plot: paired gain over T0 with 95% CI
    rows = [c for c in cells if c.get("value_over_baseline")]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.4))
        ys = list(range(len(rows)))[::-1]
        for y, c in zip(ys, rows, strict=True):
            h = c["value_over_baseline"]["headline"]
            color = COLORS.get(c["topology"], INK_2)
            ax.plot(
                [h["ci_low"], h["ci_high"]],
                [y, y],
                color=color,
                linewidth=2,
                solid_capstyle="round",
            )
            ax.plot(
                h["gain"],
                y,
                marker=MARKERS.get(c["topology"], "o"),
                color=color,
                markersize=7,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1.5,
            )
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_yticks(
            ys,
            [_label(c) for c in rows],
            fontsize=8,
        )
        ax.set_xlabel("MiniCheck gain over single agent (T0), paired per item, 95% CI")
        _style(ax)
        ax.set_title(
            "Value over baseline: which agent architectures actually improve faithfulness?",
            loc="left",
            fontsize=11,
            color=INK,
        )
        fig.tight_layout()
        path = fig_dir / "gain_over_baseline.svg"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    # 2. Pareto: faithfulness vs tokens per task, one panel per task that compares at least three
    # architectures (with only T0 and T5 both sit at the same token cost and the panel says nothing)
    from matplotlib.ticker import FuncFormatter, NullFormatter

    multi = [t for t in tasks if len({c["topology"] for c in cells if c["task"] == t}) >= 3]
    if multi:
        fig, axes = plt.subplots(1, len(multi), figsize=(5.2 * len(multi), 4.4), squeeze=False)
        seen: dict[str, Any] = {}
        for ax, task in zip(axes[0], multi, strict=True):
            for c in (c for c in cells if c["task"] == task):
                x, y = c["efficiency"]["tokens_per_task"], c["outcome"]["headline"]
                if x is None or y is None:
                    continue
                h = ax.scatter(
                    x,
                    y,
                    s=90 if c["pareto_optimal"] else 48,
                    marker=MARKERS.get(c["topology"], "o"),
                    color=COLORS.get(c["topology"], INK_2),
                    edgecolors="#fcfcfb",
                    linewidths=1.5,
                    zorder=3,
                )
                seen.setdefault(c["topology"], h)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(task_spec(task).name, loc="left", fontsize=10, color=INK)
            ax.set_xlabel("tokens per task (log)")
            _style(ax)
        axes[0][0].set_ylabel("MiniCheck (higher = more faithful)")
        fig.legend(
            [seen[t] for t in TOPOLOGY_ORDER if t in seen],
            [f"{profile_for(t).code} {profile_for(t).name}" for t in TOPOLOGY_ORDER if t in seen],
            loc="lower center",
            ncol=min(3, len(seen)),
            frameon=False,
            fontsize=8,
        )
        fig.suptitle(
            "Quality vs. inference cost (larger marker = Pareto-optimal for its backbone)",
            x=0.01,
            ha="left",
            fontsize=11,
            color=INK,
        )
        fig.tight_layout(rect=(0, 0.12, 1, 0.94))
        path = fig_dir / "pareto_quality_vs_cost.svg"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))

    # 3. reliability: pass@k (any seed) vs pass^k (every seed) dumbbells
    rows = [c for c in cells if c["reliability"]["pass^k"] is not None and c["n_seeds"] >= 2]
    if rows:
        fig, ax = plt.subplots(figsize=(8, 0.32 * len(rows) + 1.4))
        ys = list(range(len(rows)))[::-1]
        for y, c in zip(ys, rows, strict=True):
            r = c["reliability"]
            color = COLORS.get(c["topology"], INK_2)
            ax.plot(
                [r["pass^k"], r["pass@k"]], [y, y], color=GRID, linewidth=3, solid_capstyle="round"
            )
            ax.plot(
                r["pass^k"],
                y,
                marker=MARKERS.get(c["topology"], "o"),
                color=color,
                markersize=7,
                markeredgecolor="#fcfcfb",
                markeredgewidth=1.5,
            )
            ax.plot(r["pass@k"], y, marker="|", color=INK_2, markersize=10)
        ax.set_yticks(
            ys,
            [_label(c) for c in rows],
            fontsize=8,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(
            f"share of items with MiniCheck ≥ {card['settings']['pass_threshold']}\n"
            "marker = pass^k (every seed passes) · tick = pass@k (any seed passes)"
        )
        _style(ax)
        ax.set_title(
            "Reliability: the gap between 'can pass' and 'always passes'",
            loc="left",
            fontsize=11,
            color=INK,
        )
        fig.tight_layout()
        path = fig_dir / "reliability_pass_k.svg"
        fig.savefig(path)
        plt.close(fig)
        written.append(str(path))
    return written


# --------------------------------------------------------------------------------------------
# entry point


def write_outputs(card: dict[str, Any], out_dir: Path, figures: bool = True) -> dict[str, Path]:
    """Write the scorecard files into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "scorecard.json",
        "csv": out_dir / "leaderboard.csv",
        "markdown": out_dir / "SCORECARD.md",
    }
    paths["json"].write_text(json.dumps(card, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEADERBOARD_FIELDS)
        writer.writeheader()
        for cell in sorted(card["cells"], key=_order):
            writer.writerow(_row(cell))
    written = _figures(card, out_dir / "fig") if figures else []
    paths["markdown"].write_text(_markdown(card, written), encoding="utf-8")
    return paths


def main(argv: Sequence[str] | None = None, repo_root: Path | None = None) -> int:
    """Command-line entry point; see the module docstring."""
    root = repo_root or Path.cwd()
    parser = argparse.ArgumentParser(description="Build the agentic eval scorecard from results/.")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--out-dir", type=Path, default=root / "evals")
    parser.add_argument("--pass-threshold", type=float, default=0.5)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=123)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    cfg = ScorecardConfig(
        pass_threshold=args.pass_threshold,
        bootstrap=BootstrapConfig(n_resamples=args.n_resamples, seed=args.bootstrap_seed),
    )
    card = build_scorecard(scan_results(args.results_dir), cfg)
    paths = write_outputs(card, args.out_dir, figures=not args.no_figures)
    print(f"{card['counts']['cells']} agent configurations scored")
    for name, path in paths.items():
        print(f"  {name:9s} {path}")
    return 0
