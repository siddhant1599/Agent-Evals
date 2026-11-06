"""Finished runs compared with a baseline: gains, length-adjusted changes, and loop behaviour.

For every topology and backbone in ``analysis.runs`` this reports, against the baseline topology
on the same backbone (D-115):

- ``paired``: the paired gain on each metric, with its bootstrap interval;
- ``length_adjusted``: for ``analysis.length_metrics``, the same change with the change in answer
  length held fixed (:meth:`PairedBootstrap.length_adjusted`);
- ``feedback_replies``: for loops with a feedback step, how its replies are written;
- ``first_feedback``: the score change grouped by the kind of the first feedback reply;
- ``rewrites``: on questions rewritten at least once, the first draft against the final answer.

It generates nothing and reads only ``results/`` (AGENTS.md §3). It writes an ordinary
``results/<run_id>/``: one row per source run and question in ``metrics.csv``, the sections above
in ``summary.json``, and the source runs with the SHA-256 of each file read in the manifest.

Usage::

    python src/run_analysis.py --config src/configs/analysis/<name>.yaml --set run.owner=<name>
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..agents.generator import Generator
from ..agents.self_feedback import STOP_WORD, SelfFeedback, is_stop_reply
from ..config import AnalysisConfig, ExperimentConfig, LoggingConfig, load_config
from ..logging_utils import configure_logging, get_logger
from ..metrics.citation_markers import CITATION
from ..run import RunDirectory
from .bootstrap import AnalysisError, PairedBootstrap
from .sources import SourceRun, TraceMessage, read_source_run

__all__ = [
    "REPLY_KINDS",
    "TopologyDiagnostics",
    "main",
    "reply_kind",
    "uncited_share",
]

logger = get_logger(__name__)

#: D-150's feedback format: one line per unsupported passage, each starting with this.
BULLET = "- "
#: How a feedback reply is written: the stop signal (D-101), only ``BULLET`` lines, the stop word
#: followed by more text, or anything else.
REPLY_KINDS = ("stop", "bullets", "stop_word_then_text", "prose")
_STARTS_WITH_STOP_WORD = re.compile(rf"^\s*{re.escape(STOP_WORD)}\b", re.IGNORECASE)

SentenceSplitter = Callable[[str], list[str]]
Group = tuple[str, str]


def reply_kind(text: str) -> str:
    """How a feedback reply is written, one of ``REPLY_KINDS``.

    Args:
        text: A feedback reply.

    Returns:
        ``stop`` if D-101's rule reads it as the stop signal; ``bullets`` if every non-blank line
        starts with ``BULLET``; ``stop_word_then_text`` if it opens with the stop word and goes on;
        otherwise ``prose``.
    """
    if is_stop_reply(text):
        return "stop"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and all(line.startswith(BULLET) for line in lines):
        return "bullets"
    if _STARTS_WITH_STOP_WORD.match(text):
        return "stop_word_then_text"
    return "prose"


def uncited_share(text: str, split_sentences: SentenceSplitter) -> float | None:
    """The share of an answer's sentences that cite no passage, by the citation metric's rule.

    Args:
        text: An answer.
        split_sentences: The sentence splitter the citation metric uses (NLTK, as ALCE).

    Returns:
        A value in [0, 1], or None for an answer with no sentences.
    """
    sentences = split_sentences(text)
    if not sentences:
        return None
    return sum(1 for s in sentences if not CITATION.search(s)) / len(sentences)


def _nltk_sentences(text: str) -> list[str]:
    import nltk

    return nltk.tokenize.sent_tokenize(text)


class TopologyDiagnostics:
    """Reads ``analysis.runs`` and writes the comparison with the baseline.

    Args:
        cfg: The analysis config; ``cfg.analysis`` must be set.
        repo_root: Repository root; source runs are read from ``run.output_root`` under it.
        split_sentences: Sentence splitter for :func:`uncited_share`. Defaults to NLTK's, the
            citation metric's.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        repo_root: Path,
        split_sentences: SentenceSplitter | None = None,
    ) -> None:
        if cfg.analysis is None:
            raise AnalysisError("set the analysis group: which runs to read")
        self._cfg = cfg
        self._analysis: AnalysisConfig = cfg.analysis
        self._results = repo_root / cfg.run.output_root
        self._repo_root = repo_root
        self._split = split_sentences or _nltk_sentences
        self._bootstrap = PairedBootstrap(cfg.analysis.bootstrap)

    def run(self) -> Path:
        """Read the source runs and write the analysis run.

        Returns:
            The analysis's run directory.

        Raises:
            AnalysisError: If a source run cannot be read or the runs do not form a comparison.
        """
        sources = [
            read_source_run(self._results / run_id, self._analysis.metrics)
            for run_id in self._analysis.runs
        ]
        groups = self._group(sources)
        manifest = {
            "source_runs": [
                {
                    "run_id": s.run_id,
                    "metrics_sha256": s.metrics_sha256,
                    "trace_sha256": s.trace_sha256,
                }
                for s in sources
            ]
        }
        with RunDirectory(self._cfg, self._repo_root, manifest_extra=manifest) as run:
            records = {
                g: [self._record(s, q) for s in runs for q in sorted(s.scores)]
                for g, runs in groups.items()
            }
            for group_records in records.values():
                for record in group_records:
                    run.metrics.write(record)
            run.add_summary(**self._summarize(records))
        return run.path

    def _group(self, sources: Sequence[SourceRun]) -> dict[Group, list[SourceRun]]:
        """Group the runs by topology and backbone, refusing an incomplete comparison."""
        groups: dict[Group, list[SourceRun]] = defaultdict(list)
        cells = set()
        for source in sources:
            if source.task != self._analysis.task:
                raise AnalysisError(
                    f"{source.run_id}: task {source.task!r}, not the configured one"
                )
            cell = (source.topology, source.backbone, source.seed)
            if cell in cells:
                raise AnalysisError(f"topology, backbone and seed appear twice: {cell}")
            cells.add(cell)
            groups[(source.topology, source.backbone)].append(source)
        for (topology, backbone), runs in groups.items():
            baseline = groups.get((self._analysis.baseline, backbone))
            if baseline is None:
                raise AnalysisError(
                    f"{topology} on {backbone} has no {self._analysis.baseline} runs"
                )
            questions = set(baseline[0].scores)
            if any(set(r.scores) != questions for r in runs):
                raise AnalysisError(f"{topology} on {backbone}: runs answer different questions")
        return dict(groups)

    def _record(self, source: SourceRun, question: str) -> dict[str, Any]:
        """One row of ``metrics.csv``: a question's scores, answers and first feedback."""
        calls = source.calls[question]
        answers = [c for c in calls if c.role == Generator.ROLE]
        if not answers:
            raise AnalysisError(f"{source.run_id}: question {question} has no generated answer")
        feedback = [c for c in calls if c.role == SelfFeedback.ROLE]
        first = feedback[0] if feedback else None
        return {
            "source_run_id": source.run_id,
            "topology": source.topology,
            "backbone": source.backbone,
            "seed": source.seed,
            "example_id": question,
            "n_answers": len(answers),
            **self._answer_fields("draft", answers[0]),
            **self._answer_fields("final", answers[-1]),
            "first_feedback": reply_kind(first.content) if first else "",
            "first_feedback_hit_cap": first.hit_cap if first else "",
            "feedback_kinds": " ".join(reply_kind(c.content) for c in feedback),
            "feedback_hit_cap": sum(c.hit_cap for c in feedback),
            **source.scores[question],
        }

    def _answer_fields(self, label: str, answer: TraceMessage) -> dict[str, Any]:
        share = uncited_share(answer.content, self._split)
        return {
            f"{label}_tokens": answer.completion_tokens,
            f"{label}_hit_cap": answer.hit_cap,
            f"{label}_uncited_share": "" if share is None else share,
        }

    def _summarize(self, records: Mapping[Group, list[dict[str, Any]]]) -> dict[str, Any]:
        baseline = self._analysis.baseline
        sections: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))
        for (topology, backbone), rows in sorted(records.items()):
            sections["n_seeds"][topology][backbone] = len({r["seed"] for r in rows})
            if topology == baseline:
                continue
            base_rows = records[(baseline, backbone)]
            sections["paired"][topology][backbone] = self._paired(rows, base_rows)
            if self._analysis.length_metrics:
                sections["length_adjusted"][topology][backbone] = self._length(rows, base_rows)
            if any(r["feedback_kinds"] for r in rows):
                sections["feedback_replies"][topology][backbone] = _reply_counts(rows)
                sections["first_feedback"][topology][backbone] = self._by_first(rows, base_rows)
            rewritten = [r for r in rows if r["n_answers"] > 1]
            if rewritten:
                sections["rewrites"][topology][backbone] = _rewrites(rewritten)
        return {"baseline": baseline, **{k: _plain(v) for k, v in sections.items()}}

    def _paired(self, rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict:
        result = {}
        for metric in self._analysis.metrics:
            topo, base = _question_means(rows, metric), _question_means(base_rows, metric)
            questions = sorted(base)
            interval = self._bootstrap.mean(np.array([topo[q] - base[q] for q in questions]))
            result[metric] = {
                "baseline": float(np.mean(list(base.values()))),
                "topology": float(np.mean(list(topo.values()))),
                "gain": interval.estimate,
                "low": interval.low,
                "high": interval.high,
            }
        return result

    def _length(self, rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict:
        topo_len = _question_means(rows, "final_tokens")
        base_len = _question_means(base_rows, "final_tokens")
        questions = sorted(base_len)
        length_diffs = np.array([topo_len[q] - base_len[q] for q in questions])
        result = {}
        for metric in self._analysis.length_metrics:
            topo, base = _question_means(rows, metric), _question_means(base_rows, metric)
            fit = self._bootstrap.length_adjusted(
                np.array([topo[q] - base[q] for q in questions]), length_diffs
            )
            result[metric] = {
                "observed": fit.observed,
                "mean_length_change_tokens": fit.mean_length_change,
                "slope_per_token": fit.slope,
                "adjusted": fit.adjusted.estimate,
                "low": fit.adjusted.low,
                "high": fit.adjusted.high,
                "dropped_resamples": fit.dropped_resamples,
            }
        return result

    def _by_first(self, rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict:
        """Per kind of first feedback reply: the change of each answer against the baseline.

        Each row is one question and seed; its change is its score minus the baseline's score on
        that question averaged over seeds, so seeds are not matched across topologies (D-115).
        """
        base = {m: _question_means(base_rows, m) for m in self._analysis.metrics}
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_kind[row["first_feedback"]].append(row)
        result = {}
        for kind, group in sorted(by_kind.items()):
            entry: dict[str, Any] = {
                "n": len(group),
                "mean_answers": float(np.mean([r["n_answers"] for r in group])),
            }
            for metric in self._analysis.metrics:
                changes = [r[metric] - base[metric][r["example_id"]] for r in group]
                entry[f"{metric}_change"] = float(np.mean(changes))
            result[kind] = entry
        return result


def _question_means(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """A value averaged over seeds within each question (D-115)."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        values[row["example_id"]].append(float(row[key]))
    return {q: float(np.mean(v)) for q, v in values.items()}


def _reply_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How every feedback reply of one topology and backbone is written; how many hit the cap."""
    counts = Counter(kind for r in rows for kind in r["feedback_kinds"].split())
    return {
        "n": sum(counts.values()),
        **{kind: counts.get(kind, 0) for kind in REPLY_KINDS},
        "hit_cap": sum(r["feedback_hit_cap"] for r in rows),
    }


def _rewrites(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """First draft against final answer, over answers rewritten at least once."""

    def mean(key: str) -> float:
        values = [float(r[key]) for r in rows if r[key] != ""]
        return float(np.mean(values)) if values else float("nan")

    return {
        "n_rewritten": len(rows),
        **{
            f"{label}_{field}": mean(f"{label}_{field}")
            for label in ("draft", "final")
            for field in ("tokens", "uncited_share", "hit_cap")
        },
    }


def _plain(value: Any) -> Any:
    """Nested defaultdicts as plain dicts, for ``summary.json``."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def main(argv: list[str] | None, repo_root: Path) -> int:
    """Command-line wrapper.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root.

    Returns:
        0 on success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Compare finished runs with a baseline (D-115).")
    parser.add_argument("--config", required=True, type=Path, help="analysis config")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config, args.overrides or [])
        configure_logging(cfg.logging)
        path = TopologyDiagnostics(cfg, repo_root).run()
        logger.info("analysis written to %s", path)
    except Exception:  # the one top-level catch (AGENTS.md §6)
        configure_logging(LoggingConfig())
        logger.exception("analysis did not complete")
        return 1
    return 0
