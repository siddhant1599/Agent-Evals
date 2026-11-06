"""Analyses of finished runs. They read ``results/`` and nothing else (AGENTS.md §3)."""

from .bootstrap import AnalysisError, Interval, LengthAdjusted, PairedBootstrap
from .diagnostics import REPLY_KINDS, TopologyDiagnostics, reply_kind, uncited_share
from .sources import SourceRun, TraceMessage, read_source_run

__all__ = [
    "REPLY_KINDS",
    "AnalysisError",
    "Interval",
    "LengthAdjusted",
    "PairedBootstrap",
    "SourceRun",
    "TopologyDiagnostics",
    "TraceMessage",
    "read_source_run",
    "reply_kind",
    "uncited_share",
]
