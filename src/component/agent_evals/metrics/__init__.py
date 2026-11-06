"""Metrics: factual consistency and NLG quality, one registered class per metric (D-022).

Importing this package registers every metric it holds. Heavy scorer libraries are imported inside
each metric's constructor, so importing the package loads no models and downloads nothing.
"""

from .alignscore import AlignScore
from .base import Metric, MetricError, reference_texts
from .bertscore import BertScore
from .citation import CitationQuality
from .citation_markers import remove_citations
from .factcc import FactCC
from .minicheck import MiniCheck
from .rouge import RougeL
from .str_em import StrEM
from .summac import SummaC
from .worked_example_overlap import (
    ASQASelfRefineExampleOverlap,
    ASQAWorkedExampleOverlap,
    WorkedExampleOverlap,
)

__all__ = [
    "ASQASelfRefineExampleOverlap",
    "ASQAWorkedExampleOverlap",
    "AlignScore",
    "BertScore",
    "CitationQuality",
    "FactCC",
    "Metric",
    "MetricError",
    "MiniCheck",
    "RougeL",
    "StrEM",
    "SummaC",
    "WorkedExampleOverlap",
    "reference_texts",
    "remove_citations",
]
