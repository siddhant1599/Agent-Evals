"""Dataset adapters; import implementations so their registry entries are available."""

from .asqa import ASQAAdapter, ASQATrainAdapter
from .base import DatasetAdapter, DatasetError
from .cnndm import CnnDailyMailAdapter
from .samsum import SAMSumAdapter
from .xsum import XSumAdapter

__all__ = [
    "ASQAAdapter",
    "ASQATrainAdapter",
    "CnnDailyMailAdapter",
    "DatasetAdapter",
    "DatasetError",
    "SAMSumAdapter",
    "XSumAdapter",
]
