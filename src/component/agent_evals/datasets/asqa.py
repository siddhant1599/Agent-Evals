"""ASQA from ALCE's release: retrieved passages in, gold answers out, strata by ambiguity.

ALCE (Gao et al. 2023) ships ASQA's development split with each question's top 100 GTR passages in
one archive on the Hub (D-116). The adapter reads that pinned, checksummed file, gives the model the
first ``data.n_passages`` passages in ALCE's format (D-117), and keeps the gold annotations the
metrics need in ``Example.gold`` (D-126). The sample is stratified by number of sub-questions
(D-122).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from datasets import Dataset

from ..config import DataConfig
from ..logging_utils import get_logger
from ..registry import DATASETS
from .base import DatasetAdapter, DatasetError

__all__ = [
    "ASQAAdapter",
    "ASQATrainAdapter",
    "PASSAGE_FORMAT",
    "format_passages",
    "parse_passages",
]

logger = get_logger(__name__)

#: ALCE's passage format (``doc_prompt`` in ``prompts/asqa_default.json``, commit ``246c476``),
#: numbered from 1 (D-117). Shared with the Generator's ASQA prompt so examples and the question
#: under work read the same way.
PASSAGE_FORMAT = "Document [{number}](Title: {title}): {text}"
_PASSAGE_LINE = re.compile(r"Document \[(\d+)\]\(Title: (.*?)\): (.*)")


def format_passages(passages: Sequence[Mapping[str, Any]]) -> str:
    """Render passages in ALCE's format, one per line, numbered from 1.

    Args:
        passages: Each with ``title`` and ``text``, in retrieval order.

    Returns:
        The passages joined by newlines, with no trailing newline.
    """
    return "\n".join(
        PASSAGE_FORMAT.format(number=number, title=passage["title"], text=passage["text"])
        for number, passage in enumerate(passages, start=1)
    )


def parse_passages(text: str) -> list[dict[str, str]]:
    """Recover the passages from text written by :func:`format_passages`.

    Args:
        text: Passages in ALCE's format, one per line, numbered from 1.

    Returns:
        Each passage's ``title`` and ``text``, in order.

    Raises:
        ValueError: If a line is not in the format, the numbering is not 1, 2, ..., or the
            passages do not render back to exactly ``text``.
    """
    passages = []
    for number, line in enumerate(text.split("\n"), start=1):
        match = _PASSAGE_LINE.fullmatch(line)
        if match is None or int(match.group(1)) != number:
            raise ValueError(f"line {number} is not passage [{number}] in ALCE's format")
        passages.append({"title": match.group(2), "text": match.group(3)})
    if format_passages(passages) != text:
        raise ValueError("the parsed passages do not render back to the same text")
    return passages


@DATASETS.register("tier2_asqa")
class ASQAAdapter(DatasetAdapter):
    """Tier 2 grounded long-form QA. Evaluation-only here: ALCE's file holds the dev split only.

    Per question: ``source`` and ``evidence`` are the first ``data.n_passages`` passages in ALCE's
    format, ``question`` is the ambiguous question, ``reference`` is ALCE's ``answer`` field (the
    second of the two human answers), and ``gold`` holds every sub-question's short answers and
    both human answers.

    Args:
        cfg: Must set ``n_passages`` and use split ``dev``.

    Raises:
        DatasetError: If ``n_passages`` is unset or the split is not the file's.
    """

    TIER = 2
    TASK = "grounded_qa"
    #: The only split in ALCE's ASQA file (Gao et al. 2023, §2).
    FILE_SPLIT: ClassVar[str] = "dev"

    def __init__(self, cfg: DataConfig) -> None:
        super().__init__(cfg)
        if self._n_passages is None:
            raise DatasetError(f"{self._name} needs data.n_passages (D-117)")
        if self._split != self.FILE_SPLIT:
            raise DatasetError(
                f"{self._name} reads ALCE's ASQA file, which holds split {self.FILE_SPLIT!r} only"
            )

    def _load_split(self, split: str) -> Dataset:
        if split != self.FILE_SPLIT:
            raise DatasetError(f"{self._name} has no split {split!r}")
        rows = [
            self._row(item["sample_id"], item, item["question"], item["docs"], item["answer"])
            for item in json.loads(self._read_pinned_file())
        ]
        return Dataset.from_list(rows)

    def _row(
        self,
        sample_id: str,
        annotated: Mapping[str, Any],
        question: str,
        docs: Sequence[Mapping[str, Any]],
        answer: str,
    ) -> dict[str, str]:
        """One raw row: the first ``data.n_passages`` passages and the gold annotations.

        Args:
            sample_id: ASQA's ID for the question.
            annotated: The item holding ``qa_pairs`` and ``annotations`` in ASQA's format.
            question: The ambiguous question.
            docs: Retrieved passages, each with ``title`` and ``text``, in retrieval order.
            answer: The human answer used as ``reference``.

        Raises:
            DatasetError: If the question has fewer passages than ``data.n_passages``.
        """
        passages = list(docs[: self._n_passages])
        if len(passages) < self._n_passages:
            raise DatasetError(
                f"question {sample_id} has {len(passages)} passages, fewer than "
                f"data.n_passages={self._n_passages}"
            )
        gold = {
            "short_answers": [pair["short_answers"] for pair in annotated["qa_pairs"]],
            "long_answers": [note["long_answer"] for note in annotated["annotations"]],
        }
        return {
            "sample_id": sample_id,
            "question": question,
            "passages": format_passages(passages),
            "answer": answer,
            "gold": json.dumps(gold, ensure_ascii=False),
        }

    def _map_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "id": row["sample_id"],
            "source": row["passages"],
            "evidence": row["passages"],
            "reference": row["answer"],
            "question": row["question"],
            "gold": row["gold"],
        }

    def _partition(self, pool: Dataset) -> tuple[tuple[int, ...], ...]:
        """One stratum per group of sub-question counts in ``data.strata`` (D-122)."""
        if self._stratify_by != "sub_questions":
            return super()._partition(pool)
        stratum_of = {count: index for index, group in enumerate(self._strata) for count in group}
        buckets: list[list[int]] = [[] for _ in self._strata]
        for index, gold in enumerate(pool["gold"]):
            count = len(json.loads(gold)["short_answers"])
            if count not in stratum_of:
                raise DatasetError(
                    f"{pool['id'][index]} has {count} sub-questions, in no group of data.strata"
                )
            buckets[stratum_of[count]].append(index)
        logger.info(
            "%s sub-question strata %s; stratum sizes=%s",
            self._name,
            self._strata,
            [len(bucket) for bucket in buckets],
        )
        return tuple(tuple(bucket) for bucket in buckets)


@DATASETS.register("tier2_asqa_train")
class ASQATrainAdapter(ASQAAdapter):
    """ASQA's training split with GTR top-5 passages, for T5's GRPO training only (D-132).

    ALCE ships passages for the dev split only. The questions, IDs and gold annotations come from
    the official ASQA release; the passages come from Huang et al. 2024's release, whose files
    hold every training question with its top-5 GTR passages from the 2018-12-20 Wikipedia
    snapshot, retrieved as ALCE retrieved the dev split. The two are joined on the question text,
    which is unique in the training split. Both are ``data.source_files``, checked by SHA-256.

    ``reference`` is the question's human answer (the training split has one per question), as
    ALCE's ``answer`` field is the last of the dev split's two. The ALCE worked examples are
    training questions and are removed through ``data.exclude_ids`` (D-132).

    Args:
        cfg: Must set ``n_passages``, split ``train``, and ``source_files`` naming the official
            release ``ASQA.json`` and every passage file.

    Raises:
        DatasetError: As :class:`ASQAAdapter`; on loading, if the files disagree (see
            ``_load_split``).
    """

    FILE_SPLIT: ClassVar[str] = "train"
    TRAINABLE_SPLITS = frozenset({"train"})
    #: The official release's file name (``gs://gresearch/ASQA/data/ASQA.json``): its ``train``
    #: and ``dev`` splits, keyed by ``sample_id``.
    ANNOTATIONS_FILE: ClassVar[str] = "ASQA.json"

    def _load_split(self, split: str) -> Dataset:
        """Join the official training split with its passages.

        Raises:
            DatasetError: If a training question is also a dev question, a question is missing
                from either source or appears twice, or the passage file's sub-questions differ
                from the official ones.
        """
        if split != self.FILE_SPLIT:
            raise DatasetError(f"{self._name} has no split {split!r}")
        official = json.loads(self._read_source_file(self.ANNOTATIONS_FILE))
        train, dev = official["train"], official["dev"]
        dev_questions = {item["ambiguous_question"] for item in dev.values()}
        leaked = set(train) & set(dev) or {
            sid for sid, item in train.items() if item["ambiguous_question"] in dev_questions
        }
        if leaked:
            raise DatasetError(f"training questions also in ASQA's dev split: {sorted(leaked)}")
        retrieved = self._passages_by_question()
        questions = [item["ambiguous_question"] for item in train.values()]
        if len(set(questions)) != len(questions):
            raise DatasetError("ASQA's training split repeats a question; it cannot be joined")
        unmatched = set(questions) ^ set(retrieved)
        if unmatched:
            raise DatasetError(
                f"{len(unmatched)} questions are in only one of the official release and the "
                f"passage files, e.g. {sorted(unmatched)[0]!r}"
            )
        rows = []
        for sample_id, item in train.items():
            question = item["ambiguous_question"]
            found = retrieved[question]
            if found["qa_pairs"] != item["qa_pairs"]:
                raise DatasetError(f"question {sample_id}: the passage file's sub-questions differ")
            answer = item["annotations"][-1]["long_answer"]
            rows.append(self._row(sample_id, item, question, found["docs"], answer))
        logger.info("%s joined %d training questions with their passages", self._name, len(rows))
        return Dataset.from_list(rows)

    def _passages_by_question(self) -> dict[str, Mapping[str, Any]]:
        """Every passage file's items, by question text."""
        by_question: dict[str, Mapping[str, Any]] = {}
        for name in self._source_files:
            if name == self.ANNOTATIONS_FILE:
                continue
            for item in json.loads(self._read_source_file(name)):
                if item["question"] in by_question:
                    raise DatasetError(f"question {item['question']!r} appears twice in {name}")
                by_question[item["question"]] = item
        return by_question
