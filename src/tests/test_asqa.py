"""ASQA adapter (D-116, D-117, D-122, D-126) against an offline archive shaped like ALCE's. CPU."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
from agent_evals.config import DataConfig, load_config
from agent_evals.datasets import ASQAAdapter, DatasetError
from agent_evals.datasets import base as base_module
from agent_evals.datasets.asqa import format_passages
from agent_evals.registry import DATASETS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MEMBER = "ALCE-data/asqa_eval_gtr_top100.json"
#: Sub-question counts of the fixture questions, in file order.
SUB_QUESTIONS = (2, 2, 3, 4, 5, 6)


def _item(index, n_sub_questions, n_docs=6):
    return {
        "sample_id": f"FAKE_EXAMPLE_q{index}",
        "question": f"Who won fixture contest {index}?",
        "qa_pairs": [
            {"question": f"Sub-question {k}?", "short_answers": [f"Answer {index}.{k}", f"A{k}"]}
            for k in range(n_sub_questions)
        ],
        "annotations": [
            {"long_answer": f"First human answer {index}.", "knowledge": []},
            {"long_answer": f"Second human answer {index}.", "knowledge": []},
        ],
        "answer": f"Second human answer {index}.",
        "docs": [
            {"id": str(d), "title": f"Title {index}-{d}", "text": f"Passage {index}-{d} text."}
            for d in range(n_docs)
        ],
    }


def _archive(path, items, member=MEMBER):
    payload = json.dumps(items).encode("utf-8")
    with tarfile.open(path, "w") as archive:
        info = tarfile.TarInfo(member)
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A tiny ALCE-shaped archive served in place of the Hub download."""
    path = tmp_path / "ALCE-data.tar"
    digest = _archive(path, [_item(i, n) for i, n in enumerate(SUB_QUESTIONS)])
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(path)

    monkeypatch.setattr(base_module, "hf_hub_download", fake_download)
    return path, digest, calls


def _cfg(digest, **overrides):
    values = {
        "name": "tier2_asqa",
        "hf_path": "princeton-nlp/ALCE-data",
        "hf_revision": "b" * 40,
        "hf_filename": "ALCE-data.tar",
        "hf_file_sha256": digest,
        "archive_member": MEMBER,
        "split": "dev",
        "n_passages": 5,
        "strip_citations": True,
        "n_samples": 3,
        "n_strata": 3,
        "stratify_by": "sub_questions",
        "strata": ((2,), (3, 4), (5, 6)),
        "min_source_words": 0,
    }
    values.update(overrides)
    return DataConfig(**values)


def _pool(digest, **overrides):
    return {row["id"]: row for row in ASQAAdapter(_cfg(digest, **overrides)).eligible_pool()}


class TestLoading:
    def test_registered_as_tier2_grounded_qa(self):
        assert DATASETS.get("tier2_asqa") is ASQAAdapter
        assert (ASQAAdapter.TIER, ASQAAdapter.TASK) == (2, "grounded_qa")

    def test_downloads_the_pinned_file_without_credentials(self, archive):
        _, digest, calls = archive
        _pool(digest)
        assert calls == [
            {
                "repo_id": "princeton-nlp/ALCE-data",
                "filename": "ALCE-data.tar",
                "repo_type": "dataset",
                "revision": "b" * 40,
                "token": False,
            }
        ]

    def test_a_changed_file_is_refused(self, archive):
        with pytest.raises(DatasetError, match="SHA-256"):
            _pool("0" * 64)

    def test_a_missing_member_is_refused(self, archive):
        _, digest, _ = archive
        with pytest.raises(DatasetError, match="no member"):
            _pool(digest, archive_member="ALCE-data/eli5.json")

    def test_too_few_passages_is_refused(self, tmp_path, monkeypatch):
        path = tmp_path / "short.tar"
        digest = _archive(path, [_item(0, 2, n_docs=4)])
        monkeypatch.setattr(base_module, "hf_hub_download", lambda **_: str(path))
        with pytest.raises(DatasetError, match="fewer than data.n_passages"):
            _pool(digest)

    def test_requires_n_passages_and_the_dev_split(self, archive):
        _, digest, _ = archive
        with pytest.raises(DatasetError, match="n_passages"):
            ASQAAdapter(_cfg(digest, n_passages=None))
        with pytest.raises(DatasetError, match="'dev' only"):
            ASQAAdapter(_cfg(digest, split="train"))

    def test_is_not_trainable(self, archive):
        _, digest, _ = archive
        with pytest.raises(DatasetError, match="not trainable"):
            ASQAAdapter(_cfg(digest)).assert_trainable("dev")


class TestMapping:
    def test_source_and_evidence_are_the_first_n_passages_in_alce_format(self, archive):
        _, digest, _ = archive
        row = _pool(digest)["FAKE_EXAMPLE_q0"]
        lines = row["source"].split("\n")
        assert row["evidence"] == row["source"]
        assert len(lines) == 5
        assert lines[0] == "Document [1](Title: Title 0-0): Passage 0-0 text."
        assert lines[4] == "Document [5](Title: Title 0-4): Passage 0-4 text."

    def test_question_reference_and_gold(self, archive):
        _, digest, _ = archive
        row = _pool(digest)["FAKE_EXAMPLE_q2"]
        assert row["question"] == "Who won fixture contest 2?"
        assert row["reference"] == "Second human answer 2."
        assert json.loads(row["gold"]) == {
            "short_answers": [["Answer 2.0", "A0"], ["Answer 2.1", "A1"], ["Answer 2.2", "A2"]],
            "long_answers": ["First human answer 2.", "Second human answer 2."],
        }

    def test_rows_convert_to_tier2_examples(self, archive):
        _, digest, _ = archive
        adapter = ASQAAdapter(_cfg(digest))
        example = adapter.as_example(adapter.eligible_pool()[0])
        assert (example.tier, example.task) == (2, "grounded_qa")
        assert example.question is not None and example.gold is not None

    def test_format_passages_numbers_from_one(self):
        passages = [{"title": "T1", "text": "One."}, {"title": "T2", "text": "Two."}]
        assert format_passages(passages) == (
            "Document [1](Title: T1): One.\nDocument [2](Title: T2): Two."
        )


class TestStrata:
    def test_one_stratum_per_group_of_sub_question_counts(self, archive):
        _, digest, _ = archive
        adapter = ASQAAdapter(_cfg(digest))
        pool = adapter.eligible_pool()
        buckets = adapter._partition(pool)
        ids = [sorted(pool["id"][i] for i in bucket) for bucket in buckets]
        assert ids == [
            ["FAKE_EXAMPLE_q0", "FAKE_EXAMPLE_q1"],
            ["FAKE_EXAMPLE_q2", "FAKE_EXAMPLE_q3"],
            ["FAKE_EXAMPLE_q4", "FAKE_EXAMPLE_q5"],
        ]

    def test_draw_takes_equal_shares_and_is_repeatable(self, archive):
        _, digest, _ = archive
        first = ASQAAdapter(_cfg(digest, n_samples=6)).draw_sample_ids()
        again = ASQAAdapter(_cfg(digest, n_samples=6)).draw_sample_ids()
        assert first == again
        assert sorted(first[:2]) == ["FAKE_EXAMPLE_q0", "FAKE_EXAMPLE_q1"]
        assert sorted(first[2:4]) == ["FAKE_EXAMPLE_q2", "FAKE_EXAMPLE_q3"]
        assert sorted(first[4:]) == ["FAKE_EXAMPLE_q4", "FAKE_EXAMPLE_q5"]

    def test_a_count_in_no_group_is_refused(self, archive):
        _, digest, _ = archive
        adapter = ASQAAdapter(_cfg(digest, strata=((2,), (3, 4), (5,))))
        with pytest.raises(DatasetError, match="in no group"):
            adapter._partition(adapter.eligible_pool())

    def test_shipped_config_resolves_to_the_approved_values(self):
        cfg = load_config(CONFIGS / "data" / "tier2_asqa.yaml", ["run.owner=sameer"])
        data = cfg.data
        assert (data.name, data.split, data.n_passages, data.strip_citations) == (
            "tier2_asqa",
            "dev",
            5,
            True,
        )
        assert data.hf_revision == "334fa2e7dd32040c3fef931a123c4be1a81e91a0"
        assert data.hf_file_sha256 == (
            "eda837bf659a91b3648dc6e7ab6b17197664d93593857e8fdf3800b6aa6a98f0"
        )
        assert (data.stratify_by, data.strata) == ("sub_questions", ((2,), (3, 4), (5, 6)))
        assert (data.n_samples, data.sample_seed, len(set(data.sample_ids))) == (500, 123, 500)
        assert cfg.sampling.max_tokens == 300

    def test_frozen_ids_replay_in_order(self, archive):
        _, digest, _ = archive
        ids = ("FAKE_EXAMPLE_q5", "FAKE_EXAMPLE_q0", "FAKE_EXAMPLE_q3")
        rows = ASQAAdapter(replace(_cfg(digest), sample_ids=ids)).build()
        assert tuple(rows["id"]) == ids
