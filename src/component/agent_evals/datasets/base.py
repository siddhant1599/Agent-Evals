"""Shared evidence schema, length sampling, and training eligibility for dataset adapters."""

from __future__ import annotations

import hashlib
import tarfile
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import asdict, fields
from operator import itemgetter
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from datasets import Dataset, Features, Value, load_dataset
from huggingface_hub import hf_hub_download

from ..config import DataConfig
from ..logging_utils import get_logger
from ..seeding import make_sampling_rng
from ..types import Example

logger = get_logger(__name__)


class DatasetError(ValueError):
    """Raised when a dataset, frozen sample, or training request violates the contract."""


class DatasetAdapter(ABC):
    """Base for all tiers; dataset-specific classes implement raw loading and column mapping.

    Args:
        cfg: Immutable dataset configuration. Parameters bind once at construction.

    ``build`` is concrete and shared by every tier; subclasses supply only ``_map_row``.
    ``_load_raw`` loads the pinned Hub split and is overridden by tests with offline fixtures.
    ``_partition`` is the customization point for later stratification policies.
    """

    TIER: ClassVar[int]
    TASK: ClassVar[str]
    TRAINABLE_SPLITS: ClassVar[frozenset[str]] = frozenset()
    SCHEMA: ClassVar[tuple[str, ...]] = tuple(field.name for field in fields(Example))
    _TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("id", "task", "source", "evidence", "reference")

    def __init__(self, cfg: DataConfig) -> None:
        self._name = cfg.name
        self._hf_path = cfg.hf_path
        self._hf_revision = cfg.hf_revision
        self._hf_config = cfg.hf_config
        self._hf_filename = cfg.hf_filename
        self._hf_file_sha256 = cfg.hf_file_sha256
        self._archive_member = cfg.archive_member
        self._n_passages = cfg.n_passages
        self._dedupe_sources = cfg.dedupe_sources
        self._exclude_split = cfg.exclude_sources_from_split
        self._split = cfg.split
        self._n_samples = cfg.n_samples
        self._n_strata = cfg.n_strata
        self._stratify_by = cfg.stratify_by
        self._strata = cfg.strata
        self._sample_seed = cfg.sample_seed
        self._minimum = cfg.min_source_words
        self._maximum = cfg.max_source_words
        self._sample_ids = cfg.sample_ids
        self._source_files = {entry.name: entry for entry in cfg.source_files or ()}
        self._cache_dir = Path(cfg.cache_dir).expanduser()
        self._exclude_ids = cfg.exclude_ids
        self._tier = self.TIER
        self._task = self.TASK
        if type(self._tier) is not int or self._tier not in (1, 2, 3) or not self._task:
            raise DatasetError("adapter must declare a valid TIER and TASK")
        self._features = Features(
            {
                name: Value(
                    "int64"
                    if name in ("tier", "source_words")
                    else "bool" if name == "truncated" else "string"
                )
                for name in self.SCHEMA
            }
        )

    def build(self) -> Dataset:
        """Replay the frozen evaluation IDs, preserving their order and the common schema.

        Shared by every tier: subclasses customise loading and column mapping through
        ``_load_raw`` and ``_map_row``, not by overriding this.

        Returns:
            An evidence-paired HuggingFace Dataset with exactly the configured sample size.

        Raises:
            DatasetError: If IDs are unfrozen, absent, duplicated, or no longer eligible.
        """
        if self._sample_ids is None:
            raise DatasetError(
                "evaluation sample has not been frozen; explicitly prepare sample IDs and "
                "store them in data.sample_ids before building"
            )
        pool = self._prepare_pool()
        positions = {identifier: index for index, identifier in enumerate(pool["id"])}
        missing = set(self._sample_ids) - positions.keys()
        if missing:
            raise DatasetError(
                f"frozen sample IDs are missing or excluded by filtering: {sorted(missing)}"
            )
        selected = pool.select([positions[identifier] for identifier in self._sample_ids])
        logger.info(
            "%s replayed %s frozen IDs from revision %s",
            self._name,
            len(selected),
            self._hf_revision,
        )
        return selected

    def eligible_pool(self) -> Dataset:
        """Every example that survives this config's filters, in id order.

        Returns:
            The filtered pool frozen IDs are drawn from, in the common schema.

        Raises:
            DatasetError: If no example survives filtering.
        """
        return self._prepare_pool()

    def draw_sample_ids(self) -> tuple[str, ...]:
        """Prepare ordered IDs explicitly, without saving configuration or changing global seeds.

        Returns:
            Unique IDs drawn without replacement, in stratum order. Freeze these in YAML.

        Raises:
            DatasetError: If already frozen, or a stratum cannot supply its allocated quota.
        """
        if self._sample_ids is not None:
            raise DatasetError("sample IDs are already frozen; refusing to redraw")
        pool = self._prepare_pool()
        buckets = self._partition(pool)
        quotient, remainder = divmod(self._n_samples, self._n_strata)
        rng = make_sampling_rng(self._sample_seed)
        indices: list[int] = []
        for index, bucket in enumerate(buckets):
            quota = quotient + (index < remainder)
            if len(bucket) < quota:
                raise DatasetError(
                    f"stratum {index + 1} has {len(bucket)} rows but requires {quota}; "
                    "refusing to reduce the sample or change strata"
                )
            indices.extend(rng.sample(bucket, quota))
        identifiers = pool["id"]
        return tuple(identifiers[index] for index in indices)

    def assert_trainable(self, split: str) -> None:
        """Reject training outside the adapter's declared split permissions.

        Every backbone is trained the same way (D-047), so eligibility depends on the dataset and
        split alone.

        Args:
            split: Requested training split; must match the split bound to this adapter.

        Returns:
            None when eligible. The RL training entrypoint calls this gate before loading a model.

        Raises:
            DatasetError: For Tier 3, evaluation-only datasets, or a split not declared trainable.
        """
        if self._tier == 3:
            raise DatasetError("Tier 3 is never trainable")
        if split != self._split:
            raise DatasetError(
                f"requested split {split!r} differs from loaded split {self._split!r}"
            )
        if split not in self.TRAINABLE_SPLITS:
            raise DatasetError(f"{self._name} split={split!r} is not trainable")

    def as_example(self, row: Mapping[str, Any]) -> Example:
        """Validate and convert one row from the common dataset schema.

        Args:
            row: A complete evidence-paired row, including source length and truncation flag.

        Returns:
            The immutable Example, preserving evidence and original source length.

        Raises:
            DatasetError: On schema, tier/task, text, or recorded length mismatch.
        """
        if set(row) != set(self.SCHEMA):
            raise DatasetError(f"expected exactly the common schema: {self.SCHEMA}")
        self._validate_text(row)
        if type(row["tier"]) is not int or row["tier"] != self._tier or row["task"] != self._task:
            raise DatasetError("row tier/task must match its adapter")
        example = Example(
            **{field.name: row[field.name] for field in fields(Example) if field.init}
        )
        if type(row["source_words"]) is not int or row["source_words"] != example.source_words:
            raise DatasetError("source_words must match the original source whitespace word count")
        return example

    def _prepare_pool(self) -> Dataset:
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for raw_row in self._load_raw():
            row = self._normalize(raw_row)
            if row["id"] in seen:
                raise DatasetError(f"duplicate raw example id: {row['id']!r}")
            seen.add(row["id"])
            rows.append(row)
        rows.sort(key=itemgetter("id"))
        if self._exclude_ids is not None:
            rows = self._drop_excluded_ids(rows)
        if self._dedupe_sources:
            rows = self._drop_duplicate_sources(rows)
        if self._exclude_split is not None:
            rows = self._drop_sources_in_split(rows, self._exclude_split)
        kept = [
            row
            for row in rows
            if row["source_words"] >= self._minimum
            and (self._maximum is None or row["source_words"] <= self._maximum)
        ]
        logger.info(
            "%s/%s source-word filtering: %s → %s (min=%s, max=%s)",
            self._name,
            self._split,
            len(rows),
            len(kept),
            self._minimum,
            self._maximum,
        )
        if not kept:
            raise DatasetError("no examples remain after source-word filtering")
        return Dataset.from_list(kept, features=self._features)

    def _drop_excluded_ids(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop ``data.exclude_ids``; each must exist, so a mistyped ID cannot pass silently."""
        excluded = set(self._exclude_ids or ())
        missing = excluded - {row["id"] for row in rows}
        if missing:
            raise DatasetError(f"data.exclude_ids not in {self._name}: {sorted(missing)}")
        kept = [row for row in rows if row["id"] not in excluded]
        logger.info(
            "%s/%s removal of excluded IDs: %s → %s", self._name, self._split, len(rows), len(kept)
        )
        return kept

    def _drop_duplicate_sources(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep the first row per identical source text; rows arrive sorted by id (D-035)."""
        sources: set[str] = set()
        unique = []
        for row in rows:
            if row["source"] not in sources:
                sources.add(row["source"])
                unique.append(row)
        logger.info(
            "%s/%s duplicate-source removal: %s → %s",
            self._name,
            self._split,
            len(rows),
            len(unique),
        )
        return unique

    def _drop_sources_in_split(
        self, rows: list[dict[str, Any]], split: str
    ) -> list[dict[str, Any]]:
        """Drop rows whose source text also appears in ``split`` (D-055)."""
        excluded = {self._map_row(raw_row)["source"] for raw_row in self._load_split(split)}
        kept = [row for row in rows if row["source"] not in excluded]
        logger.info(
            "%s/%s removal of sources also in %s: %s → %s",
            self._name,
            self._split,
            split,
            len(rows),
            len(kept),
        )
        return kept

    def _normalize(self, raw_row: Mapping[str, Any]) -> dict[str, Any]:
        mapped = dict(self._map_row(raw_row))
        mapped.update(tier=self._tier, task=self._task)
        self._validate_text(mapped)
        example = Example(
            id=mapped["id"],
            tier=self._tier,
            task=self._task,
            source=mapped["source"],
            evidence=mapped["evidence"],
            reference=mapped["reference"],
            question=mapped.get("question"),
            gold=mapped.get("gold"),
        )
        return asdict(example)

    def _validate_text(self, row: Mapping[str, Any]) -> None:
        for name in self._TEXT_FIELDS:
            if not isinstance(row.get(name), str):
                raise DatasetError(f"{name} must be a string")
        if not row["id"].strip():
            raise DatasetError("example id must be non-empty")
        if row.get("question") is not None and not isinstance(row["question"], str):
            raise DatasetError("question must be a string or None")
        if row.get("gold") is not None and not isinstance(row["gold"], str):
            raise DatasetError("gold must be a JSON string or None")

    def _partition(self, pool: Dataset) -> tuple[tuple[int, ...], ...]:
        """Partition the filtered pool by linear empirical quantiles; boundary ties go lower."""
        if self._stratify_by != "whitespace_words":
            raise DatasetError(f"{self._name} cannot stratify by {self._stratify_by!r}")
        lengths = pool["source_words"]
        fractions = np.arange(1, self._n_strata) / self._n_strata
        boundaries = np.quantile(lengths, fractions, method="linear").tolist()
        buckets: list[list[int]] = [[] for _ in range(self._n_strata)]
        for index, length in enumerate(lengths):
            buckets[bisect_left(boundaries, length)].append(index)
        logger.info(
            "%s filtered length boundaries=%s; stratum sizes=%s",
            self._name,
            boundaries,
            [len(bucket) for bucket in buckets],
        )
        return tuple(tuple(bucket) for bucket in buckets)

    def _load_raw(self) -> Dataset:
        """Load the pinned raw split from the Hub; tests override this hook with an offline fixture.

        Returns:
            The configured split at the pinned revision, loaded without credentials.

        Raises:
            DatasetError: If the loader does not return one non-streaming split.
        """
        return self._load_split(self._split)

    def _load_split(self, split: str) -> Dataset:
        """Load any split of the pinned dataset; tests override this hook with an offline fixture.

        Args:
            split: The split to load.

        Returns:
            That split at the pinned revision, loaded without credentials.

        Raises:
            DatasetError: If the loader does not return one non-streaming split.
        """
        raw = load_dataset(
            self._hf_path,
            name=self._hf_config,
            revision=self._hf_revision,
            split=split,
            token=False,
        )
        if not isinstance(raw, Dataset):
            raise DatasetError(f"{self._name} requires one non-streaming dataset split")
        return raw

    def _read_pinned_file(self) -> bytes:
        """Read ``data.hf_filename`` at the pinned revision, verified, for file-shipped datasets.

        The file is downloaded into the local Hub cache, its SHA-256 is checked against
        ``data.hf_file_sha256``, and either its bytes or, for a tar archive, the bytes of
        ``data.archive_member`` are returned. Nothing is extracted to disk (D-116).

        Returns:
            The file's or the archive member's content.

        Raises:
            DatasetError: If no file is configured, the checksum differs, or the member is missing
                or not a regular file.
        """
        if self._hf_filename is None:
            raise DatasetError(f"{self._name} has no data.hf_filename to read")
        path = Path(
            hf_hub_download(
                repo_id=self._hf_path,
                filename=self._hf_filename,
                repo_type="dataset",
                revision=self._hf_revision,
                token=False,
            )
        )
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != self._hf_file_sha256:
            raise DatasetError(
                f"{self._hf_path}/{self._hf_filename} at {self._hf_revision} has SHA-256 {digest}, "
                f"expected {self._hf_file_sha256}; refusing to load a changed file"
            )
        if self._archive_member is None:
            return path.read_bytes()
        with tarfile.open(path) as archive:
            try:
                member = archive.getmember(self._archive_member)
            except KeyError as exc:
                raise DatasetError(
                    f"{self._hf_filename} has no member {self._archive_member}"
                ) from exc
            handle = archive.extractfile(member) if member.isfile() else None
            if handle is None:
                raise DatasetError(f"{self._archive_member} is not a regular file")
            with handle:
                return handle.read()

    def _read_source_file(self, name: str) -> bytes:
        """Read one of ``data.source_files``, fetched from its address and verified (D-132).

        The file is downloaded once into ``data.cache_dir`` and its SHA-256 is checked on every
        read, so a changed or corrupted copy is refused. A copy placed in the cache by hand (a
        backup) is accepted when its checksum matches.

        Args:
            name: The file's configured name.

        Returns:
            The file's bytes.

        Raises:
            DatasetError: If the file is not configured, cannot be fetched, or its checksum differs.
        """
        # Imported here: metrics/ imports datasets/ at module level (the citation metric).
        from ..metrics.artifacts import fetch_pinned
        from ..metrics.base import MetricError

        entry = self._source_files.get(name)
        if entry is None:
            raise DatasetError(f"{self._name} needs data.source_files to include {name!r}")
        try:
            path = fetch_pinned(entry.url, entry.sha256, self._cache_dir, filename=entry.name)
        except MetricError as exc:
            raise DatasetError(f"{self._name} could not read {name!r}: {exc}") from exc
        return path.read_bytes()

    @abstractmethod
    def _map_row(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        """Map source columns onto id/source/evidence/reference/question, before stamping."""
