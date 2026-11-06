"""The dataset group: pinned source and frozen evaluation-sample recipe."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..seeding import MAX_SEED
from .validation import ConfigError

_SHA256 = r"[0-9a-f]{64}"


@dataclass(frozen=True, slots=True)
class SourceFile:
    """One data file fetched from its original address and pinned by checksum (D-132).

    Attributes:
        name: The file's name in the local cache, and the name the adapter reads it by.
        url: Where the file is published.
        sha256: Its SHA-256, checked on every load; 64 lowercase hex characters.
    """

    name: str
    url: str
    sha256: str

    def __post_init__(self) -> None:
        for field_name in ("name", "url"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"data.source_files.{field_name} must be a non-empty string")
        if "/" in self.name:
            raise ConfigError("data.source_files.name must be a file name, not a path")
        if not isinstance(self.sha256, str) or not re.fullmatch(_SHA256, self.sha256):
            raise ConfigError("data.source_files.sha256 must be 64 lowercase hex characters")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Pinned dataset and frozen evaluation-sample recipe; see decisions D-002, D-005, D-006.

    ``sample_ids=None`` permits explicit sample preparation only. Evaluation must replay
    frozen IDs. Tier and task are adapter properties and cannot be supplied as configuration.
    """

    name: str = "tier1_xsum"  # Registered adapter name; non-empty.
    # Public dataset repository and its immutable commit SHA. Both None when the data comes from
    # source_files instead of the Hub.
    hf_path: str | None = "EdinburghNLP/xsum"
    hf_revision: str | None = "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9"
    split: str = "test"  # Raw split to load; non-empty, with training separately gated.
    hf_config: str | None = None  # Hub dataset config name; None or non-empty (D-035).
    # A dataset shipped as one file in the Hub repository rather than as a loadable split (D-116).
    # None for every Hub-split dataset; when set, hf_file_sha256 is required.
    hf_filename: str | None = None  # File in the dataset repository at hf_revision.
    hf_file_sha256: str | None = None  # Its SHA-256, checked on every load; 64 hex characters.
    archive_member: str | None = None  # Member to read when hf_filename is a tar archive.
    n_passages: int | None = None  # Retrieved passages given per question (D-117); None or >= 1.
    # A dataset assembled from files published outside the Hub, each fetched from its original
    # address and checked against its SHA-256 (D-132). None for every Hub dataset; when set,
    # hf_path, hf_revision and hf_filename are None.
    source_files: tuple[SourceFile, ...] | None = None
    cache_dir: str = "~/.cache/agent_evals/data"  # Where source_files are cached; ~ expanded.
    # IDs dropped from the pool before any filter, e.g. questions shown as worked examples
    # (D-132). Each must exist in the raw data. None drops nothing.
    exclude_ids: tuple[str, ...] | None = None
    dedupe_sources: bool = False  # Keep only the lowest id among identical sources (D-035).
    # Drop rows whose source text also appears in this split, so a training article is never an
    # evaluation article (D-055). None, or a split other than `split`.
    exclude_sources_from_split: str | None = None
    n_samples: int = 500  # Frozen evaluation size; >= n_strata.
    n_strata: int = 3  # Within-dataset length quantiles; >= 1.
    # whitespace_words: length quantiles (D-002). sub_questions: ASQA's number of sub-questions,
    # grouped as listed in `strata` (D-122).
    stratify_by: str = "whitespace_words"
    # One stratum per group of sub-question counts, in order; required for sub_questions only.
    strata: tuple[tuple[int, ...], ...] | None = None
    sample_seed: int = 123  # Isolated sampling seed; 0 through MAX_SEED, inclusive.
    min_source_words: int = 20  # Inclusive source-length floor; >= 0.
    max_source_words: int | None = None  # Inclusive ceiling; None or >= min_source_words.
    # Outputs cite passages as [1][2]; metrics other than citation metrics score them with the
    # markers removed (D-126). False for every Tier 1 dataset.
    strip_citations: bool = False
    sample_ids: tuple[str, ...] | None = None  # Ordered unique IDs; None before preparation.

    def __post_init__(self) -> None:
        for name in ("name", "split", "cache_dir"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"data.{name} must be a non-empty string")
        self._validate_source()
        if self.hf_config is not None and (
            not isinstance(self.hf_config, str) or not self.hf_config.strip()
        ):
            raise ConfigError("data.hf_config must be None or a non-empty string")
        if type(self.dedupe_sources) is not bool:
            raise ConfigError("data.dedupe_sources must be a boolean")
        if type(self.strip_citations) is not bool:
            raise ConfigError("data.strip_citations must be a boolean")
        excluded = self.exclude_sources_from_split
        if excluded is not None and (
            not isinstance(excluded, str) or not excluded.strip() or excluded == self.split
        ):
            raise ConfigError(
                "data.exclude_sources_from_split must be None or a split other than data.split"
            )
        for name in ("n_samples", "n_strata", "sample_seed", "min_source_words"):
            if type(getattr(self, name)) is not int:
                raise ConfigError(f"data.{name} must be an integer")
        if not 1 <= self.n_strata <= self.n_samples:
            raise ConfigError("data requires n_samples >= n_strata >= 1")
        if not 0 <= self.sample_seed <= MAX_SEED:
            raise ConfigError(f"data.sample_seed must be in [0, {MAX_SEED}]")
        if self.min_source_words < 0:
            raise ConfigError("data.min_source_words must be non-negative")
        if self.max_source_words is not None and (
            type(self.max_source_words) is not int or self.max_source_words < self.min_source_words
        ):
            raise ConfigError(
                "data.max_source_words must be None or an integer >= min_source_words"
            )
        self._validate_pinned_file()
        self._validate_strata()
        self._validate_sample_ids()

    def _validate_source(self) -> None:
        if self.source_files is not None:
            if not isinstance(self.source_files, tuple) or not self.source_files:
                raise ConfigError("data.source_files must be None or a non-empty list")
            names = [entry.name for entry in self.source_files]
            if len(set(names)) != len(names):
                raise ConfigError("data.source_files names must be unique")
            for name in ("hf_path", "hf_revision", "hf_filename"):
                if getattr(self, name) is not None:
                    raise ConfigError(f"data.{name} must be None when data.source_files is set")
            return
        if not isinstance(self.hf_path, str) or not self.hf_path.strip():
            raise ConfigError("data.hf_path must be a non-empty string")
        if not isinstance(self.hf_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.hf_revision
        ):
            raise ConfigError("data.hf_revision must be an immutable 40-character commit SHA")

    def _validate_pinned_file(self) -> None:
        for name in ("hf_filename", "archive_member"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ConfigError(f"data.{name} must be None or a non-empty string")
        if (self.hf_filename is None) != (self.hf_file_sha256 is None):
            raise ConfigError("data.hf_filename and data.hf_file_sha256 must be set together")
        if self.hf_file_sha256 is not None and (
            not isinstance(self.hf_file_sha256, str)
            or not re.fullmatch(_SHA256, self.hf_file_sha256)
        ):
            raise ConfigError("data.hf_file_sha256 must be 64 lowercase hex characters")
        if self.archive_member is not None and self.hf_filename is None:
            raise ConfigError("data.archive_member requires data.hf_filename")
        if self.n_passages is not None and (
            type(self.n_passages) is not int or self.n_passages < 1
        ):
            raise ConfigError("data.n_passages must be None or an integer >= 1")

    def _validate_strata(self) -> None:
        if self.stratify_by == "whitespace_words":
            if self.strata is not None:
                raise ConfigError("data.strata applies to stratify_by=sub_questions only")
            return
        if self.stratify_by != "sub_questions":
            raise ConfigError("data.stratify_by must be whitespace_words or sub_questions")
        groups = self.strata
        if not isinstance(groups, tuple) or not groups:
            raise ConfigError("data.strata must list the groups for stratify_by=sub_questions")
        values = [value for group in groups for value in group]
        if any(not group for group in groups) or any(
            type(value) is not int or value < 1 for value in values
        ):
            raise ConfigError("data.strata groups must be non-empty lists of integers >= 1")
        if len(set(values)) != len(values):
            raise ConfigError("data.strata must not repeat a value across groups")
        if len(groups) != self.n_strata:
            raise ConfigError("data.strata must have exactly data.n_strata groups")

    def _validate_sample_ids(self) -> None:
        excluded = self.exclude_ids
        if excluded is not None and (
            not isinstance(excluded, tuple)
            or not excluded
            or any(not isinstance(value, str) or not value.strip() for value in excluded)
            or len(set(excluded)) != len(excluded)
        ):
            raise ConfigError("data.exclude_ids must be None or a list of unique non-empty strings")
        if excluded is not None and self.sample_ids is not None:
            overlap = set(excluded) & set(self.sample_ids)
            if overlap:
                raise ConfigError(f"data.sample_ids contains excluded IDs: {sorted(overlap)}")
        if self.sample_ids is None:
            return
        if not isinstance(self.sample_ids, tuple) or any(
            not isinstance(value, str) or not value.strip() for value in self.sample_ids
        ):
            raise ConfigError("data.sample_ids must be a tuple of non-empty strings")
        if len(self.sample_ids) != self.n_samples:
            raise ConfigError("data.sample_ids length must equal data.n_samples")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ConfigError("data.sample_ids must be unique")
