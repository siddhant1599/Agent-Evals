"""Draw the frozen evaluation sample once and record it as YAML.

The evaluation sample is stored as an explicit ID list rather than re-derived from a seed at
load time, so that an upstream change to the dataset fails loudly instead of silently selecting
different documents. See decision D-006.

This module is the only supported way to produce that list. Run it once per dataset; the
resulting file is evidence and is not regenerated casually::

    python -m agent_evals.datasets.freeze \\
        --config src/configs/data/tier1_xsum.yaml \\
        --out src/configs/data/tier1_xsum_sample.yaml \\
        --set run.owner=sameer
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import date
from pathlib import Path

from ..config import ConfigError, DataConfig, load_config
from ..logging_utils import get_logger
from ..registry import DATASETS
from .base import DatasetAdapter, DatasetError

__all__ = ["freeze_sample", "render_sample_yaml", "main"]

logger = get_logger(__name__)

_PLACEHOLDER = (
    "# Placeholder. Replaced by `python -m agent_evals.datasets.freeze`.\n"
    "data:\n"
    "  sample_ids: null\n"
)


def render_sample_yaml(cfg: DataConfig, sample_ids: tuple[str, ...], drawn_on: str) -> str:
    """Render the frozen sample as a YAML fragment with its provenance header.

    Args:
        cfg: The dataset config the sample was drawn under.
        sample_ids: Ordered IDs as returned by ``DatasetAdapter.draw_sample_ids``.
        drawn_on: ISO date the draw was performed.

    Returns:
        YAML text defining ``data.sample_ids``, safe to compose via a ``defaults:`` entry.
    """
    pinned_file = (
        f"# hf_filename:      {cfg.hf_filename}\n"
        f"# hf_file_sha256:   {cfg.hf_file_sha256}\n"
        f"# archive_member:   {cfg.archive_member}\n"
        if cfg.hf_filename is not None
        else ""
    )
    strata = f"# strata:           {cfg.strata}\n" if cfg.strata is not None else ""
    header = (
        "# Frozen evaluation sample -- evidence, not a parameter (D-002, D-006).\n"
        "#\n"
        "# Regenerating this file changes which documents every topology is evaluated on and\n"
        "# invalidates every comparison already run against it. It is produced once, by\n"
        "# `python -m agent_evals.datasets.freeze`, and thereafter only by a superseding\n"
        "# decision-log entry.\n"
        "#\n"
        f"# adapter:          {cfg.name}\n"
        f"# hf_path:          {cfg.hf_path}\n"
        f"# hf_revision:      {cfg.hf_revision}\n"
        f"# hf_config:        {cfg.hf_config}\n"
        f"{pinned_file}"
        f"# split:            {cfg.split}\n"
        f"# dedupe_sources:   {cfg.dedupe_sources}\n"
        f"# exclude sources:  {cfg.exclude_sources_from_split}\n"
        f"# n_samples:        {cfg.n_samples} in {cfg.n_strata} strata by {cfg.stratify_by}\n"
        f"{strata}"
        f"# sample_seed:      {cfg.sample_seed}\n"
        f"# min_source_words: {cfg.min_source_words}\n"
        f"# max_source_words: {cfg.max_source_words}\n"
        f"# drawn_on:         {drawn_on}\n"
        "\n"
        "data:\n"
        "  sample_ids:\n"
    )
    return header + "".join(f'    - "{identifier}"\n' for identifier in sample_ids)


def freeze_sample(
    config_path: Path,
    output_path: Path,
    overrides: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[str, ...]:
    """Draw the evaluation sample for a config and write it to ``output_path``.

    The config is expected to compose ``output_path`` through its ``defaults:`` list. If that
    file is absent -- a fresh checkout, or a deliberate redraw -- an empty placeholder is
    written first so the config still resolves. Whether a sample already exists is therefore
    decided by the resolved ``data.sample_ids``, not by the file being present.

    Args:
        config_path: Dataset config to draw under.
        output_path: Where to write the YAML fragment.
        overrides: ``key.path=value`` strings, as accepted by the CLI.
        overwrite: Redraw even though a sample is already frozen. Discards the existing
            sample and invalidates every run made against it, so it is off by default.

    Returns:
        The drawn IDs, in the order written.

    Raises:
        ConfigError: If the config has no ``data`` group.
        DatasetError: If a sample is already frozen and ``overwrite`` is not set.
    """
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(_PLACEHOLDER, encoding="utf-8")
        logger.info("wrote placeholder %s so the config resolves", output_path)

    cfg = load_config(config_path, overrides)
    if cfg.data is None:
        raise ConfigError(f"{config_path} defines no data group; nothing to freeze")
    if cfg.data.sample_ids is not None and not overwrite:
        raise DatasetError(
            f"{config_path} already resolves to a frozen sample of {len(cfg.data.sample_ids)} "
            "IDs; refusing to redraw. Redrawing changes which documents every topology is "
            "evaluated on and invalidates every comparison already run. Pass overwrite only "
            "alongside a superseding decision-log entry"
        )

    unfrozen = dataclasses.replace(cfg.data, sample_ids=None)
    adapter: DatasetAdapter = DATASETS.build(cfg.data.name, unfrozen)
    sample_ids = adapter.draw_sample_ids()
    output_path.write_text(
        render_sample_yaml(unfrozen, sample_ids, date.today().isoformat()), encoding="utf-8"
    )
    logger.info("froze %s ids for %s into %s", len(sample_ids), cfg.data.name, output_path)
    return sample_ids


def main(argv: list[str] | None = None) -> int:
    """Command-line wrapper. Parses arguments and calls :func:`freeze_sample`.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--config", type=Path, required=True, help="dataset config to draw under")
    parser.add_argument("--out", type=Path, required=True, help="YAML fragment to write")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing frozen sample file"
    )
    args = parser.parse_args(argv)
    freeze_sample(args.config, args.out, args.overrides, args.overwrite)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
