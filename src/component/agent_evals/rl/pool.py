"""Freezing the T5 training pool: a seeded draw of training articles whose prompt fits the cap.

The pool is drawn once and stored as an explicit ID list next to the training config, as
``<config stem>_pool.yaml``, which the config composes through its ``defaults:``. Training then
replays those IDs; it never re-filters or redraws. An article is eligible only if its T0 prompt,
chat template included, fits ``pool.prompt_cap`` tokens under every backbone's tokenizer, so
every backbone trains on the same articles (D-057). Size and seed are ``data.n_samples`` and
``data.sample_seed`` (D-057, D-058)::

    python -m agent_evals.rl.pool --config src/configs/rl/grpo_xsum.yaml --set run.owner=sameer
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from ..config import DataConfig, ExperimentConfig, LoggingConfig, PoolConfig, load_config
from ..datasets import DatasetAdapter
from ..logging_utils import configure_logging, get_logger
from ..registry import DATASETS
from ..seeding import make_sampling_rng
from .base import training_template, worked_examples_seed
from .memory import PromptLengths, load_tokenizers

__all__ = ["PoolError", "TrainingPool", "draw_pool", "main", "pool_path", "render_pool_yaml"]

logger = get_logger(__name__)

_PLACEHOLDER = (
    "# Placeholder. Replaced by `python -m agent_evals.rl.pool`.\n"
    "data:\n"
    "  sample_ids: null\n"
)


class PoolError(RuntimeError):
    """Raised for a pool that cannot be drawn, or one that is already frozen."""


def pool_path(config: Path) -> Path:
    """Where a training config's pool is frozen: next to it, as ``<stem>_pool.yaml``."""
    return config.with_name(f"{config.stem}_pool.yaml")


def draw_pool(
    ids: Sequence[str], lengths: Sequence[int], cap: int, count: int, seed: int
) -> tuple[str, ...]:
    """Draw ``count`` IDs at random from those whose prompt fits ``cap``.

    Args:
        ids: Example IDs.
        lengths: Prompt length in tokens per ID, aligned with ``ids``.
        cap: Longest prompt allowed, in tokens.
        count: How many IDs to draw.
        seed: Sampling seed, through the project's isolated sampler.

    Returns:
        The drawn IDs, in draw order. The fitting IDs are sorted first, so the draw depends on
        the seed and the data alone.

    Raises:
        PoolError: If fewer than ``count`` prompts fit.
    """
    fitting = sorted(i for i, n in zip(ids, lengths, strict=True) if n <= cap)
    if len(fitting) < count:
        raise PoolError(f"only {len(fitting)} prompts fit {cap} tokens; {count} needed")
    return tuple(make_sampling_rng(seed).sample(fitting, count))


def render_pool_yaml(
    data: DataConfig,
    pool: PoolConfig,
    ids: Sequence[str],
    eligible: int,
    fitting: int,
    drawn_on: str,
) -> str:
    """The frozen pool as a YAML fragment with its provenance header.

    Args:
        data: The data config the pool was drawn under.
        pool: The prompt-length filter applied.
        ids: The drawn IDs, in draw order.
        eligible: Articles that passed the data filters.
        fitting: Of those, articles whose prompt fits the cap.
        drawn_on: ISO date of the draw.

    Returns:
        YAML text defining ``data.sample_ids``, composed by the training config.
    """
    header = (
        "# Frozen T5 training pool -- evidence, not a parameter (D-057, D-058).\n"
        "#\n"
        "# Redrawing changes which articles every T5 fine-tune trains on. It is produced once, by\n"
        "# `python -m agent_evals.rl.pool`, and thereafter only by a superseding decision-log\n"
        "# entry.\n"
        "#\n"
        f"# adapter:          {data.name}\n"
        f"# hf_revision:      {data.hf_revision}\n"
        f"# split:            {data.split}\n"
        f"# dedupe_sources:   {data.dedupe_sources}\n"
        f"# exclude sources:  {data.exclude_sources_from_split}\n"
        f"# min_source_words: {data.min_source_words}\n"
        f"# prompt cap:       {pool.prompt_cap} tokens under {', '.join(pool.backbones)}\n"
        f"# articles:         {eligible} eligible, {fitting} within the cap, {len(ids)} drawn\n"
        f"# sample_seed:      {data.sample_seed}\n"
        f"# drawn_on:         {drawn_on}\n"
        "\n"
        "data:\n"
        "  sample_ids:\n"
    )
    return header + "".join(f'    - "{identifier}"\n' for identifier in ids)


class TrainingPool:
    """Draws and freezes the training pool of one training config.

    Args:
        config: The training config, which composes its pool file through ``defaults:``.
        repo_root: Repository root; the registry and ``.env`` resolve against it.
        overrides: ``key.path=value`` strings, as accepted by the CLI.
    """

    def __init__(self, config: Path, repo_root: Path, overrides: Sequence[str] = ()) -> None:
        self._config = config
        self._repo_root = repo_root
        self._overrides = list(overrides)

    def freeze(self, overwrite: bool = False) -> tuple[str, ...]:
        """Draw the pool and write it next to the config.

        If the pool file is absent, a placeholder is written first so the config resolves.

        Args:
            overwrite: Redraw although a pool is frozen. That changes what every fine-tune trains
                on, so it is off by default.

        Returns:
            The drawn IDs.

        Raises:
            PoolError: If the config lacks a group, a pool is already frozen, or too few fit.
            DatasetError: If the data config is not a trainable split.
        """
        path = pool_path(self._config)
        if not path.exists():
            path.write_text(_PLACEHOLDER, encoding="utf-8")
            logger.info("wrote placeholder %s so the config resolves", path)
        cfg = load_config(self._config, self._overrides)
        if cfg.data is None or cfg.pool is None:
            raise PoolError(f"{self._config} needs a data and a pool group")
        if cfg.data.sample_ids is not None and not overwrite:
            raise PoolError(
                f"{path} already freezes {len(cfg.data.sample_ids)} articles; refusing to redraw "
                "what every fine-tune trains on. Overwrite only with a superseding decision"
            )
        data = dataclasses.replace(cfg.data, sample_ids=None)
        adapter = self._build_adapter(data)
        adapter.assert_trainable(data.split)
        examples = [adapter.as_example(row) for row in adapter.eligible_pool()]
        meter = PromptLengths(
            self._load_tokenizers(cfg), training_template(cfg), worked_examples_seed(cfg)
        )
        lengths = meter.longest(examples)
        ids = [e.id for e in examples]
        drawn = draw_pool(ids, lengths, cfg.pool.prompt_cap, data.n_samples, data.sample_seed)
        fitting = sum(n <= cfg.pool.prompt_cap for n in lengths)
        path.write_text(
            render_pool_yaml(
                data, cfg.pool, drawn, len(examples), fitting, date.today().isoformat()
            ),
            encoding="utf-8",
        )
        logger.info(
            "froze %d of %d articles within %d tokens (of %d eligible) into %s",
            len(drawn),
            fitting,
            cfg.pool.prompt_cap,
            len(examples),
            path,
        )
        return drawn

    def _build_adapter(self, data: DataConfig) -> DatasetAdapter:
        """The dataset adapter for ``data``. Tests substitute an offline one."""
        return DATASETS.build(data.name, data)

    def _load_tokenizers(self, cfg: ExperimentConfig) -> dict[str, Any]:
        """The pool backbones' tokenizers at their pinned revisions. Tests use tiny ones."""
        return load_tokenizers(cfg, cfg.pool.backbones, self._repo_root)


def main(argv: list[str] | None, repo_root: Path) -> int:
    """Command-line wrapper for :meth:`TrainingPool.freeze`.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root.

    Returns:
        0 on success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Freeze the T5 training pool.")
    parser.add_argument("--config", required=True, type=Path, help="training config")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    configure_logging(LoggingConfig())
    try:
        TrainingPool(args.config, repo_root, args.overrides or []).freeze()
    except Exception:  # the one top-level catch (AGENTS.md §6)
        logger.exception("the training pool was not frozen")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], Path(__file__).resolve().parents[4]))
