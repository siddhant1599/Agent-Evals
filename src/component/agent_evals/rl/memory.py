"""The GRPO memory measurement (Q-004): which group sizes and prompt lengths fit the A10G.

Two steps, both started by a human:

1. ``select`` (CPU): for each prompt cap, find the longest eligible training articles whose T0
   prompt fits the cap under every backbone's tokenizer, and freeze them as a config next to the
   base config, ``<base>_<cap>.yaml``. These are the worst case for memory at that cap.
2. ``sweep`` (GPU, AGENTS.md §12.4): one short training run per backbone, cap and group size,
   each in its own process so that its peak VRAM is its own. A cell that runs out of memory
   skips the larger group sizes for that backbone and cap; any other failure stops the sweep.

A second pair of steps times a typical step rather than the worst one: ``select-typical`` freezes
the articles whose prompt is closest to the median of the training pool, and ``sweep-typical``
runs one cell per backbone on them at the training group size. ``sweep-forced`` reruns the
longest articles of one cap with every completion forced to the completion cap, the most memory
a training step can need.

Each cell writes an ordinary ``results/<run_id>/``: peak VRAM in ``summary.json`` and seconds
per step in ``metrics.csv``. Usage::

    python -m agent_evals.rl.memory select \\
        --config src/configs/rl/grpo_xsum_memory.yaml --set run.owner=sameer
    python -m agent_evals.rl.memory sweep \\
        --config src/configs/rl/grpo_xsum_memory.yaml --set run.owner=sameer
    python -m agent_evals.rl.memory select-typical \\
        --config src/configs/rl/grpo_xsum_memory.yaml --set run.owner=sameer
    python -m agent_evals.rl.memory sweep-typical \\
        --config src/configs/rl/grpo_xsum_memory.yaml --set run.owner=sameer
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ..agents.generator import InitialPromptMixin
from ..config import DataConfig, ExperimentConfig, LoggingConfig, load_config
from ..datasets import DatasetAdapter
from ..env import load_env_file
from ..logging_utils import configure_logging, get_logger
from ..registry import DATASETS
from ..run import STATUS_OOM, STATUS_SUCCESS
from ..serving import load_model_registry, select_backbone
from ..types import Example
from .base import training_template, worked_examples_seed

__all__ = [
    "CellResult",
    "MeasurementArticles",
    "MeasurementError",
    "MemorySweep",
    "PromptLengths",
    "cap_config_path",
    "closest_to_median",
    "load_tokenizers",
    "longest_under_cap",
    "main",
    "render_cap_config",
    "render_typical_config",
    "typical_config_path",
]

logger = get_logger(__name__)

#: Articles rendered and tokenized per batch, to bound the memory of the selection step.
_CHUNK = 10_000


class MeasurementError(RuntimeError):
    """Raised for a measurement that cannot be set up, or a cell that fails for a reason other
    than running out of memory."""


def cap_config_path(base_config: Path, cap: int) -> Path:
    """Where the article list for ``cap`` is frozen: next to the base config, ``<stem>_<cap>``."""
    return base_config.with_name(f"{base_config.stem}_{cap}.yaml")


def typical_config_path(base_config: Path) -> Path:
    """Where the typical-length article list is frozen: next to the base config."""
    return base_config.with_name(f"{base_config.stem}_typical.yaml")


def longest_under_cap(
    ids: Sequence[str], lengths: Sequence[int], cap: int, count: int
) -> tuple[str, ...]:
    """The ``count`` IDs with the longest prompts not over ``cap``.

    Args:
        ids: Example IDs.
        lengths: Prompt length in tokens per ID, aligned with ``ids``.
        cap: Longest prompt allowed, in tokens.
        count: How many IDs to return.

    Returns:
        IDs ordered longest first; equal lengths are ordered by ID, so the result is
        deterministic.

    Raises:
        MeasurementError: If fewer than ``count`` prompts fit the cap.
    """
    fitting = sorted(
        ((length, identifier) for identifier, length in zip(ids, lengths, strict=True)),
        key=lambda pair: (-pair[0], pair[1]),
    )
    fitting = [(length, identifier) for length, identifier in fitting if length <= cap]
    if len(fitting) < count:
        raise MeasurementError(f"only {len(fitting)} prompts fit {cap} tokens; {count} needed")
    return tuple(identifier for _, identifier in fitting[:count])


def closest_to_median(
    ids: Sequence[str], lengths: Sequence[int], cap: int, count: int
) -> tuple[tuple[str, ...], int]:
    """The ``count`` IDs whose prompts are closest to the median prompt not over ``cap``.

    Args:
        ids: Example IDs.
        lengths: Prompt length in tokens per ID, aligned with ``ids``.
        cap: Longest prompt in the pool the median is taken over, in tokens.
        count: How many IDs to return.

    Returns:
        The IDs, ordered by distance from the median, then length, then ID, so the result is
        deterministic; and the median, the lower one for an even count.

    Raises:
        MeasurementError: If fewer than ``count`` prompts fit the cap.
    """
    fitting = [(n, identifier) for identifier, n in zip(ids, lengths, strict=True) if n <= cap]
    if len(fitting) < count:
        raise MeasurementError(f"only {len(fitting)} prompts fit {cap} tokens; {count} needed")
    median = statistics.median_low(n for n, _ in fitting)
    ranked = sorted(fitting, key=lambda pair: (abs(pair[0] - median), pair[0], pair[1]))
    return tuple(identifier for _, identifier in ranked[:count]), median


class PromptLengths:
    """Training prompt length in tokens, chat template included, for several backbones.

    The prompt is every message of the first generation call: any worked examples as earlier
    turns, then the item (D-118, D-136). A zero-shot template sends one user message.

    Args:
        tokenizers: Backbone name to tokenizer.
        template: The Generator template whose prompt is measured.
        seed: Seed a few-shot template draws its worked examples from; ignored otherwise.
    """

    def __init__(
        self, tokenizers: Mapping[str, Any], template: InitialPromptMixin, seed: int = 0
    ) -> None:
        if not tokenizers:
            raise MeasurementError("at least one tokenizer is needed")
        self._tokenizers = dict(tokenizers)
        self._template = template
        self._seed = seed

    def longest(self, examples: Sequence[Example]) -> list[int]:
        """Each example's longest prompt over the backbones.

        Args:
            examples: Examples to measure.

        Returns:
            One length per example, in tokens.
        """
        per_backbone = [self.lengths(name, examples) for name in self._tokenizers]
        return [max(values) for values in zip(*per_backbone, strict=True)]

    def lengths(self, backbone: str, examples: Sequence[Example]) -> list[int]:
        """Each example's prompt length for one backbone, as that backbone's template renders it.

        Args:
            backbone: A name given at construction.
            examples: Examples to measure.

        Returns:
            One length per example, in tokens.
        """
        tokenizer = self._tokenizers[backbone]
        lengths: list[int] = []
        for start in range(0, len(examples), _CHUNK):
            rendered = [
                tokenizer.apply_chat_template(
                    self._template.messages(example, self._seed),
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for example in examples[start : start + _CHUNK]
            ]
            encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
            lengths.extend(len(ids) for ids in encoded)
        return lengths


def _article_config(
    base_name: str,
    suffix: str,
    intro: str,
    data: DataConfig,
    ids: Sequence[str],
    lengths: Sequence[int],
    selected_on: str,
) -> str:
    """A frozen article list as a config composing the base measurement config."""
    header = (
        intro + "#\n"
        f"# adapter:          {data.name}\n"
        f"# hf_revision:      {data.hf_revision}\n"
        f"# split:            {data.split}\n"
        f"# dedupe_sources:   {data.dedupe_sources}\n"
        f"# exclude sources:  {data.exclude_sources_from_split}\n"
        f"# min_source_words: {data.min_source_words}\n"
        f"# prompt tokens:    {min(lengths)} to {max(lengths)} (longest over backbones)\n"
        f"# selected_on:      {selected_on}\n"
        "\n"
        f"defaults: [{base_name}]\n"
        f"name: {base_name}_{suffix}\n"
        "\n"
        "data:\n"
        "  sample_ids:\n"
    )
    return header + "".join(f'    - "{identifier}"\n' for identifier in ids)


def load_tokenizers(
    cfg: ExperimentConfig, backbones: Sequence[str], repo_root: Path
) -> dict[str, Any]:
    """Each named backbone's tokenizer, at the revision ``model_registry.csv`` pins.

    Args:
        cfg: A resolved config; supplies the registry path and the ``.env`` file.
        backbones: Registry names.
        repo_root: Repository root the registry and ``.env`` resolve against.

    Returns:
        Backbone name to tokenizer.
    """
    from transformers import AutoTokenizer

    load_env_file(repo_root / cfg.serving.env_file)
    registry = load_model_registry(repo_root / cfg.serving.model_registry)
    tokenizers = {}
    for name in backbones:
        spec = select_backbone(registry, name)
        tokenizers[name] = AutoTokenizer.from_pretrained(spec.hf_path, revision=spec.hf_revision)
    return tokenizers


def render_cap_config(
    base_name: str,
    cap: int,
    data: DataConfig,
    ids: Sequence[str],
    lengths: Sequence[int],
    selected_on: str,
) -> str:
    """The frozen article list for one cap, as a config composing the base measurement config.

    Args:
        base_name: The base config's file stem, which this file composes.
        cap: The prompt cap, in tokens.
        data: The data config the articles were selected under.
        ids: Selected IDs, longest first.
        lengths: Their longest prompt over the backbones, aligned with ``ids``.
        selected_on: ISO date of the selection.

    Returns:
        YAML text.
    """
    intro = (
        f"# Measurement articles for a {cap}-token prompt cap -- evidence, not a parameter "
        "(D-055).\n"
        "#\n"
        f"# The {len(ids)} longest eligible training articles whose T0 prompt, chat template\n"
        f"# included, fits {cap} tokens under every measured backbone's tokenizer: the worst case\n"
        "# for memory at this cap. Produced by `python -m agent_evals.rl.memory select`.\n"
    )
    return _article_config(base_name, str(cap), intro, data, ids, lengths, selected_on)


def render_typical_config(
    base_name: str,
    within_cap: int,
    median: int,
    data: DataConfig,
    ids: Sequence[str],
    lengths: Sequence[int],
    selected_on: str,
) -> str:
    """The frozen typical-length article list, as a config composing the base measurement config.

    Args:
        base_name: The base config's file stem, which this file composes.
        within_cap: The cap of the pool the median is taken over, in tokens.
        median: The median prompt length of that pool, in tokens.
        data: The data config the articles were selected under.
        ids: Selected IDs, closest to the median first.
        lengths: Their longest prompt over the backbones, aligned with ``ids``.
        selected_on: ISO date of the selection.

    Returns:
        YAML text.
    """
    intro = (
        "# Typical-length measurement articles -- evidence, not a parameter (D-056).\n"
        "#\n"
        f"# The {len(ids)} eligible training articles whose T0 prompt, chat template included, is\n"
        f"# closest to the median prompt ({median} tokens, longest over backbones) among prompts\n"
        f"# within {within_cap} tokens: a typical training step, for timing. Produced by\n"
        "# `python -m agent_evals.rl.memory select-typical`.\n"
    )
    return _article_config(base_name, "typical", intro, data, ids, lengths, selected_on)


def _require_groups(cfg: ExperimentConfig) -> None:
    missing = [name for name in ("data", "training", "measurement") if getattr(cfg, name) is None]
    if missing:
        raise MeasurementError(f"the measurement needs the config group(s) {missing}")


class MeasurementArticles:
    """Selects and freezes the measurement articles for every prompt cap.

    Args:
        cfg: The resolved base measurement config.
        base_config: Its path; the per-cap files are written next to it.
        repo_root: Repository root; the registry and ``.env`` resolve against it.

    Raises:
        MeasurementError: If a config group is missing, or the base config already freezes IDs.
    """

    def __init__(self, cfg: ExperimentConfig, base_config: Path, repo_root: Path) -> None:
        _require_groups(cfg)
        if cfg.data.sample_ids is not None:
            raise MeasurementError("select from the base config, which freezes no articles")
        if cfg.training.max_steps > cfg.data.n_samples:
            raise MeasurementError(
                f"data.n_samples ({cfg.data.n_samples}) must cover training.max_steps "
                f"({cfg.training.max_steps}) prompts, so no article repeats"
            )
        self._cfg = cfg
        self._base_config = base_config
        self._repo_root = repo_root

    def select(self) -> list[Path]:
        """Measure every eligible training prompt and freeze the longest that fit each cap.

        Returns:
            The files written, one per cap.

        Raises:
            DatasetError: If the data config is not a trainable split.
            MeasurementError: If a cap cannot supply enough articles.
        """
        cfg = self._cfg
        by_id = self._measure()
        written = []
        for cap in cfg.measurement.prompt_caps:
            ids = longest_under_cap(list(by_id), list(by_id.values()), cap, cfg.data.n_samples)
            path = cap_config_path(self._base_config, cap)
            path.write_text(
                render_cap_config(
                    self._base_config.stem,
                    cap,
                    cfg.data,
                    ids,
                    [by_id[i] for i in ids],
                    date.today().isoformat(),
                ),
                encoding="utf-8",
            )
            logger.info("froze %d articles for the %d-token cap in %s", len(ids), cap, path)
            written.append(path)
        return written

    def select_typical(self) -> Path:
        """Measure every eligible training prompt and freeze those closest to the median.

        The median is taken over the prompts within ``measurement.typical_within_cap``, the
        pool training draws from.

        Returns:
            The file written.

        Raises:
            DatasetError: If the data config is not a trainable split.
            MeasurementError: If the timing cells are not configured, or the pool is too small.
        """
        cfg = self._cfg
        within_cap = cfg.measurement.typical_within_cap
        if within_cap is None:
            raise MeasurementError("set measurement.typical_within_cap to select typical articles")
        by_id = self._measure()
        ids, median = closest_to_median(
            list(by_id), list(by_id.values()), within_cap, cfg.data.n_samples
        )
        path = typical_config_path(self._base_config)
        path.write_text(
            render_typical_config(
                self._base_config.stem,
                within_cap,
                median,
                cfg.data,
                ids,
                [by_id[i] for i in ids],
                date.today().isoformat(),
            ),
            encoding="utf-8",
        )
        logger.info("froze %d articles around the %d-token median in %s", len(ids), median, path)
        return path

    def _measure(self) -> dict[str, int]:
        """Each eligible training example's longest prompt over the backbones, by ID."""
        cfg = self._cfg
        adapter = self._build_adapter()
        adapter.assert_trainable(cfg.data.split)
        examples = [adapter.as_example(row) for row in adapter.eligible_pool()]
        meter = PromptLengths(
            self._load_tokenizers(), training_template(cfg), worked_examples_seed(cfg)
        )
        return dict(zip((e.id for e in examples), meter.longest(examples), strict=True))

    def _build_adapter(self) -> DatasetAdapter:
        """The dataset adapter for ``cfg.data``. Tests substitute an offline one."""
        return DATASETS.build(self._cfg.data.name, self._cfg.data)

    def _load_tokenizers(self) -> dict[str, Any]:
        """Each measured backbone's tokenizer at its pinned revision. Tests use tiny ones."""
        return load_tokenizers(self._cfg, self._cfg.measurement.backbones, self._repo_root)


@dataclass(frozen=True, slots=True)
class CellResult:
    """One measurement cell's outcome.

    Attributes:
        backbone: Registry name.
        prompt_cap: Prompt cap in tokens.
        group_size: Group size G.
        status: The run's ``summary.json`` status: success, oom or failed.
        run_dir: The run directory the cell wrote.
    """

    backbone: str
    prompt_cap: int
    group_size: int
    status: str
    run_dir: Path


class MemorySweep:
    """Runs every measurement cell, each as its own training process.

    Args:
        cfg: The resolved base measurement config.
        base_config: Its path; each cap's frozen article list sits next to it.
        repo_root: Repository root; ``results/`` and ``src/run_training.py`` resolve against it.
        overrides: ``key.path=value`` strings passed on to every cell, such as ``run.owner``.

    Raises:
        MeasurementError: If a config group is missing.
    """

    def __init__(
        self,
        cfg: ExperimentConfig,
        base_config: Path,
        repo_root: Path,
        overrides: Sequence[str] = (),
    ) -> None:
        _require_groups(cfg)
        self._cfg = cfg
        self._base_config = base_config
        self._repo_root = repo_root
        self._overrides = tuple(overrides)

    def run(self) -> list[CellResult]:
        """Run the grid: backbones in order, caps in order, group sizes ascending.

        Returns:
            Every cell run, in order.

        Raises:
            MeasurementError: If a cap's article list is missing, or a cell fails for a reason
                other than running out of memory.
        """
        grid = self._cfg.measurement
        configs = {cap: cap_config_path(self._base_config, cap) for cap in grid.prompt_caps}
        missing = [str(path) for path in configs.values() if not path.is_file()]
        if missing:
            raise MeasurementError(f"run `select` first; missing {missing}")
        results: list[CellResult] = []
        for backbone in grid.backbones:
            for cap, config in configs.items():
                for group_size in grid.group_sizes:
                    status, run_dir = self._run_cell(config, backbone, group_size)
                    result = CellResult(backbone, cap, group_size, status, run_dir)
                    results.append(result)
                    cell = dataclasses.astuple(result)[:3]
                    logger.info("cell %s: %s (%s)", cell, status, run_dir)
                    if status == STATUS_OOM:
                        logger.info("out of memory; skipping larger G for %s at %d", backbone, cap)
                        break
                    if status != STATUS_SUCCESS:
                        raise MeasurementError(
                            f"cell failed without running out of memory: {run_dir}"
                        )
        return results

    def run_typical(self) -> list[CellResult]:
        """Run one timing cell per backbone on the typical-length articles.

        Returns:
            Every cell run, in backbone order.

        Raises:
            MeasurementError: If the timing cells are not configured, their article list is
                missing, or a cell does not succeed.
        """
        grid = self._cfg.measurement
        if grid.typical_group_size is None:
            raise MeasurementError("set measurement.typical_group_size to run timing cells")
        config = typical_config_path(self._base_config)
        if not config.is_file():
            raise MeasurementError(f"run `select-typical` first; missing {config}")
        results: list[CellResult] = []
        for backbone in grid.backbones:
            status, run_dir = self._run_cell(config, backbone, grid.typical_group_size)
            results.append(
                CellResult(
                    backbone, grid.typical_within_cap, grid.typical_group_size, status, run_dir
                )
            )
            logger.info("timing cell %s: %s (%s)", backbone, status, run_dir)
            if status != STATUS_SUCCESS:
                raise MeasurementError(f"timing cell did not succeed ({status}): {run_dir}")
        return results

    def run_forced(self) -> list[CellResult]:
        """Rerun the longest articles of ``forced_cap`` with every completion at the cap.

        An out-of-memory cell is a result here, the answer the cells exist for, so the sweep
        records it and goes on to the next backbone.

        Returns:
            Every cell run, in backbone order.

        Raises:
            MeasurementError: If the forced cells are not configured, their article list is
                missing, or a cell fails for a reason other than running out of memory.
        """
        grid = self._cfg.measurement
        if grid.forced_cap is None:
            raise MeasurementError("set measurement.forced_cap to run forced-length cells")
        config = cap_config_path(self._base_config, grid.forced_cap)
        if not config.is_file():
            raise MeasurementError(f"run `select` first; missing {config}")
        extra = (
            "training.force_full_completions=true",
            f"name={self._base_config.stem}_{grid.forced_cap}_forced",
        )
        results: list[CellResult] = []
        for backbone in grid.backbones:
            status, run_dir = self._run_cell(config, backbone, grid.typical_group_size, extra)
            results.append(
                CellResult(backbone, grid.forced_cap, grid.typical_group_size, status, run_dir)
            )
            logger.info("forced-length cell %s: %s (%s)", backbone, status, run_dir)
            if status not in (STATUS_SUCCESS, STATUS_OOM):
                raise MeasurementError(f"cell failed without running out of memory: {run_dir}")
        return results

    def _run_cell(
        self, config: Path, backbone: str, group_size: int, extra: Sequence[str] = ()
    ) -> tuple[str, Path]:
        """Run one cell in a fresh process and return its status and run directory."""
        results_root = self._repo_root / self._cfg.run.output_root
        before = set(results_root.iterdir()) if results_root.is_dir() else set()
        adapter_root = Path(tempfile.mkdtemp(prefix="agent_evals_memory_"))
        try:
            command = self._command(config, backbone, group_size, adapter_root, extra)
            subprocess.run(command, check=False)
        finally:
            shutil.rmtree(adapter_root, ignore_errors=True)
        created = sorted(set(results_root.iterdir()) - before) if results_root.is_dir() else []
        if len(created) != 1:
            raise MeasurementError(f"expected one new run directory, found {len(created)}")
        summary = json.loads((created[0] / "summary.json").read_text(encoding="utf-8"))
        return summary["status"], created[0]

    def _command(
        self,
        config: Path,
        backbone: str,
        group_size: int,
        adapter_root: Path,
        extra: Sequence[str] = (),
    ) -> list[str]:
        """The training command for one cell: G completions per prompt, one prompt per pass."""
        command = [sys.executable, str(self._repo_root / "src" / "run_training.py")]
        command += ["--config", str(config), "--backbone", backbone]
        for override in (
            *self._overrides,
            f"training.num_generations={group_size}",
            f"training.per_device_train_batch_size={group_size}",
            f"training.adapter_root={adapter_root}",
            *extra,
        ):
            command += ["--set", override]
        return command


def main(argv: list[str] | None, repo_root: Path) -> int:
    """Command-line wrapper for the measurement steps.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root.

    Returns:
        0 on success, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="GRPO memory measurement (Q-004).")
    parser.add_argument(
        "step", choices=("select", "sweep", "select-typical", "sweep-typical", "sweep-forced")
    )
    parser.add_argument("--config", required=True, type=Path, help="base measurement config")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    overrides = args.overrides or []
    try:
        cfg = load_config(args.config, overrides)
        configure_logging(cfg.logging)
        if args.step == "select":
            MeasurementArticles(cfg, args.config, repo_root).select()
        elif args.step == "select-typical":
            MeasurementArticles(cfg, args.config, repo_root).select_typical()
        elif args.step == "sweep":
            MemorySweep(cfg, args.config, repo_root, overrides).run()
        elif args.step == "sweep-typical":
            MemorySweep(cfg, args.config, repo_root, overrides).run_typical()
        else:
            MemorySweep(cfg, args.config, repo_root, overrides).run_forced()
    except Exception:  # the one top-level catch (AGENTS.md §6)
        configure_logging(LoggingConfig())
        logger.exception("measurement step %s did not complete", args.step)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], Path(__file__).resolve().parents[4]))
