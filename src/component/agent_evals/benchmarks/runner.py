"""Run benchmark cells end to end: generate, score, and write one ``results/<run_id>/`` per cell.

A cell is one (topology, backbone, task) at the seed in ``run.seed``. Generation and scoring are
separate phases because vLLM and the scorer models do not fit on one 24 GiB GPU together: every
output is generated while the server runs, the server is stopped, and each metric then loads,
scores and is freed in turn. Rows are written to ``metrics.csv`` once every output is scored.

    python3 src/run_benchmark.py --topology t0_single_agent --backbone qwen2.5-7b-instruct \\
        --task tier1_xsum --set run.owner=sameer --set run.seed=0
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import itertools
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..agents import AgentError
from ..config import (
    AgentConfig,
    ConfigError,
    ExperimentConfig,
    LoggingConfig,
    ServingConfig,
    TopologyConfig,
    load_config,
    load_group,
)
from ..datasets import DatasetError
from ..env import EnvFileError, load_env_file
from ..logging_utils import configure_logging, get_logger
from ..metrics import MetricError, remove_citations
from ..registry import AGENTS, DATASETS, METRICS, TOPOLOGIES, RegistryError
from ..run import RunDirectory
from ..serving import (
    BackboneSpec,
    ModelRegistryError,
    ServingError,
    VLLMServer,
    get_client,
    load_model_registry,
    select_backbone,
)
from ..topologies import TopologyError
from ..types import Example, TopologyResult

__all__ = ["BenchmarkCell", "BenchmarkError", "BenchmarkRunner", "main"]

logger = get_logger(__name__)

#: Log generation progress every N examples. Affects the log only.
_PROGRESS_EVERY = 50
#: The LoRA weights file PEFT writes into every adapter directory; hashed into the manifest.
_ADAPTER_WEIGHTS = "adapter_model.safetensors"

#: Selector value meaning every available name.
_ALL = "all"


class BenchmarkError(ValueError):
    """Raised when a cell cannot run as requested."""


#: Failures that end one cell but let the remaining cells run.
_CELL_ERRORS = (
    AgentError,
    BenchmarkError,
    ConfigError,
    DatasetError,
    EnvFileError,
    MetricError,
    ModelRegistryError,
    RegistryError,
    ServingError,
    TopologyError,
)


@dataclass(frozen=True, slots=True)
class BenchmarkCell:
    """One (topology, backbone, task) combination; the seed comes from ``run.seed``.

    Attributes:
        topology: Registered topology name.
        backbone: Name from ``model_registry.csv``.
        task: Registered dataset name; its config is ``src/configs/data/<task>.yaml``.
    """

    topology: str
    backbone: str
    task: str


class BenchmarkRunner:
    """Runs cells for one resolved config.

    Args:
        cfg: The resolved experiment config, including ``data``.
        repo_root: Repository root; registry, ``.env`` and ``results/`` resolve against it.
    """

    #: Server lifecycle for generation. Tests substitute one that launches nothing.
    server_class: ClassVar[type[VLLMServer]] = VLLMServer

    def __init__(self, cfg: ExperimentConfig, repo_root: Path) -> None:
        self._cfg = cfg
        self._repo_root = repo_root

    def run(self, cell: BenchmarkCell) -> Path:
        """Generate, score and record one cell.

        Args:
            cell: What to run.

        Returns:
            The run directory written.

        Raises:
            BenchmarkError: If the config does not match the cell or the metric suite is invalid.
            RegistryError, DatasetError, ModelRegistryError, ServingError, MetricError: From the
                component that failed. The run directory still records the failure.
        """
        cfg = self._cfg
        if cfg.data is None or cfg.data.name != cell.task:
            raise BenchmarkError(f"config data group must be task {cell.task!r}")
        if cfg.topology is not None and cfg.topology.name != cell.topology:
            raise BenchmarkError(f"config topology group must be topology {cell.topology!r}")
        suite = cfg.metrics.suite
        if not suite or len(set(suite)) != len(suite):
            raise BenchmarkError(f"metrics.suite must be non-empty and unique, got {suite}")
        metric_classes = [METRICS.get(name) for name in suite]

        load_env_file(self._repo_root / cfg.serving.env_file)
        registry = load_model_registry(self._repo_root / cfg.serving.model_registry)
        backbone = select_backbone(registry, cell.backbone)
        topology_cfg = cfg.topology
        if topology_cfg is None:
            topology_cfg = TopologyConfig(name=cell.topology)
        topology = TOPOLOGIES.build(cell.topology, topology_cfg)
        lora_adapter = self._lora_adapter(topology_cfg, backbone.name)
        client = get_client(cfg.serving, cfg.sampling, backbone)
        agent_cfg = AgentConfig(dataset=cfg.data.name, seed=cfg.run.seed)
        agents = {role: AGENTS.build(role, agent_cfg, client) for role in topology.ROLES}
        adapter = DATASETS.build(cfg.data.name, cfg.data)
        examples = [adapter.as_example(row) for row in adapter.build()]

        manifest = {
            "cell": dataclasses.asdict(cell),
            "backbone": dataclasses.asdict(backbone),
            "prompt_templates": {
                role: dict(agent.prompt_templates) for role, agent in agents.items()
            },
            "metric_suite": list(suite),
            "metric_keys": [key for cls in metric_classes for key in cls.KEYS],
            "n_examples": len(examples),
            "lora_adapter": lora_adapter,
        }
        named = dataclasses.replace(cfg, name=f"{cell.topology}_{cell.task}_{cell.backbone}")
        with RunDirectory(named, self._repo_root, manifest_extra=manifest) as run:
            results = self._generate(run.run_id, backbone, topology, agents, examples)
            scores = self._score(results, examples)
            for example, result, row_scores in zip(examples, results, scores, strict=True):
                run.record_result(
                    result,
                    example_id=example.id,
                    source_words=example.source_words,
                    finish_reason=result.finish_reason,
                    hit_length_cap=result.finish_reason == "length",
                    blank_output=self._is_blank(result.output),
                    **row_scores,
                )
            run.add_summary(**self._summarize(results, scores))
        return run.path

    def _generate(
        self,
        run_id: str,
        backbone: BackboneSpec,
        topology: Any,
        agents: dict[str, Any],
        examples: list[Example],
    ) -> list[TopologyResult]:
        log_path = Path(tempfile.gettempdir()) / f"agent_evals_vllm_{run_id}.log"
        results: list[TopologyResult] = []
        with self.server_class(backbone, self._cfg.serving, log_path):
            for index, example in enumerate(examples, start=1):
                results.append(topology.run(example, agents))
                if index % _PROGRESS_EVERY == 0 or index == len(examples):
                    logger.info("generated %d / %d", index, len(examples))
        return results

    def _lora_adapter(self, topology_cfg: TopologyConfig, backbone: str) -> dict[str, str] | None:
        """The adapter served for this cell, with the hash of its weights, for the manifest.

        Args:
            topology_cfg: Says whether the topology must be served with an adapter.
            backbone: Name from ``model_registry.csv``.

        Returns:
            The adapter directory and the SHA-256 of ``adapter_model.safetensors``, or None when
            the backbone is served without an adapter.

        Raises:
            BenchmarkError: If the topology requires an adapter and config names none, or the
                named directory has no adapter weights (D-111).
        """
        path = self._cfg.serving.adapter_for(backbone)
        if path is None:
            if topology_cfg.requires_adapter:
                raise BenchmarkError(
                    f"topology {topology_cfg.name!r} requires a LoRA adapter for {backbone!r}; "
                    "add it to serving.lora_adapters"
                )
            return None
        weights = Path(path).expanduser() / _ADAPTER_WEIGHTS
        if not weights.is_file():
            raise BenchmarkError(f"no LoRA adapter weights at {weights}")
        return {"path": path, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest()}

    def _score(
        self, results: list[TopologyResult], examples: list[Example]
    ) -> list[dict[str, float | None]]:
        rows: list[dict[str, float | None]] = [{} for _ in results]
        scored = [i for i, result in enumerate(results) if not self._is_blank(result.output)]
        if len(scored) < len(results):
            logger.warning(
                "%d blank output(s) recorded as generation failures and not scored",
                len(results) - len(scored),
            )
        outputs = [results[i].output for i in scored]
        cleaned = [self._metric_text(output) for output in outputs]
        references = [examples[i].reference for i in scored]
        evidence = [examples[i].evidence for i in scored]
        gold = [examples[i].gold for i in scored]
        for name in self._cfg.metrics.suite:
            metric_class = METRICS.get(name)
            for row in rows:
                row.update(dict.fromkeys(metric_class.KEYS))
            if not scored:
                continue
            logger.info("scoring %d outputs with %s", len(scored), name)
            predictions = outputs if metric_class.READS_CITATIONS else cleaned
            metric = METRICS.build(name, self._cfg.metrics)
            for index, values in zip(
                scored, metric.score(predictions, references, evidence, gold), strict=True
            ):
                rows[index].update(values)
            del metric
            _release_gpu_memory()
        return rows

    def _metric_text(self, output: str) -> str:
        """The output as metrics other than citation metrics score it (D-126)."""
        return remove_citations(output) if self._cfg.data.strip_citations else output

    def _is_blank(self, output: str) -> bool:
        """A generation failure: nothing left once citation markers are removed, if they are."""
        return not self._metric_text(output).strip()

    def _summarize(
        self, results: list[TopologyResult], scores: list[dict[str, float | None]]
    ) -> dict[str, Any]:
        n = len(results)
        keys = list(scores[0]) if scores else []
        means = {}
        for key in keys:
            values = [row[key] for row in scores if row[key] is not None]
            means[key] = math.fsum(values) / len(values) if values else None
        output_tokens = [t for t in map(_output_tokens, results) if t is not None]
        return {
            "n_examples": n,
            "n_blank_outputs": sum(self._is_blank(r.output) for r in results),
            "length_cap_rate": sum(r.finish_reason == "length" for r in results) / n,
            # D-125: outputs are scored whole, so this shows how many hold a line break.
            "newline_rate": sum("\n" in r.output.strip() for r in results) / n,
            # D-119: completion tokens of the call that wrote each output.
            "mean_output_tokens": (
                math.fsum(output_tokens) / len(output_tokens) if output_tokens else None
            ),
            "prompt_tokens": sum(r.token_cost.prompt_tokens for r in results),
            "completion_tokens": sum(r.token_cost.completion_tokens for r in results),
            "total_tokens": sum(r.token_cost.total for r in results),
            "generation_wall_clock_s": math.fsum(r.wall_clock_s for r in results),
            "mean_example_wall_clock_s": math.fsum(r.wall_clock_s for r in results) / n,
            "metric_means": means,
        }


def _output_tokens(result: TopologyResult) -> int | None:
    """Completion tokens of the message whose text is the output; None if the trace lacks it.

    The output is one message's text verbatim (``Topology._assemble``), so the latest message with
    that text is the call that wrote it.
    """
    for message in reversed(result.trace):
        if message.content == result.output:
            return message.token_cost.completion_tokens
    return None


def _release_gpu_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _expand(value: str, available: tuple[str, ...], kind: str) -> list[str]:
    if value == _ALL:
        return list(available)
    if value not in available:
        raise BenchmarkError(f"unknown {kind} {value!r}; available: {', '.join(available)}")
    return [value]


def main(argv: list[str] | None, repo_root: Path, configs_dir: Path) -> int:
    """Command-line wrapper: expand the three selectors and run every cell in turn.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root.
        configs_dir: ``src/configs``; a task's config is ``data/<task>.yaml`` inside it, and a
            topology's parameters, if it has any, are ``topology/<topology>.yaml``.

    Returns:
        0 when every cell finished, 1 when any failed. Each failed cell still writes its run
        directory with the error recorded.
    """
    parser = argparse.ArgumentParser(description="Run benchmark cells and write results/.")
    parser.add_argument("--topology", required=True, help="topology name, or 'all'")
    parser.add_argument("--backbone", required=True, help="model_registry.csv name, or 'all'")
    parser.add_argument("--task", required=True, help="dataset name, or 'all'")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    base = configs_dir / "base.yaml"
    try:
        configure_logging(load_group(base, "logging", LoggingConfig, args.overrides))
        serving = load_group(base, "serving", ServingConfig, args.overrides)
        backbones = tuple(load_model_registry(repo_root / serving.model_registry))
        tasks = tuple(n for n in DATASETS.names() if (configs_dir / "data" / f"{n}.yaml").is_file())
        cells = [
            BenchmarkCell(topology=t, backbone=b, task=k)
            for k, t, b in itertools.product(
                _expand(args.task, tasks, "task"),
                _expand(args.topology, TOPOLOGIES.names(), "topology"),
                _expand(args.backbone, backbones, "backbone"),
            )
        ]
    except (BenchmarkError, ConfigError, ModelRegistryError) as exc:
        logger.error("benchmark not started: %s", exc)
        return 1

    failures = 0
    for cell in cells:
        try:
            topology_file = configs_dir / "topology" / f"{cell.topology}.yaml"
            cfg = load_config(
                configs_dir / "data" / f"{cell.task}.yaml",
                args.overrides,
                extra_paths=[topology_file] if topology_file.is_file() else [],
            )
            configure_logging(cfg.logging)
            path = BenchmarkRunner(cfg, repo_root).run(cell)
            configure_logging(cfg.logging)
            logger.info("cell %s finished: %s", cell, path)
        except _CELL_ERRORS as exc:
            configure_logging(load_group(base, "logging", LoggingConfig, args.overrides))
            logger.error("cell %s failed: %s", cell, exc)
            failures += 1
    return 1 if failures else 0
