"""Check that one backbone serves on this machine: start vLLM for it, send one request, stop.

This is a connectivity check, not a run. Nothing goes to ``results/`` and the reply is never
scored. Use it before a sweep, after changing a backbone's registry row, or after a driver or
library upgrade::

    python3 src/check_backbone.py --backbone qwen2.5-7b-instruct
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from ..config import ConfigError, LoggingConfig, SamplingConfig, ServingConfig, load_group
from ..env import EnvFileError, load_env_file
from ..logging_utils import configure_logging, get_logger
from .backbones import ModelRegistryError, load_model_registry, select_backbone
from .base import Completion, ServingError, get_client
from .launch import VLLMServer

__all__ = ["check_backbone", "main"]

logger = get_logger(__name__)


def check_backbone(
    config_path: Path, backbone_name: str, repo_root: Path, overrides: list[str] | None = None
) -> Completion:
    """Serve one backbone, send the connectivity probe, and stop the server.

    Args:
        config_path: Config holding the ``serving`` and ``sampling`` groups.
        backbone_name: Registry name of the backbone to check.
        repo_root: Repository root; the registry and ``.env`` paths resolve against it.
        overrides: ``key.path=value`` strings applied to the config.

    Returns:
        The probe's completion, with token counts, latency and finish reason.

    Raises:
        ConfigError: If the config is invalid.
        EnvFileError: If ``.env`` is malformed.
        ModelRegistryError: If the backbone is unknown or its context length is unmeasured.
        ServingError: If the server fails to start or the request fails.
    """
    serving = load_group(config_path, "serving", ServingConfig, overrides)
    sampling = load_group(config_path, "sampling", SamplingConfig, overrides)
    load_env_file(repo_root / serving.env_file)
    registry = load_model_registry(repo_root / serving.model_registry)
    backbone = select_backbone(registry, backbone_name)
    if backbone.gated and not os.environ.get(serving.token_env):
        logger.warning(
            "%s is gated and %s is not set; the download will fail unless you are logged in to "
            "HuggingFace and have accepted the model's license",
            backbone.name,
            serving.token_env,
        )
    client = get_client(serving, sampling, backbone)
    log_path = Path(tempfile.gettempdir()) / f"agent_evals_vllm_{backbone.name}.log"
    with VLLMServer(backbone, serving, log_path):
        completion = client.complete(serving.check_prompt)
    logger.info(
        "%s served OK: finish_reason=%s prompt_tokens=%s completion_tokens=%s latency_s=%.2f",
        backbone.name,
        completion.finish_reason,
        completion.token_cost.prompt_tokens,
        completion.token_cost.completion_tokens,
        completion.latency_s,
    )
    logger.info("%s replied: %r", backbone.name, completion.text)
    return completion


def main(argv: list[str] | None, repo_root: Path, default_config: Path) -> int:
    """Command-line wrapper for :func:`check_backbone`.

    Args:
        argv: Arguments, excluding the program name.
        repo_root: Repository root, against which registry and ``.env`` paths resolve.
        default_config: Config used when ``--config`` is not given.

    Returns:
        0 when the backbone served and answered, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Start vLLM for one backbone, send one request, and stop it."
    )
    parser.add_argument("--backbone", required=True, help="backbone name from model_registry.csv")
    parser.add_argument("--config", type=Path, default=default_config, help="config file to use")
    parser.add_argument("--set", dest="overrides", action="append", help="key.path=value")
    args = parser.parse_args(argv)
    try:
        configure_logging(load_group(args.config, "logging", LoggingConfig, args.overrides))
        check_backbone(args.config, args.backbone, repo_root, args.overrides)
    except (ConfigError, EnvFileError, ModelRegistryError, ServingError) as exc:
        logger.error("backbone check failed: %s", exc)
        return 1
    return 0
