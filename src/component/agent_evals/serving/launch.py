"""Start and stop a vLLM server for one backbone, built from its registry row and config."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import IO, ClassVar
from urllib.parse import urlsplit

from ..config import ServingConfig
from ..logging_utils import get_logger
from .backbones import BackboneSpec
from .base import LORA_NAME_SUFFIX, ServingError, endpoint_url

__all__ = ["VLLMServer", "build_serve_command", "server_environment"]

logger = get_logger(__name__)


def build_serve_command(
    backbone: BackboneSpec,
    url: str,
    adapter: str | None = None,
    max_lora_rank: int = 16,
    gpu_memory_utilization: float | None = None,
) -> list[str]:
    """The command that serves one backbone with the current interpreter's vLLM.

    Only what the project decided is passed: the pinned revision, the quantization, the measured
    context length and, for T5, its LoRA adapter (D-111). Sampling defaults, GPU memory
    utilization and the server seed are left to vLLM and the model's own files, so none of them
    appears here. Secrets never appear either --
    process arguments are visible to every user of the machine -- and reach the server only
    through the environment it inherits.

    Args:
        backbone: The model to serve.
        url: Endpoint the server must listen on, of the form ``http://host:port``.
        adapter: A LoRA adapter directory to serve with the backbone (T5, D-111), or None to
            serve the base model with exactly the command used before T5.
        max_lora_rank: Largest LoRA rank the server accepts; used only with ``adapter``.
        gpu_memory_utilization: Share of GPU memory vLLM may reserve (D-112), or None to omit
            the flag and keep vLLM's default.

    Returns:
        The argument list for ``subprocess``.

    Raises:
        ServingError: If the URL lacks an explicit host and port.
        ModelRegistryError: If the backbone's context length has not been measured.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ServingError(f"endpoint {url!r} is not a valid URL") from exc
    if parts.scheme != "http" or not parts.hostname or port is None:
        raise ServingError(f"endpoint must look like http://host:port, got {url!r}")
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        backbone.hf_path,
        "--revision",
        backbone.hf_revision,
        "--quantization",
        backbone.quantization,
        "--max-model-len",
        str(backbone.require_max_model_len()),
        "--host",
        parts.hostname,
        "--port",
        str(port),
    ]
    if gpu_memory_utilization is not None:
        command += ["--gpu-memory-utilization", str(gpu_memory_utilization)]
    if adapter is not None:
        command += [
            "--enable-lora",
            "--max-lora-rank",
            str(max_lora_rank),
            "--lora-modules",
            f"{backbone.name}{LORA_NAME_SUFFIX}={Path(adapter).expanduser()}",
        ]
    return command


def server_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment a vLLM server process runs with.

    The current interpreter's ``bin`` directory goes first on ``PATH``. vLLM compiles kernels at
    startup with tools installed into that directory -- FlashInfer runs ``ninja`` by name -- so
    without this the server fails whenever the virtual environment was not activated in the
    calling shell. Every other variable, including secrets such as the access token, is inherited
    unchanged and never placed on the command line.

    Args:
        base: Environment to start from. The current process environment when None.

    Returns:
        A new mapping for ``subprocess``; ``base`` is not modified.
    """
    env = dict(os.environ if base is None else base)
    bin_dir = str(Path(sys.executable).parent)
    rest = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry and entry != bin_dir]
    env["PATH"] = os.pathsep.join([bin_dir, *rest])
    return env


class VLLMServer:
    """A vLLM server process for one backbone, managed as a context manager.

    Entering starts the server and blocks until it reports healthy. Leaving always stops it, so the
    GPU is released even when the work inside the block fails.

    Args:
        backbone: The model to serve.
        serving: Endpoint variable name and startup timeout.
        log_path: File that receives the server's output.
    """

    #: Seconds between health checks while waiting for startup. Affects waiting only, not results.
    poll_interval_s: ClassVar[float] = 5.0
    #: Seconds allowed for a graceful shutdown before the process is killed.
    stop_grace_s: ClassVar[float] = 60.0

    def __init__(self, backbone: BackboneSpec, serving: ServingConfig, log_path: Path) -> None:
        self._backbone = backbone
        self._adapter = serving.adapter_for(backbone.name)
        self._max_lora_rank = serving.max_lora_rank
        self._gpu_memory_utilization = serving.gpu_memory_utilization
        self._endpoint_env = serving.endpoint_env
        self._startup_timeout_s = serving.startup_timeout_s
        self._log_path = log_path
        self._process: subprocess.Popen[bytes] | None = None
        self._log: IO[bytes] | None = None

    def __enter__(self) -> VLLMServer:
        url = endpoint_url(self._endpoint_env)
        command = build_serve_command(
            self._backbone, url, self._adapter, self._max_lora_rank, self._gpu_memory_utilization
        )
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._log_path.open("wb")
        logger.info(
            "starting vLLM for %s; server output in %s", self._backbone.name, self._log_path
        )
        self._process = self._spawn(command, self._log)
        try:
            self._wait_until_healthy(url)
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        self.stop()
        return False

    def stop(self) -> None:
        """Stop the server and every process it started, and close its log.

        vLLM runs its engine in child processes that can outlive the API server -- an interrupted
        run once left one holding 22 GiB of GPU memory -- so the signal goes to the server's whole
        process group, and anything still in the group once the server has exited is killed.
        Safe to call more than once.
        """
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                self._signal_group(process, signal.SIGTERM)
                try:
                    process.wait(timeout=self.stop_grace_s)
                except subprocess.TimeoutExpired:
                    logger.warning("vLLM did not stop within %s s; killing it", self.stop_grace_s)
                    self._signal_group(process, signal.SIGKILL)
                    process.wait()
            if self._signal_group(process, signal.SIGKILL):
                logger.warning("killed vLLM child processes left running after the server exited")
        if self._log is not None:
            self._log.close()
            self._log = None

    def _wait_until_healthy(self, url: str) -> None:
        process = self._process
        if process is None:
            raise ServingError("vLLM server was not started")
        started = time.monotonic()
        while time.monotonic() - started < self._startup_timeout_s:
            code = process.poll()
            if code is not None:
                raise ServingError(
                    f"vLLM exited with code {code} before becoming ready; see {self._log_path}"
                )
            if self._health_ok(url):
                elapsed = time.monotonic() - started
                logger.info("vLLM ready for %s after %.0f s", self._backbone.name, elapsed)
                return
            time.sleep(self.poll_interval_s)
        raise ServingError(
            f"vLLM was not ready within {self._startup_timeout_s} s; see {self._log_path}"
        )

    def _spawn(self, command: list[str], log: IO[bytes]) -> subprocess.Popen[bytes]:
        # A new session makes the server the leader of its own process group, so stop() can
        # reach the engine processes it forks.
        return subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=server_environment(),
            start_new_session=True,
        )

    def _signal_group(self, process: subprocess.Popen[bytes], sig: signal.Signals) -> bool:
        """Send ``sig`` to the server's process group.

        Returns:
            True if any process in the group received it, False if the group is already gone.
        """
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return False
        return True

    def _health_ok(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=self.poll_interval_s) as handle:
                return handle.status == 200
        except OSError:
            return False
