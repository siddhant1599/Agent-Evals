"""The sampling and serving groups: generation limits and how backbones are reached."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ConfigError


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Generation limits shared by every topology and backbone.

    Temperature, top-p, top-k and repetition penalty are deliberately absent: every backbone
    decodes with its own shipped defaults (D-011).

    Attributes:
        max_tokens: Completion cap per model call (D-012). Every response's finish reason is
            recorded, and the share of outputs stopped by this cap is reported. Safe range: >= 1.
    """

    max_tokens: int = 256

    def __post_init__(self) -> None:
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ConfigError(
                f"sampling.max_tokens must be an integer >= 1, got {self.max_tokens!r}"
            )


def _split_adapter(entry: str) -> tuple[str, str]:
    """``backbone=directory`` as its two stripped parts."""
    name, sep, path = entry.partition("=") if isinstance(entry, str) else ("", "", "")
    if not sep or not name.strip() or not path.strip():
        raise ConfigError(
            f"serving.lora_adapters entries must look like backbone=directory, got {entry!r}"
        )
    return name.strip(), path.strip()


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """How backbones are served and reached (D-008, D-013, D-014).

    Per-backbone details -- repository, pinned revision, quantization, context length -- live in
    ``model_registry.csv``, not here. Secrets and the endpoint URL reach code only through
    environment variables; config holds their names, never their values.

    Attributes:
        client: Registered serving client name.
        model_registry: Path to ``model_registry.csv``, relative to the repository root.
        env_file: Path to the local ``.env`` file, relative to the repository root. May be absent.
        endpoint_env: Name of the environment variable holding the vLLM endpoint URL.
        token_env: Name of the environment variable holding the HuggingFace access token.
        request_timeout_s: Seconds before one model call is abandoned and recorded as failed.
            Safe range: > 0.
        startup_timeout_s: Seconds to wait for a vLLM server to report healthy. Safe range: > 0.
        check_prompt: Connectivity probe sent by ``check_backbone.py``. Never scored.
        lora_adapters: LoRA adapters to serve, as ``backbone=adapter_directory`` entries; a
            backbone with an entry is served with its adapter and every request goes to it
            (D-111). Empty for every topology except T5. ``~`` is expanded.
        max_lora_rank: Largest LoRA rank the server accepts; must be at least the adapters'
            rank. Used only when an adapter is served. Safe range: >= 1.
        gpu_memory_utilization: Share of GPU memory vLLM may reserve, or None to leave vLLM's
            own default (every run before D-112). Lower values shrink only the KV cache; with
            one request at a time the generated text is unchanged. Safe range: (0, 1].
    """

    client: str = "vllm_openai"
    model_registry: str = "src/model_registry.csv"
    env_file: str = ".env"
    endpoint_env: str = "FACTAGENT_VLLM_URL"
    token_env: str = "HF_TOKEN"
    request_timeout_s: float = 120.0
    startup_timeout_s: float = 600.0
    check_prompt: str = "Reply with one word: ready."
    lora_adapters: tuple[str, ...] = ()
    max_lora_rank: int = 16
    gpu_memory_utilization: float | None = None

    def __post_init__(self) -> None:
        text_fields = (
            "client",
            "model_registry",
            "env_file",
            "endpoint_env",
            "token_env",
            "check_prompt",
        )
        for name in text_fields:
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"serving.{name} must be a non-empty string")
        for name in ("request_timeout_s", "startup_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ConfigError(f"serving.{name} must be a positive number, got {value!r}")
        rank = self.max_lora_rank
        if type(rank) is not int or rank < 1:
            raise ConfigError(f"serving.max_lora_rank must be an integer >= 1, got {rank!r}")
        share = self.gpu_memory_utilization
        if share is not None and (
            isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 < share <= 1
        ):
            raise ConfigError(
                f"serving.gpu_memory_utilization must be in (0, 1] or None, got {share!r}"
            )
        names = [_split_adapter(entry)[0] for entry in self.lora_adapters]
        if len(set(names)) != len(names):
            raise ConfigError(f"serving.lora_adapters names a backbone twice: {names}")

    def adapter_for(self, backbone: str) -> str | None:
        """The adapter directory to serve with ``backbone``, or None to serve the base model.

        Args:
            backbone: Name from ``model_registry.csv``.

        Returns:
            The directory as written in config, or None when the backbone has no entry.
        """
        for entry in self.lora_adapters:
            name, path = _split_adapter(entry)
            if name == backbone:
                return path
        return None
