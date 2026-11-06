"""The client interface every backbone is reached through, and what a single model call returns.

No topology or agent ever branches on which backbone it is talking to: each holds a ``ChatClient``
and calls ``complete``. Every call returns token counts and latency alongside the text, because
token cost and latency are reported results, not telemetry.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import SamplingConfig, ServingConfig
from ..registry import SERVING
from ..types import TokenCost
from .backbones import BackboneSpec

__all__ = [
    "ChatClient",
    "Completion",
    "ServingError",
    "endpoint_url",
    "get_client",
    "LORA_NAME_SUFFIX",
    "served_model_name",
]

#: Appended to a backbone name to name its served LoRA adapter (D-111).
LORA_NAME_SUFFIX = "-lora"


class ServingError(RuntimeError):
    """Raised when a model call fails, or returns something the result contract cannot accept."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One model call's output, with the cost and latency every reported result needs.

    Attributes:
        text: The generated text.
        token_cost: Prompt and completion tokens, as counted by the server.
        latency_s: Client-measured wall-clock seconds for the call. Non-negative.
        finish_reason: Why generation stopped. ``"length"`` means the completion cap cut it off.
    """

    text: str
    token_cost: TokenCost
    latency_s: float
    finish_reason: str

    def __post_init__(self) -> None:
        if self.latency_s < 0:
            raise ValueError(f"latency_s must be non-negative, got {self.latency_s}")
        if not self.finish_reason:
            raise ValueError("finish_reason must be non-empty")

    @property
    def hit_length_cap(self) -> bool:
        """Whether the completion cap stopped generation; its rate is reported for every run."""
        return self.finish_reason == "length"


class ChatClient(ABC):
    """Base for every serving backend.

    Sampling settings are deliberately absent from the constructor: every backbone decodes with
    its own shipped defaults, so a client only carries the completion cap.

    Args:
        serving: How to reach the server.
        sampling: The completion cap.
        backbone: The model this client talks to.
    """

    def __init__(
        self, serving: ServingConfig, sampling: SamplingConfig, backbone: BackboneSpec
    ) -> None:
        self._client_name = serving.client
        self._max_tokens = sampling.max_tokens
        self._backbone = backbone
        self._served_model = served_model_name(serving, backbone)

    @property
    def backbone(self) -> BackboneSpec:
        """The model this client talks to."""
        return self._backbone

    @abstractmethod
    def complete(
        self,
        prompt: str,
        seed: int | None = None,
        system: str | None = None,
        turns: Sequence[tuple[str, str]] = (),
    ) -> Completion:
        """Send one user message, with an optional system message, and return the completion.

        Args:
            prompt: The full user message. The backbone's own chat template wraps it.
            seed: Per-request sampling seed, or None for an unseeded call.
            system: A system message to send before the user message (D-044), or None to send
                none at all.
            turns: Earlier (user, assistant) message pairs, oldest first, sent between the
                system message and ``prompt`` (D-110: T1's worked examples). Empty sends none.

        Returns:
            The generated text with its token counts, latency and finish reason.

        Raises:
            ServingError: If the call fails, or the response lacks text, token counts or a
                finish reason. Token cost is never recorded as zero or null.
        """


def served_model_name(serving: ServingConfig, backbone: BackboneSpec) -> str:
    """The model name requests address: the adapter's when one is served, else the backbone's.

    Args:
        serving: Holds ``lora_adapters``.
        backbone: The model being served.

    Returns:
        ``<backbone name>-lora`` when ``serving`` names an adapter for the backbone (D-111),
        otherwise the backbone's HuggingFace path, unchanged from every run before T5.
    """
    if serving.adapter_for(backbone.name) is None:
        return backbone.hf_path
    return f"{backbone.name}{LORA_NAME_SUFFIX}"


def get_client(
    serving: ServingConfig, sampling: SamplingConfig, backbone: BackboneSpec
) -> ChatClient:
    """Construct the serving client named in config, bound to one backbone.

    Args:
        serving: Selects the client by ``serving.client`` and configures it.
        sampling: The completion cap.
        backbone: The model the client talks to.

    Returns:
        A ready client.

    Raises:
        ServingError: If the registered name does not construct a ``ChatClient``.
    """
    client = SERVING.build(serving.client, serving, sampling, backbone)
    if not isinstance(client, ChatClient):
        raise ServingError(f"serving client {serving.client!r} is not a ChatClient")
    return client


def endpoint_url(env_var: str) -> str:
    """Read the vLLM endpoint URL from the environment.

    Args:
        env_var: Name of the variable holding the URL.

    Returns:
        The URL without a trailing slash.

    Raises:
        ServingError: If the variable is unset or empty.
    """
    url = os.environ.get(env_var, "").strip().rstrip("/")
    if not url:
        raise ServingError(
            f"environment variable {env_var} is not set; add a line such as "
            f"{env_var}=http://127.0.0.1:8000 to .env"
        )
    return url
