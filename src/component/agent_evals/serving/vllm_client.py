"""Chat client for vLLM's OpenAI-compatible server, using only the standard library."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

from ..config import SamplingConfig, ServingConfig
from ..registry import SERVING
from ..types import TokenCost
from .backbones import BackboneSpec
from .base import ChatClient, Completion, ServingError, endpoint_url

__all__ = ["VLLMChatClient"]

_CHAT_PATH = "/v1/chat/completions"


@SERVING.register("vllm_openai")
class VLLMChatClient(ChatClient):
    """Reach a backbone through ``/v1/chat/completions`` on a running vLLM server.

    A request carries only the model, an optional system message, one user message, the
    completion cap and the seed. No sampling setting is sent, so the server applies the
    backbone's own ``generation_config.json``, and the backbone's own chat template wraps the
    messages.

    Args:
        serving: Endpoint variable name and request timeout.
        sampling: The completion cap.
        backbone: The model this client talks to.
    """

    def __init__(
        self, serving: ServingConfig, sampling: SamplingConfig, backbone: BackboneSpec
    ) -> None:
        super().__init__(serving, sampling, backbone)
        self._endpoint_env = serving.endpoint_env
        self._timeout_s = serving.request_timeout_s

    def complete(
        self,
        prompt: str,
        seed: int | None = None,
        system: str | None = None,
        turns: Sequence[tuple[str, str]] = (),
    ) -> Completion:
        """Send one user message, with an optional system message, and return the completion.

        Args:
            prompt: The full user message.
            seed: Per-request sampling seed, or None for an unseeded call.
            system: A system message to send before the user message (D-044), or None to send
                none at all.
            turns: Earlier (user, assistant) message pairs, oldest first, sent before
                ``prompt`` (D-110). Empty sends none.

        Returns:
            The generated text with its token counts, client-measured latency and finish reason.

        Raises:
            ServingError: If the endpoint is unset, the call fails or times out, or the response
                is missing text, token counts or a finish reason.
        """
        url = endpoint_url(self._endpoint_env) + _CHAT_PATH
        body = self.request_body(prompt, seed, system, turns)
        started = time.perf_counter()
        response = self._post(url, body)
        return self._parse(response, time.perf_counter() - started)

    def request_body(
        self,
        prompt: str,
        seed: int | None = None,
        system: str | None = None,
        turns: Sequence[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        """The exact JSON sent for one call.

        Public so tests can pin what is sent and, as importantly, what is not.

        Args:
            prompt: The full user message.
            seed: Per-request sampling seed, or None to omit it.
            system: A system message to send before the user message, or None to omit it.
            turns: Earlier (user, assistant) message pairs, oldest first, sent before
                ``prompt``.

        Returns:
            The request body.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for user, assistant in turns:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self._served_model,
            "messages": messages,
            "max_tokens": self._max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        return body

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as handle:
                payload = json.load(handle)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ServingError(f"vLLM returned HTTP {exc.code}: {detail}") from exc
        except (OSError, ValueError) as exc:
            raise ServingError(f"request to {url} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise ServingError("vLLM response is not a JSON object")
        return payload

    def _parse(self, response: dict[str, Any], latency_s: float) -> Completion:
        try:
            choice = response["choices"][0]
            text = choice["message"]["content"]
            finish_reason = choice["finish_reason"]
            prompt_tokens = response["usage"]["prompt_tokens"]
            completion_tokens = response["usage"]["completion_tokens"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ServingError(f"vLLM response is missing a required field: {exc!r}") from exc
        if not isinstance(text, str):
            raise ServingError("vLLM response text is not a string")
        for name, value in (
            ("prompt_tokens", prompt_tokens),
            ("completion_tokens", completion_tokens),
        ):
            if type(value) is not int or value < 0:
                raise ServingError(
                    f"vLLM response has invalid {name} {value!r}; token cost is a reported "
                    "result and is never recorded as zero or null"
                )
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ServingError("vLLM response has no finish_reason")
        return Completion(
            text=text,
            token_cost=TokenCost(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
            latency_s=latency_s,
            finish_reason=finish_reason,
        )
