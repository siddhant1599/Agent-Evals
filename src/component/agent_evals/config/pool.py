"""How the frozen T5 training pool is filtered by prompt length (D-057)."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ConfigError


@dataclass(frozen=True, slots=True)
class PoolConfig:
    """The prompt-length filter applied once, when the training pool is frozen.

    An article is eligible only if its T0 prompt, chat template included, fits ``prompt_cap``
    tokens under every tokenizer in ``backbones``, so every backbone trains on the same articles.
    The pool's size and draw seed are ``data.n_samples`` and ``data.sample_seed``.

    Attributes:
        prompt_cap: Longest prompt allowed, in tokens (D-057). Required; >= 1.
        backbones: ``model_registry.csv`` names whose tokenizers must all fit the cap. Required.
    """

    prompt_cap: int
    backbones: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.prompt_cap) is not int or self.prompt_cap < 1:
            raise ConfigError(f"pool.prompt_cap must be an integer >= 1, got {self.prompt_cap!r}")
        if not self.backbones or any(not isinstance(b, str) or not b for b in self.backbones):
            raise ConfigError("pool.backbones must be a non-empty list of registry names")
        if len(set(self.backbones)) != len(self.backbones):
            raise ConfigError("pool.backbones must not repeat a name")
