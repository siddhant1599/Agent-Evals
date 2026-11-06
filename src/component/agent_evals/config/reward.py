"""What a GRPO reward binds at construction."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .validation import ConfigError, _check_text


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """The reward T5 is trained on with GRPO: one registered reward, or several terms.

    Attributes:
        name: Registered reward name, for a single reward. ``alignscore`` is the XSum reward
            (D-048). None when ``terms`` is set.
        terms: Registered reward names, for a reward of several terms, each given to TRL as its
            own reward function and combined as ``training.multi_objective_aggregation`` says.
            The ASQA reward has three (D-133). None for a single reward.
        weights: One weight per term, in order; TRL's ``reward_weights``. Equal for ASQA
            (D-135). Required with ``terms``; each > 0.
        blank_reward: Reward for a blank completion, on every term. No metric scores blank text
            (``metrics/base.py``), so the reward assigns this value instead. 0.0 is the bottom of
            every registered reward's [0, 1] range, so GRPO never prefers a blank to a real
            output (D-050). Safe range: each selected reward's output range, which it checks.
    """

    name: str | None = "alignscore"
    terms: tuple[str, ...] | None = None
    weights: tuple[float, ...] | None = None
    blank_reward: float = 0.0

    def __post_init__(self) -> None:
        if (self.name is None) == (self.terms is None):
            raise ConfigError("set exactly one of reward.name and reward.terms (with name: null)")
        if self.name is not None:
            _check_text("reward", self, ("name",))
            if self.weights is not None:
                raise ConfigError("reward.weights applies to reward.terms only")
        else:
            self._check_terms()
        value = self.blank_reward
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"reward.blank_reward must be a number, got {value!r}")
        if not math.isfinite(value):
            raise ConfigError(f"reward.blank_reward must be finite, got {value!r}")

    @property
    def names(self) -> tuple[str, ...]:
        """Every reward function's registered name, in the order TRL receives them."""
        return self.terms if self.terms is not None else (self.name,)

    def single(self, name: str) -> RewardConfig:
        """The configuration of one term, as its reward class is constructed with.

        Args:
            name: One of :attr:`names`.

        Returns:
            A single-reward configuration with this one's ``blank_reward``.
        """
        if name not in self.names:
            raise ConfigError(f"{name!r} is not one of reward {self.names}")
        return RewardConfig(name=name, blank_reward=self.blank_reward)

    def _check_terms(self) -> None:
        terms, weights = self.terms, self.weights
        if (
            not isinstance(terms, tuple)
            or len(terms) < 2
            or any(not isinstance(term, str) or not term.strip() for term in terms)
            or len(set(terms)) != len(terms)
        ):
            raise ConfigError("reward.terms must list at least two unique registered rewards")
        if weights is None or len(weights) != len(terms):
            raise ConfigError("reward.weights must give one weight per reward.terms entry")
        for weight in weights:
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ConfigError(f"reward.weights must be numbers, got {weight!r}")
            if not math.isfinite(weight) or weight <= 0:
                raise ConfigError(f"reward.weights must be finite and > 0, got {weight!r}")
