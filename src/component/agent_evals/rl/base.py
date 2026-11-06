"""The reward interface: GRPO rewards wrap validated metrics and never reimplement them.

A reward turns a batch of completions into one float each, which GRPO then normalizes within each
prompt's group. A reward that recomputed a score itself would drift from the metric the paper
reports against, so every reward here holds a ``Metric`` from ``metrics/`` (AGENTS.md §4.6).

Blank completions never reach the metric, which refuses them (``metrics/base.py``). They receive
the configured ``blank_reward`` instead (D-050).

TRL calls a reward as a plain function with the completions, the prompts, and every dataset column
as keyword arguments. ``as_trl_reward`` is the one adapter exposing a ``Reward`` that way.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from ..agents.base import AgentError
from ..agents.generator import TEMPLATES, TEMPLATES_BY_NAME, InitialPromptMixin
from ..config import ExperimentConfig, RewardConfig
from ..metrics import Metric, MetricError
from ..metrics.citation_markers import remove_citations
from ..registry import REWARDS, RegistryError

__all__ = [
    "MetricReward",
    "Reward",
    "RewardError",
    "SharedMetric",
    "as_trl_reward",
    "completion_text",
    "training_template",
    "worked_examples_seed",
]


class RewardError(RuntimeError):
    """Raised for input a reward cannot score, or a value outside the reward's contract."""


def completion_text(completion: Any) -> str:
    """The text of one completion, in either of the forms TRL passes.

    Args:
        completion: A string, or a conversational completion: a list holding one assistant
            message, ``[{"role": "assistant", "content": "..."}]``.

    Returns:
        The completion's text.

    Raises:
        RewardError: For any other shape, rather than guessing which text was meant.
    """
    if isinstance(completion, str):
        return completion
    if (
        isinstance(completion, Sequence)
        and len(completion) == 1
        and isinstance(completion[0], Mapping)
        and completion[0].get("role") == "assistant"
        and isinstance(completion[0].get("content"), str)
    ):
        return completion[0]["content"]
    raise RewardError(
        "a completion must be a string or a single assistant message "
        f"[{{'role': 'assistant', 'content': str}}], got {completion!r}"
    )


class Reward(ABC):
    """Base for every GRPO reward.

    Subclasses declare ``OUTPUT_RANGE``, the dataset columns they read in ``COLUMNS``, and
    implement ``_score_texts``. ``score`` handles everything they share: extracting text, giving
    blank completions ``blank_reward``, aligning columns, and checking every value returned.

    Args:
        cfg: Reward configuration. ``cfg.name`` must be the name this class is registered under,
            so TRL's logs carry the right label.

    Raises:
        RewardError: If ``cfg.name`` is registered to another class, or ``cfg.blank_reward``
            lies outside ``OUTPUT_RANGE``.
    """

    OUTPUT_RANGE: ClassVar[tuple[float, float]]
    COLUMNS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, cfg: RewardConfig) -> None:
        if cfg.name is None:
            raise RewardError("construct each term of a several-term reward with cfg.single(name)")
        try:
            registered = REWARDS.get(cfg.name)
        except RegistryError as exc:
            raise RewardError(str(exc)) from None
        if registered is not type(self):
            raise RewardError(
                f"reward name {cfg.name!r} is registered to {registered.__name__}, "
                f"not {type(self).__name__}"
            )
        low, high = self.OUTPUT_RANGE
        if not low <= cfg.blank_reward <= high:
            raise RewardError(
                f"reward.blank_reward {cfg.blank_reward} lies outside {type(self).__name__}'s "
                f"output range [{low}, {high}]"
            )
        self._name = cfg.name
        self._blank_reward = float(cfg.blank_reward)

    @property
    def name(self) -> str:
        """The registered name, used as the reward's label in training logs."""
        return self._name

    def score(self, completions: Sequence[Any], **kwargs: Any) -> list[float]:
        """Reward every completion.

        Args:
            completions: Completions as TRL passes them: strings, or single assistant messages.
            **kwargs: Dataset columns aligned with ``completions``, plus whatever else TRL
                passes (prompts, completion ids, trainer state). Only ``COLUMNS`` are read.

        Returns:
            One float per completion, in order, each within ``OUTPUT_RANGE``.

        Raises:
            RewardError: On an empty batch, a malformed completion, a missing or misaligned
                column, or a value the metric cannot produce or the range does not allow.
        """
        if len(completions) == 0:
            raise RewardError("no completions to score")
        texts = [self._prepare_text(completion_text(completion)) for completion in completions]
        columns = {name: self._column(kwargs, name, len(texts)) for name in self.COLUMNS}
        rewards = [self._blank_reward] * len(texts)
        indices = [i for i, text in enumerate(texts) if text.strip()]
        if indices:
            values = self._score_texts(
                [texts[i] for i in indices],
                {name: [column[i] for i in indices] for name, column in columns.items()},
            )
            self._check_values(values, len(indices))
            for i, value in zip(indices, values, strict=True):
                rewards[i] = value
        return rewards

    def _prepare_text(self, text: str) -> str:
        """The text the reward scores; a text blank after this gets ``blank_reward``."""
        return text

    @abstractmethod
    def _score_texts(self, texts: list[str], columns: dict[str, list[Any]]) -> list[float]:
        """Score non-blank texts; ``columns`` holds each of ``COLUMNS``, aligned with ``texts``."""

    def _column(self, kwargs: Mapping[str, Any], name: str, size: int) -> list[Any]:
        if name not in kwargs:
            raise RewardError(f"{type(self).__name__} needs the dataset column {name!r}")
        values = kwargs[name]
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise RewardError(f"column {name!r} must be a list aligned with the completions")
        if len(values) != size:
            raise RewardError(f"column {name!r} has {len(values)} values for {size} completions")
        return list(values)

    def _check_values(self, values: list[float], count: int) -> None:
        if len(values) != count:
            raise RewardError(f"{type(self).__name__} gave {len(values)} values for {count} texts")
        low, high = self.OUTPUT_RANGE
        for index, value in enumerate(values):
            if not isinstance(value, float) or not math.isfinite(value):
                raise RewardError(f"{type(self).__name__} value {index} is {value!r}")
            if not low <= value <= high:
                raise RewardError(
                    f"{type(self).__name__} value {index} is {value}, outside [{low}, {high}]"
                )


class SharedMetric:
    """A metric shared by several rewards, scoring each batch once.

    TRL calls every reward function on the same completions one after another. Two rewards that
    read different keys of one metric, such as citation recall and precision, would otherwise
    run its model twice. This holds the last batch's rows and returns them when the next call
    has identical inputs; any other call is scored afresh. Nothing else is cached.

    Args:
        metric: The metric, constructed once for the run.
    """

    def __init__(self, metric: Metric) -> None:
        self._metric = metric
        self._last: tuple[tuple[Any, ...], list[dict[str, float]]] | None = None

    @property
    def KEYS(self) -> tuple[str, ...]:  # noqa: N802 - mirrors Metric.KEYS
        """The wrapped metric's keys."""
        return self._metric.KEYS

    @property
    def READS_CITATIONS(self) -> bool:  # noqa: N802 - mirrors Metric.READS_CITATIONS
        """Whether the wrapped metric reads citation markers."""
        return self._metric.READS_CITATIONS

    def score(
        self,
        predictions: Sequence[str],
        references: Sequence[str],
        evidence: Sequence[str],
        gold: Sequence[str | None] | None = None,
    ) -> list[dict[str, float]]:
        """:meth:`Metric.score`, reusing the previous result for identical inputs."""
        key = (
            tuple(predictions),
            tuple(references),
            tuple(evidence),
            None if gold is None else tuple(gold),
        )
        if self._last is None or self._last[0] != key:
            self._last = (key, self._metric.score(predictions, references, evidence, gold))
        return self._last[1]


class MetricReward(Reward):
    """A reward that is one key of a metric, scored against each example's evidence.

    Subclasses declare ``METRIC``, the registered metric they wrap, and ``KEY``, the one key of
    it they return. Reads the ``evidence`` and ``reference`` columns of the evidence-paired schema
    (AGENTS.md §4.2), which TRL passes as keyword arguments aligned with the completions, and
    ``gold`` when a subclass lists it in ``COLUMNS`` (D-126).

    Args:
        cfg: Reward configuration.
        metric: The metric this reward wraps, constructed once by the caller and held for the
            whole run (AGENTS.md §7.1), or a :class:`SharedMetric` over it. Its resident model is
            never reloaded here.
        strip_citations: Remove citation markers before scoring, unless the metric reads them,
            as evaluation does on a dataset whose outputs cite (``data.strip_citations``, D-126).
            A completion with no text left is blank.

    Raises:
        RewardError: If ``metric`` does not emit ``KEY``, or as for :class:`Reward`.
    """

    METRIC: ClassVar[str]
    KEY: ClassVar[str]
    COLUMNS = ("evidence", "reference")

    def __init__(
        self, cfg: RewardConfig, metric: Metric | SharedMetric, strip_citations: bool = False
    ) -> None:
        super().__init__(cfg)
        if self.KEY not in metric.KEYS:
            raise RewardError(
                f"{type(self).__name__} reads metric key {self.KEY!r}; "
                f"{type(metric).__name__} emits {metric.KEYS}"
            )
        self._metric = metric
        self._strip = strip_citations and not metric.READS_CITATIONS

    def _prepare_text(self, text: str) -> str:
        return remove_citations(text) if self._strip else text

    def _score_texts(self, texts: list[str], columns: dict[str, list[Any]]) -> list[float]:
        try:
            rows = self._metric.score(
                texts, columns["reference"], columns["evidence"], columns.get("gold")
            )
        except MetricError as exc:
            raise RewardError(f"{type(self._metric).__name__} could not score: {exc}") from exc
        return [row[self.KEY] for row in rows]


def training_template(cfg: ExperimentConfig) -> InitialPromptMixin:
    """The Generator template T5 trains on: ``training.prompt_template``, else the dataset's.

    Args:
        cfg: Resolved config with ``data`` and ``training`` groups.

    Returns:
        The template; its ``messages`` give the whole first-generation conversation.

    Raises:
        AgentError: If the named template, or the dataset's, does not exist.
    """
    name = None if cfg.training is None else cfg.training.prompt_template
    try:
        return TEMPLATES[cfg.data.name] if name is None else TEMPLATES_BY_NAME[name]
    except KeyError:
        wanted = f"template {name!r}" if name else f"a template for dataset {cfg.data.name!r}"
        raise AgentError(f"no Generator prompt: {wanted}; T5 trains on T0's prompt") from None


def worked_examples_seed(cfg: ExperimentConfig) -> int:
    """The seed a few-shot training prompt draws its worked examples from (D-118, D-136).

    Args:
        cfg: Resolved config.

    Returns:
        ``training.worked_examples_seed`` when set, else ``run.seed``.
    """
    seed = None if cfg.training is None else cfg.training.worked_examples_seed
    return cfg.run.seed if seed is None else seed


def as_trl_reward(reward: Reward) -> Callable[..., list[float]]:
    """Adapt a reward to the plain callable TRL requires. The only function escape hatch (§7.1).

    Args:
        reward: A constructed reward.

    Returns:
        A function that forwards to ``reward.score``. Its ``__name__`` is the reward's registered
        name, which TRL uses to label the reward in its logs.
    """

    def trl_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return reward.score(completions, **kwargs)

    trl_reward.__name__ = reward.name
    trl_reward.__qualname__ = reward.name
    return trl_reward
