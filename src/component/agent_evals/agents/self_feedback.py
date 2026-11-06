"""The SelfFeedback role: the feedback step of T1 Self-Refine (D-101).

In Self-Refine the same model writes feedback on its own latest output, seeing only that output
and the input (Madaan et al. 2023, §2, Eq. 2), and the loop stops when the feedback signals that
nothing is left to fix. Here the feedback covers factual consistency with the article only, not
length or style (D-104). The role is a prompt with no tools; T2's Critic is the counterpart that
has an NLI tool.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from ..config import AgentConfig
from ..registry import AGENTS
from ..serving.base import ChatClient
from ..types import AgentState, Example, Message
from .base import Agent, AgentError, VersionedTemplate
from .generator import Generator, QASourceMixin
from .self_refine_examples import (
    ASQA_SELF_REFINE_EXAMPLES,
    XSUM_SELF_REFINE_EXAMPLES,
    FewShotExample,
    QASelfRefineExample,
)

__all__ = [
    "ASQA_SELF_FEEDBACK",
    "FEEDBACK_TEMPLATES",
    "STOP_WORD",
    "XSUM_SELF_FEEDBACK",
    "QASelfFeedbackPromptTemplate",
    "SelfFeedback",
    "SelfFeedbackPromptTemplate",
    "is_stop_reply",
]

#: The whole feedback reply that means the summary needs no change (D-101). Interpreted only by
#: :func:`is_stop_reply`.
STOP_WORD = "NONE"

#: The feedback prompt's worked examples come in exactly this many (D-103).
_N_EXAMPLES = 3


@dataclass(frozen=True, slots=True)
class SelfFeedbackPromptTemplate(VersionedTemplate):
    """A versioned few-shot instruction for the feedback step, sent as chat turns (D-110).

    Each worked example is one earlier user message (the instruction, an article and a summary)
    answered by one assistant message (its feedback). The item under work is the final user
    message, opened by ``target_label`` so that it is not mistaken for another example. The
    three examples come in a fixed pattern: the first as its draft with its feedback, the second
    as its revised summary with the stop word, the third as its draft with its feedback. The
    stop-word case keeps the model from learning that every summary has an error, and sits in the
    middle so that it is not the most recent example (D-103).

    Attributes:
        instruction: The task, repeated at the top of every user message.
        target_label: The line that opens the final user message, naming it as the item to
            check.
        examples: Exactly three worked examples, in the order above.

    Raises:
        ValueError: If there are not exactly three examples.
    """

    #: What the output under check is called (D-104).
    OUTPUT_LABEL: ClassVar[str] = "Summary"

    instruction: str
    target_label: str
    examples: tuple[FewShotExample, ...] | tuple[QASelfRefineExample, ...]

    def __post_init__(self) -> None:
        if len(self.examples) != _N_EXAMPLES:
            raise ValueError(
                f"the feedback prompt needs exactly {_N_EXAMPLES} examples, "
                f"got {len(self.examples)}"
            )

    def turns(self) -> tuple[tuple[str, str], ...]:
        """The worked examples as (user, assistant) message pairs, oldest first.

        Returns:
            Three pairs: the instruction with an example's source and output, then that output's
            feedback.
        """
        first, second, third = self.examples
        shown = (
            (first, first.draft, first.feedback),
            (second, second.revised, STOP_WORD),
            (third, third.draft, third.feedback),
        )
        return tuple(
            (self._body(self._example_source(example), output), feedback)
            for example, output, feedback in shown
        )

    def render(self, example: Example, summary: str) -> str:
        """The final user message for one feedback call.

        Args:
            example: The item under work; its ``source`` is the text to check against.
            summary: The latest output for that item, and only that one (Eq. 2).

        Returns:
            The instruction, ``target_label``, then the source and the output.
        """
        return self._body(self._item_source(example), summary, self.target_label)

    def _example_source(self, example: FewShotExample) -> str:
        return f"Article:\n{example.article}"

    def _item_source(self, example: Example) -> str:
        return f"Article:\n{example.source}"

    def _body(self, source: str, output: str, label: str | None = None) -> str:
        blocks = [self.instruction, *([label] if label else []), source]
        return "\n\n".join([*blocks, f"{self.OUTPUT_LABEL}:\n{output}"])


@dataclass(frozen=True, slots=True)
class QASelfFeedbackPromptTemplate(QASourceMixin, SelfFeedbackPromptTemplate):
    """The feedback step for grounded QA: the question and passages replace the article, and the
    output under check is the answer (D-150)."""

    OUTPUT_LABEL: ClassVar[str] = "Answer"


#: D-104 wording; D-110 layout: chat turns, and the item under work labelled explicitly.
XSUM_SELF_FEEDBACK = SelfFeedbackPromptTemplate(
    name="xsum_self_feedback",
    version="v3",
    instruction=(
        "Check whether every part of the summary is supported by the article. For each part "
        'that the article does not support, write one line starting with "- " that quotes the '
        "unsupported words from the summary and says what the article states instead, or that "
        "the article does not mention it. If every part of the summary is supported by the "
        f"article, reply with only the word {STOP_WORD}."
    ),
    target_label="Now check the summary of the article below.",
    examples=XSUM_SELF_REFINE_EXAMPLES,
)

#: D-150: checks each sentence against the passages it cites, in D-104's form; a sentence that cites
#: none is unsupported. D-110 layout.
ASQA_SELF_FEEDBACK = QASelfFeedbackPromptTemplate(
    name="asqa_self_feedback",
    version="v1",
    instruction=(
        "Check whether every sentence of the answer is supported by the search results it cites. "
        "For each sentence that cites no search result, or whose cited search results do not "
        'support it, write one line starting with "- " that quotes the unsupported words and '
        "says which search result supports them, or that none does. If every sentence is "
        f"supported by the search results it cites, reply with only the word {STOP_WORD}."
    ),
    target_label="Now check the answer to the question below.",
    examples=ASQA_SELF_REFINE_EXAMPLES,
)

#: Dataset name -> the feedback template. XSum (D-105) and ASQA (D-150): T1 on another dataset
#: fails at construction until that dataset has worked examples from its own train split.
FEEDBACK_TEMPLATES: dict[str, SelfFeedbackPromptTemplate] = {
    "tier1_xsum": XSUM_SELF_FEEDBACK,
    "tier2_asqa": ASQA_SELF_FEEDBACK,
}


@AGENTS.register("self_feedback")
class SelfFeedback(Agent):
    """Writes feedback on the latest summary's factual consistency, or the stop word.

    Args:
        cfg: Selects the prompt template by dataset and supplies the request seed.
        client: The serving client; in Self-Refine, the same model that wrote the summary.

    Raises:
        AgentError: If no feedback template exists for the dataset.
    """

    ROLE = "self_feedback"

    def __init__(self, cfg: AgentConfig, client: ChatClient) -> None:
        super().__init__(cfg, client)
        try:
            self._template = FEEDBACK_TEMPLATES[cfg.dataset]
        except KeyError:
            raise AgentError(
                f"no SelfFeedback prompt template for dataset {cfg.dataset!r}; available: "
                f"{', '.join(sorted(FEEDBACK_TEMPLATES))}"
            ) from None

    @property
    def prompt_templates(self) -> Mapping[str, str]:
        """The one template this role sends."""
        return {self._template.name: self._template.label}

    def act(self, state: AgentState) -> Message:
        """Write feedback on the summary that ends ``state.history``.

        Args:
            state: The example, the history ending in the Generator's latest summary, and the
                round, which the feedback message carries.

        Returns:
            The feedback as a message, its text unmodified, with cost, latency and finish
            reason.

        Raises:
            AgentError: If the history is empty or does not end in a Generator message.
            ServingError: If the model call fails.
        """
        if not state.history or state.history[-1].role != Generator.ROLE:
            raise AgentError("self-feedback needs a history ending in the Generator's summary")
        prompt = self._template.render(state.example, state.history[-1].content)
        completion = self._client.complete(prompt, seed=self._seed, turns=self._template.turns())
        return self._message(state, completion)

    def is_stop(self, message: Message) -> bool:
        """Whether a feedback message is the stop signal.

        True only when the text, stripped, with one trailing full stop removed and stripped
        again, equals the stop word in any letter case. A blank reply, the stop word followed by
        anything else, and every bulleted reply are feedback, not a stop.

        Args:
            message: A message this role produced.

        Returns:
            True if the loop should stop.
        """
        return is_stop_reply(message.content)


def is_stop_reply(text: str) -> bool:
    """Whether a feedback reply's text is the stop signal, by D-101's rule.

    The rule of :meth:`SelfFeedback.is_stop`, on the text alone, so a finished run's replies can be
    read without the role (``analysis/``).

    Args:
        text: A feedback reply.

    Returns:
        True only when the text, stripped, with one trailing full stop removed and stripped again,
        equals the stop word in any letter case.
    """
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].strip()
    return text.casefold() == STOP_WORD.casefold()
