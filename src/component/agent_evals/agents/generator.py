"""The Generator role: writes the output from the source text, with a versioned prompt (D-016),
and in T1 Self-Refine rewrites it from the feedback on it (D-101)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..config import AgentConfig
from ..datasets.asqa import format_passages
from ..registry import AGENTS
from ..seeding import make_sampling_rng
from ..serving.base import ChatClient
from ..types import AgentState, Example, Message
from .asqa_examples import ALCE_ASQA_EXAMPLES, QAExample
from .base import Agent, AgentError, VersionedTemplate
from .self_refine_examples import (
    ASQA_SELF_REFINE_EXAMPLES,
    XSUM_SELF_REFINE_EXAMPLES,
    FewShotExample,
    QASelfRefineExample,
)

__all__ = [
    "ASQA_ALCE_VANILLA",
    "ASQA_ALCE_ZERO_SHOT",
    "ASQA_SELF_REFINE",
    "CNNDM_CRITIQUE_REVISE",
    "CNNDM_THREE_SENTENCES",
    "CRITIQUE_REVISE_TEMPLATES",
    "REFINE_TEMPLATES",
    "SAMSUM_DIALOGUE_SNIPPET",
    "TEMPLATES",
    "TEMPLATES_BY_NAME",
    "XSUM_CRITIQUE_REVISE",
    "XSUM_ONE_SENTENCE",
    "XSUM_SELF_REFINE",
    "CritiqueRevisePromptTemplate",
    "FewShotQAPromptTemplate",
    "Generator",
    "InitialPromptMixin",
    "PromptTemplate",
    "QASelfRefinePromptTemplate",
    "QASourceMixin",
    "SelfRefinePromptTemplate",
]

#: Role whose message the refine step answers (D-101).
_SELF_FEEDBACK_ROLE = "self_feedback"
#: Role whose message the critique-revise step answers (D-043); ``CriticNoNLI`` keeps it.
_CRITIC_ROLE = "critic"


class InitialPromptMixin:
    """The whole first-generation conversation, for templates with ``turns`` and ``render``.

    Training sends the conversation the Generator sends: the worked examples as earlier chat
    turns, then the item under work (D-053, D-118). The token meter counts the same messages.
    """

    __slots__ = ()

    def messages(self, example: Example, seed: int) -> list[dict[str, str]]:
        """The chat messages of the first generation call for one example.

        Args:
            example: The item under work.
            seed: Seed the worked examples are drawn from; ignored by a zero-shot template.

        Returns:
            One user and one assistant message per worked example, then one user message for
            ``example``.
        """
        chat: list[dict[str, str]] = []
        for user, assistant in self.turns(seed):  # type: ignore[attr-defined]
            chat += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        chat.append({"role": "user", "content": self.render(example)})  # type: ignore[attr-defined]
        return chat


@dataclass(frozen=True, slots=True)
class PromptTemplate(InitialPromptMixin, VersionedTemplate):
    """A versioned instruction placed above the source text.

    Attributes:
        instruction: The line sent before a blank line and the source text.
    """

    instruction: str

    def render(self, example: Example) -> str:
        """The full user message for one example.

        Args:
            example: The item to generate for; its ``source`` is the text the model reads.

        Returns:
            The instruction, a blank line, then the source text, with nothing else added.
        """
        return f"{self.instruction}\n\n{example.source}"

    def drawn_examples(self, seed: int) -> tuple[()]:
        """A zero-shot template shows no worked examples."""
        return ()

    def turns(self, seed: int) -> tuple[()]:
        """A zero-shot template sends no earlier turns."""
        return ()


@dataclass(frozen=True, slots=True)
class FewShotQAPromptTemplate(InitialPromptMixin, VersionedTemplate):
    """A versioned few-shot prompt for grounded QA: worked examples as chat turns (D-110, D-118).

    Every user message has the same form: the instruction, the question, then the passages, one
    per line in ALCE's format. Each drawn worked example is one such user message answered by one
    assistant message, its cited answer; the question under work is the final user message. A run
    draws ``n_examples`` of ``examples`` once, from its seed, and sends the same ones with every
    question (ALCE ``run.py``; Gao et al. 2023, App. C).

    Attributes:
        instruction: The task, at the top of every user message.
        examples: The pool the worked examples are drawn from.
        n_examples: How many are drawn per run.
    """

    instruction: str
    examples: tuple[QAExample, ...]
    n_examples: int

    def drawn_examples(self, seed: int) -> tuple[QAExample, ...]:
        """The run's worked examples: ``n_examples`` drawn without replacement, in draw order.

        Args:
            seed: The run seed.

        Returns:
            The same examples, in the same order, for the same seed.
        """
        return tuple(make_sampling_rng(seed).sample(self.examples, self.n_examples))

    def turns(self, seed: int) -> tuple[tuple[str, str], ...]:
        """The run's worked examples as (user, assistant) message pairs.

        Args:
            seed: The run seed.

        Returns:
            One pair per drawn example: its question and passages, then its cited answer.
        """
        return tuple(
            (self._body(e.question, format_passages(e.passage_records())), e.answer)
            for e in self.drawn_examples(seed)
        )

    def render(self, example: Example) -> str:
        """The final user message: the question under work and its passages.

        Args:
            example: The item under work; ``question`` is asked and ``source`` holds the passages.

        Returns:
            The instruction, ``Question: `` with the question, then the passages.

        Raises:
            AgentError: If the example has no question.
        """
        if example.question is None:
            raise AgentError(f"{self.label} needs a question; example {example.id!r} has none")
        return self._body(example.question, example.source)

    def _body(self, question: str, passages: str) -> str:
        return f"{self.instruction}\n\nQuestion: {question}\n\n{passages}"


@dataclass(frozen=True, slots=True)
class SelfRefinePromptTemplate(VersionedTemplate):
    """A versioned few-shot instruction for Self-Refine's refine step, sent as chat turns (D-110).

    The item under work is shown with every earlier summary and the feedback on each, oldest
    first: the refine step sees its whole history (Madaan et al. 2023, §2, Eq. 4). Each worked
    example is one earlier user message (the instruction, its article, its draft as summary 1 and
    that draft's feedback) answered by one assistant message (its revision). The item under work
    is the final user message, opened by ``target_label`` so that it is not mistaken for another
    example.

    Attributes:
        instruction: The task, repeated at the top of every user message.
        target_label: The line that opens the final user message, naming it as the item to
            rewrite.
        examples: Worked examples, all sent, in this order.
    """

    #: What each output in the history is called (D-104).
    OUTPUT_LABEL: ClassVar[str] = "Summary"

    instruction: str
    target_label: str
    examples: tuple[FewShotExample, ...] | tuple[QASelfRefineExample, ...]

    def turns(self) -> tuple[tuple[str, str], ...]:
        """The worked examples as (user, assistant) message pairs, oldest first.

        Returns:
            One pair per example: the instruction with its source, draft and feedback, then its
            revision.
        """
        return tuple(
            (self._body(self._example_source(e), [(e.draft, e.feedback)]), e.revised)
            for e in self.examples
        )

    def render(self, example: Example, history: Sequence[Message]) -> str:
        """The final user message for one refine call.

        Args:
            example: The item under work; its ``source`` is the article.
            history: Outputs and feedback alternating, oldest first, ending in feedback.

        Returns:
            The instruction, ``target_label``, then the source and the numbered outputs and
            feedback.
        """
        pairs = [(s.content, f.content) for s, f in zip(history[0::2], history[1::2], strict=True)]
        return self._body(self._item_source(example), pairs, self.target_label)

    def _example_source(self, example: FewShotExample) -> str:
        return f"Article:\n{example.article}"

    def _item_source(self, example: Example) -> str:
        return f"Article:\n{example.source}"

    def _body(self, source: str, pairs: Sequence[tuple[str, str]], label: str | None = None) -> str:
        blocks = [self.instruction, *([label] if label else []), source]
        for number, (output, feedback) in enumerate(pairs, start=1):
            blocks += [
                f"{self.OUTPUT_LABEL} {number}:\n{output}",
                f"Feedback {number}:\n{feedback}",
            ]
        return "\n\n".join(blocks)


class QASourceMixin:
    """Shows a grounded-QA source as the question, then its passages in ALCE's format (D-150).

    The same layout as :class:`FewShotQAPromptTemplate`'s user messages (D-118), shared by T1's
    ASQA feedback and refine templates in place of their ``Article:`` block.
    """

    __slots__ = ()

    def _example_source(self, example: QASelfRefineExample) -> str:
        passages = format_passages(example.qa.passage_records())
        return f"Question: {example.qa.question}\n\n{passages}"

    def _item_source(self, example: Example) -> str:
        if example.question is None:
            raise AgentError(f"{self.label} needs a question; example {example.id!r} has none")
        return f"Question: {example.question}\n\n{example.source}"


@dataclass(frozen=True, slots=True)
class QASelfRefinePromptTemplate(QASourceMixin, SelfRefinePromptTemplate):
    """Self-Refine's refine step for grounded QA: the question and passages replace the article,
    and the history numbers answers instead of summaries (D-150)."""

    OUTPUT_LABEL: ClassVar[str] = "Answer"


#: D-016: one user message, no system message and no examples.
XSUM_ONE_SENTENCE = PromptTemplate(
    name="xsum_one_sentence",
    version="v1",
    instruction="Summarize the following article in one sentence:",
)

#: D-037: the same layout as D-016, with the three-sentence count of the cited CNN/DM prompts.
CNNDM_THREE_SENTENCES = PromptTemplate(
    name="cnndm_three_sentences",
    version="v1",
    instruction="Summarize the following article in three sentences:",
)

#: D-043: the zero-shot SAMSum wording of Ramprasad et al. 2024, in D-016's layout.
SAMSUM_DIALOGUE_SNIPPET = PromptTemplate(
    name="samsum_dialogue_snippet",
    version="v1",
    instruction="Generate a summary of the following dialogue snippet:",
)

#: D-104 wording: rewrite the latest summary from its feedback, using only the article. "One
#: sentence" restates D-016's task format; it is not length feedback. D-110 layout: chat turns,
#: and the item under work labelled explicitly.
XSUM_SELF_REFINE = SelfRefinePromptTemplate(
    name="xsum_self_refine",
    version="v2",
    instruction=(
        "Rewrite the latest summary so that it fixes every problem listed in its feedback. Use "
        "only information stated in the article. Keep it to one sentence. Reply with only the "
        "rewritten summary."
    ),
    target_label="Now rewrite the latest summary of the article below.",
    examples=XSUM_SELF_REFINE_EXAMPLES,
)

#: D-118: ALCE's ASQA prompt (``prompts/asqa_default.json``, commit in ``ALCE_COMMIT``): its
#: instruction verbatim, and 2 of its 4 worked examples drawn per run from the run seed. Sent as
#: chat turns (D-110): ALCE's open ``Answer:`` label is dropped, and there is no system message
#: (D-016).
ASQA_ALCE_VANILLA = FewShotQAPromptTemplate(
    name="asqa_alce_vanilla",
    version="v1",
    instruction=(
        "Instruction: Write an accurate, engaging, and concise answer for the given question using "
        "only the provided search results (some of which might be irrelevant) and cite them "
        "properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When "
        "citing several search results, use [1][2][3]. Cite at least one document and at most "
        "three documents in each sentence. If multiple documents support the sentence, only cite "
        "a minimum sufficient subset of the documents."
    ),
    examples=ALCE_ASQA_EXAMPLES,
    n_examples=2,
)

#: D-136: ALCE's ASQA instruction with no worked examples, the second setting of the ASQA GRPO
#: memory test. Not an evaluation prompt: T0-T4 use ``ASQA_ALCE_VANILLA`` (D-118).
ASQA_ALCE_ZERO_SHOT = FewShotQAPromptTemplate(
    name="asqa_alce_zero_shot",
    version="v1",
    instruction=ASQA_ALCE_VANILLA.instruction,
    examples=(),
    n_examples=0,
)

#: D-150: D-104's refine sentences around T0's ALCE instruction (D-118), word for word, so a rewrite
#: is held to the same task and citation rules as the draft. D-110 layout.
ASQA_SELF_REFINE = QASelfRefinePromptTemplate(
    name="asqa_self_refine",
    version="v1",
    instruction=(
        "Rewrite the latest answer so that it fixes every problem listed in its feedback. Use only "
        "information stated in the search results. Write an accurate, engaging, and concise "
        "answer in an unbiased and journalistic tone. Always cite for any factual claim. When "
        "citing several search results, use [1][2][3]. Cite at least one document and at most "
        "three documents in each sentence. If multiple documents support the sentence, only cite "
        "a minimum sufficient subset of the documents. Reply with only the rewritten answer."
    ),
    target_label="Now rewrite the latest answer to the question below.",
    examples=ASQA_SELF_REFINE_EXAMPLES,
)

#: Dataset name -> the Generator's template. A dataset without an entry cannot be generated for;
#: its wording is decided when its loader arrives (D-016). ASQA's training split has its
#: evaluation template, so T5 trains on T0's prompt (D-111, D-136).
TEMPLATES: dict[str, PromptTemplate | FewShotQAPromptTemplate] = {
    "tier1_xsum": XSUM_ONE_SENTENCE,
    "tier1_cnndm": CNNDM_THREE_SENTENCES,
    "tier1_samsum": SAMSUM_DIALOGUE_SNIPPET,
    "tier2_asqa": ASQA_ALCE_VANILLA,
    "tier2_asqa_train": ASQA_ALCE_VANILLA,
}

#: Template name -> every first-generation template, for ``training.prompt_template`` (D-136).
TEMPLATES_BY_NAME: dict[str, PromptTemplate | FewShotQAPromptTemplate] = {
    template.name: template for template in (*TEMPLATES.values(), ASQA_ALCE_ZERO_SHOT)
}

#: Dataset name -> the refine template. XSum (D-105) and ASQA (D-150); another dataset gets one
#: with worked examples from its own train split.
REFINE_TEMPLATES: dict[str, SelfRefinePromptTemplate] = {
    "tier1_xsum": XSUM_SELF_REFINE,
    "tier2_asqa": ASQA_SELF_REFINE,
}


@dataclass(frozen=True, slots=True)
class CritiqueRevisePromptTemplate:
    """A versioned instruction for T2's critique-informed revision turn (D-043).

    Unlike :class:`PromptTemplate`, this also renders the article, the Generator's own prior
    summary, and the Critic's critique text, so the Generator revises against external feedback.

    Attributes:
        name: Stable template name.
        version: Version string recorded in the run manifest.
        instruction: The line sent before the article, the prior summary, and the critique.
    """

    name: str
    version: str
    instruction: str

    def render(self, example: Example, previous_output: str, critique: str) -> str:
        """The full user message for one critique-revise call.

        Args:
            example: The item under work; its ``source`` is the text to check against.
            previous_output: The Generator's own most recent summary, before this critique.
            critique: The Critic's message about ``previous_output``.

        Returns:
            The instruction, the article, the prior summary, and the critique.
        """
        return (
            f"{self.instruction}\n\n"
            f"Article:\n{example.source}\n\n"
            f"Summary:\n{previous_output}\n\n"
            f"Critique:\n{critique}"
        )

    @property
    def label(self) -> str:
        """``name@version``, as recorded in the manifest."""
        return f"{self.name}@{self.version}"


#: D-043: same one-sentence length as D-016/D-042, zero-shot, task-only wording.
XSUM_CRITIQUE_REVISE = CritiqueRevisePromptTemplate(
    name="xsum_critique_revise",
    version="v1",
    instruction=(
        "Rewrite the summary below to fix the problems raised in the critique, using only the "
        "article as evidence. If the critique found no problems, repeat the summary unchanged. "
        "Reply with only the corrected one-sentence summary."
    ),
)

#: D-043: same layout as XSUM_CRITIQUE_REVISE, with D-037's three-sentence count.
CNNDM_CRITIQUE_REVISE = CritiqueRevisePromptTemplate(
    name="cnndm_critique_revise",
    version="v1",
    instruction=(
        "Rewrite the summary below to fix the problems raised in the critique, using only the "
        "article as evidence. If the critique found no problems, repeat the summary unchanged. "
        "Reply with only the corrected three-sentence summary."
    ),
)

#: Dataset name -> the Generator's T2 critique-revise template. Mirrors TEMPLATES' per-dataset
#: sentence counts; a dataset without an entry cannot be critique-revised for.
CRITIQUE_REVISE_TEMPLATES: dict[str, CritiqueRevisePromptTemplate] = {
    "tier1_xsum": XSUM_CRITIQUE_REVISE,
    "tier1_cnndm": CNNDM_CRITIQUE_REVISE,
}


@AGENTS.register("generator")
class Generator(Agent):
    """Writes the output for an example with one model call, or rewrites it from feedback.

    Which prompt is sent is decided by the history alone, never by ``round_index``: an empty
    history gets the initial template, unchanged from T0 (D-016); a history ending in
    self-feedback gets T1's refine template (D-101); a history ending in the Critic's message
    gets T2's critique-revise template (D-043), reading the Generator's own prior summary from the
    message before it. The Generator never branches on which topology is running it.

    Args:
        cfg: Selects the prompt templates by dataset and supplies the request seed.
        client: The serving client.

    Raises:
        AgentError: If no initial-generation template exists for the dataset.
    """

    ROLE = "generator"

    #: D-044/D-048: persona line folded into the user prompt for every call that asks for it
    #: (T2; T0 sets ``state.include_role_preface=False`` and keeps D-016's original zero-shot
    #: baseline). No call ever sends this as a separate system message -- that's D-048,
    #: superseding D-045's system-message delivery. Short and task-only, sized for small 7-8B
    #: models; task-specific instructions still come from the per-call templates above.
    SYSTEM_PROMPT = (
        "You are a summarization assistant. Follow the instruction exactly and reply with only "
        "the requested summary."
    )

    def __init__(self, cfg: AgentConfig, client: ChatClient) -> None:
        super().__init__(cfg, client)
        self._dataset = cfg.dataset
        try:
            self._template = TEMPLATES[cfg.dataset]
        except KeyError:
            raise AgentError(
                f"no Generator prompt template for dataset {cfg.dataset!r}; available: "
                f"{', '.join(sorted(TEMPLATES))}"
            ) from None
        self._refine_template = REFINE_TEMPLATES.get(cfg.dataset)
        self._critique_revise_template = CRITIQUE_REVISE_TEMPLATES.get(cfg.dataset)
        # Drawn once per run, so every question in the run sees the same worked examples (D-118).
        self._drawn = self._template.drawn_examples(cfg.seed)
        self._initial_turns = self._template.turns(cfg.seed)

    @property
    def prompt_templates(self) -> Mapping[str, str]:
        """Every template this role may send, and the run's drawn worked examples, if any.

        A few-shot initial template adds ``<name>_examples``: the drawn examples' source IDs, in
        the order they are sent (D-118).
        """
        templates = {self._template.name: self._template.label}
        if self._drawn:
            templates[f"{self._template.name}_examples"] = ",".join(
                example.source_id for example in self._drawn
            )
        if self._refine_template is not None:
            templates[self._refine_template.name] = self._refine_template.label
        if self._critique_revise_template is not None:
            templates[self._critique_revise_template.name] = self._critique_revise_template.label
        return templates

    def act(self, state: AgentState) -> Message:
        """Write the first summary, rewrite it from self-feedback (T1), or revise it (T2).

        Args:
            state: The example, the history so far and the round. ``state.include_role_preface``
                decides whether ``SYSTEM_PROMPT`` is prepended to this call's user prompt
                (D-048); T0 and T1 set it False.

        Returns:
            The generated text as a message, with cost, latency and finish reason.

        Raises:
            AgentError: If the history ends in a message this role does not answer, or a needed
                template is missing for the dataset.
            ServingError: If the model call fails.
        """
        turns: tuple[tuple[str, str], ...] = ()
        if not state.history:
            prompt = self._template.render(state.example)
            turns = self._initial_turns
        elif state.history[-1].role == _SELF_FEEDBACK_ROLE:
            prompt, turns = self._refine_prompt(state)
        elif state.history[-1].role == _CRITIC_ROLE:
            prompt = self._critique_revise_prompt(state)
        else:
            raise AgentError(
                f"the Generator does not answer a history ending in {state.history[-1].role!r}"
            )
        if state.include_role_preface:
            prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        if turns:
            completion = self._client.complete(prompt, seed=self._seed, turns=turns)
        else:
            completion = self._client.complete(prompt, seed=self._seed)
        return self._message(state, completion)

    def _refine_prompt(self, state: AgentState) -> tuple[str, tuple[tuple[str, str], ...]]:
        if self._refine_template is None:
            raise AgentError(
                f"no refine prompt template for dataset {self._dataset!r}; available: "
                f"{', '.join(sorted(REFINE_TEMPLATES))}"
            )
        roles = [message.role for message in state.history]
        expected = [_SELF_FEEDBACK_ROLE if i % 2 else self.ROLE for i in range(len(roles))]
        if roles != expected:
            raise AgentError(
                "refine needs summaries and self-feedback alternating from the first summary, "
                f"got roles {roles}"
            )
        return (
            self._refine_template.render(state.example, state.history),
            self._refine_template.turns(),
        )

    def _critique_revise_prompt(self, state: AgentState) -> str:
        if self._critique_revise_template is None:
            raise AgentError(f"no Generator critique-revise template for dataset {self._dataset!r}")
        if len(state.history) < 2:
            raise AgentError(
                "a critique-revise round needs the Generator's prior summary followed by "
                "the Critic's critique in history"
            )
        return self._critique_revise_template.render(
            state.example, state.history[-2].content, state.history[-1].content
        )
