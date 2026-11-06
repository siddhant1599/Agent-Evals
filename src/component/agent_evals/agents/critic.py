"""The Critic role: scrutinizes a draft against its article using the shared NLI tool (D-043).
``CriticNoNLI`` is the same role with no tool access, critiquing from context alone (D-049)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..config import AgentConfig
from ..registry import AGENTS
from ..serving.base import ChatClient
from ..types import AgentState, Example, Message
from .base import Agent, AgentError
from .tools import NLIFactCheckTool

__all__ = [
    "CRITIQUE_TEMPLATES",
    "XSUM_CRITIQUE",
    "CNNDM_CRITIQUE",
    "Critic",
    "CriticNoNLI",
    "CritiquePromptTemplate",
]


@dataclass(frozen=True, slots=True)
class CritiquePromptTemplate:
    """A versioned instruction for the Critic's turn (D-043).

    Attributes:
        name: Stable template name.
        version: Version string recorded in the run manifest.
        instruction: The line sent before the article, the draft, and any flagged sentences.
    """

    name: str
    version: str
    instruction: str

    def render(self, example: Example, draft: str, flagged: str) -> str:
        """The full user message for one critique call.

        Args:
            example: The item under work; its ``source`` is the text to check against.
            draft: The Generator's output being critiqued.
            flagged: Lines describing sentences the NLI tool found weakly supported, or empty.

        Returns:
            The instruction, the article, the draft, and the tool's findings if any.
        """
        body = f"{self.instruction}\n\nArticle:\n{example.source}\n\nSummary:\n{draft}"
        if flagged:
            body += (
                "\n\nAn automatic fact-checker flagged these summary sentences as not clearly "
                f"supported by the article:\n{flagged}"
            )
        return body

    @property
    def label(self) -> str:
        """``name@version``, as recorded in the manifest."""
        return f"{self.name}@{self.version}"


#: D-043: zero-shot, task-only wording sized for small 7-8B models, same spirit as D-016/D-042.
XSUM_CRITIQUE = CritiquePromptTemplate(
    name="xsum_critique",
    version="v1",
    instruction=(
        "List the factual problems in the summary below, using only the article as evidence. "
        "Be brief and specific. If there are none, reply with exactly: No factual problems found."
    ),
)

#: D-043: identical wording to XSUM_CRITIQUE; the instruction does not depend on sentence count.
CNNDM_CRITIQUE = CritiquePromptTemplate(
    name="cnndm_critique",
    version="v1",
    instruction=(
        "List the factual problems in the summary below, using only the article as evidence. "
        "Be brief and specific. If there are none, reply with exactly: No factual problems found."
    ),
)

#: Dataset name -> the Critic's template. A dataset without an entry cannot be critiqued for.
CRITIQUE_TEMPLATES: dict[str, CritiquePromptTemplate] = {
    "tier1_xsum": XSUM_CRITIQUE,
    "tier1_cnndm": CNNDM_CRITIQUE,
}


@AGENTS.register("critic")
class Critic(Agent):
    """Scrutinizes a Generator draft against its article, using a shared NLI tool plus one call.

    The tool runs first (code-driven, not the model invoking a function): it scores every draft
    sentence against every article sentence and flags the ones not clearly entailed. Those
    findings are spliced into the single LLM call that phrases the critique -- this is the
    "access to tools" T2 design decision (2026-09-29): a small model consulting a deterministic
    checker's output, not the model driving tool calls itself.

    Args:
        cfg: Selects the prompt template by dataset and supplies the request seed.
        client: The serving client.
        tool: The NLI fact-check tool. Built by ``_build_tool`` when not given -- the
            resident-model pattern every ``Metric`` already follows. Tests inject a fake to
            avoid loading the real DeBERTa checkpoint.

    Raises:
        AgentError: If no critique template exists for the dataset.
    """

    ROLE = "critic"

    #: D-044/D-048: persona line folded into the user prompt of every Critic call -- never sent
    #: as a separate system message (D-048 supersedes D-045's approach). Short and task-only,
    #: matching Generator's.
    SYSTEM_PROMPT = (
        "You are a fact-checking assistant. Follow the instruction exactly and reply with only "
        "the requested critique."
    )

    def __init__(
        self, cfg: AgentConfig, client: ChatClient, tool: NLIFactCheckTool | None = None
    ) -> None:
        super().__init__(cfg, client)
        try:
            self._template = CRITIQUE_TEMPLATES[cfg.dataset]
        except KeyError:
            raise AgentError(
                f"no Critic prompt template for dataset {cfg.dataset!r}; available: "
                f"{', '.join(sorted(CRITIQUE_TEMPLATES))}"
            ) from None
        self._tool = tool if tool is not None else self._build_tool()

    def _build_tool(self) -> NLIFactCheckTool | None:
        """The NLI tool this Critic consults before writing its critique.

        A subclass overrides this to return ``None`` and critique from the article and draft
        alone, with no deterministic checker's findings spliced in (D-049).
        """
        # D-047: forced to CPU, not auto-detected. Measured on the real A10G (2026-10-05): the
        # tool loading onto CUDA before vLLM starts left only 20.03/22.06 GiB free, under vLLM's
        # own 0.92 utilization target (20.3 GiB) -- vLLM refused to start. CPU avoids the two
        # models competing for the same GPU memory at all; the cost is latency on the critique
        # step, not correctness.
        return NLIFactCheckTool(device="cpu")

    @property
    def prompt_templates(self) -> Mapping[str, str]:
        """The one template this role sends."""
        return {self._template.name: self._template.label}

    def act(self, state: AgentState) -> Message:
        """Critique the most recent Generator draft in ``state.history``.

        Args:
            state: The example and round. The last message in history must be the draft to
                critique.

        Returns:
            The critique text as a message, with cost, latency and finish reason.

        Raises:
            AgentError: If history is empty.
            ServingError: If the model call fails.
        """
        if not state.history:
            raise AgentError("the Critic needs a prior Generator message to critique")
        draft = state.history[-1].content
        if self._tool is None:
            flagged = ""
        else:
            weak_claims = self._tool.find_weak_claims(state.example.source, draft)
            flagged = "\n".join(
                f'- "{claim.sentence}" ({claim.label.lower()})' for claim in weak_claims
            )
        prompt = f"{self.SYSTEM_PROMPT}\n\n{self._template.render(state.example, draft, flagged)}"
        completion = self._client.complete(prompt, seed=self._seed)
        return Message(
            role=self.ROLE,
            content=completion.text,
            round_index=state.round_index,
            token_cost=completion.token_cost,
            latency_s=completion.latency_s,
            finish_reason=completion.finish_reason,
        )


@AGENTS.register("critic_no_nli")
class CriticNoNLI(Critic):
    """Critiques the Generator's draft from the article and draft alone (D-049).

    No NLI tool is ever built or consulted -- the model judges factual consistency entirely
    from context, with no deterministic checker's findings spliced into its prompt. Everything
    else (the critique wording, the persona, the one-call contract) is identical to ``Critic``.
    """

    def _build_tool(self) -> None:
        return None
