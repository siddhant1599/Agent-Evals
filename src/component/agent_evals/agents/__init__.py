"""Agent roles; import implementations so their registry entries are available."""

from .base import Agent, AgentError
from .critic import CRITIQUE_TEMPLATES, Critic, CriticNoNLI, CritiquePromptTemplate
from .generator import TEMPLATES, Generator, PromptTemplate
from .self_feedback import SelfFeedback

__all__ = [
    "CRITIQUE_TEMPLATES",
    "TEMPLATES",
    "Agent",
    "AgentError",
    "Critic",
    "CriticNoNLI",
    "CritiquePromptTemplate",
    "Generator",
    "PromptTemplate",
    "SelfFeedback",
]
