"""T5: the RL-trained single agent; T0's one call, served with the GRPO adapter (D-111)."""

from __future__ import annotations

from ..registry import TOPOLOGIES
from .t0_single_agent import SingleAgent

__all__ = ["RLAgent"]


@TOPOLOGIES.register("t5_rl_agent")
class RLAgent(SingleAgent):
    """One Generator call with T0's prompt, sent to the backbone fine-tuned with GRPO.

    The orchestration is T0's, unchanged: GRPO trained on T0's exact prompt, sent as one user
    message (``rl/trainer.py``), so the adapter sees at inference the prompt it was trained on, and
    T0's call sends no role preface. What differs is the model
    behind the client: the topology config sets ``requires_adapter``, and serving loads the
    adapter named for the backbone in ``serving.lora_adapters`` (D-111). Registered under its own
    name so that every T5 result is labelled T5 (AGENTS.md §4.4).
    """
