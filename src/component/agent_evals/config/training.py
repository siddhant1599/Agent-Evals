"""What a T5 GRPO training run binds: the algorithm, the LoRA adapter and the 4-bit base."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .validation import ConfigError, _check_positive, _check_text

#: Compute dtypes the 4-bit base may run its matmuls in.
_COMPUTE_DTYPES = ("bfloat16", "float16", "float32")
#: TRL 1.14.1's ways of combining several reward functions.
_AGGREGATIONS = ("sum_then_normalize", "normalize_then_sum")
_EXAMPLE_DRAWS = ("per_run", "per_question")


def _check_finite(group: str, obj: object, names: tuple[str, ...], *, positive: bool) -> None:
    for name in names:
        value = getattr(obj, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{group}.{name} must be a number, got {value!r}")
        if not math.isfinite(value) or (positive and value <= 0):
            bound = "> 0" if positive else "finite"
            raise ConfigError(f"{group}.{name} must be {bound}, got {value!r}")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """GRPO training for T5 (D-046): TRL's ``GRPOTrainer`` on a LoRA adapter over the 4-bit base.

    Defaults are the approved values (D-054 unless noted). The five fields without a default are
    not decided yet and have no default on purpose, like ``run.owner``: group size and batch
    size wait on the memory measurement (Q-004), step count and checkpointing on the training
    budget (Q-011). The completion cap is ``sampling.max_tokens`` (D-012), so T5 generates under
    the same cap as T0.

    Attributes:
        num_generations: Group size G, completions sampled per prompt. Safe range: >= 2.
        per_device_train_batch_size: Completions per forward pass, and per generation call
            (D-063). Safe range: >= 1.
        gradient_accumulation_steps: Forward passes per optimizer step. Safe range: >= 1.
        max_steps: Optimizer steps in the run. Safe range: >= 1.
        save_steps: Save an adapter checkpoint every N steps. Safe range: >= 1.
        beta: KL weight against the frozen base; Shao et al. 2024 §4.2. Must be > 0: TRL computes
            and logs KL only when it is non-zero (D-053).
        loss_type: TRL loss aggregation. ``dr_grpo`` divides by a constant rather than each
            completion's length, removing GRPO's bias toward short positive-advantage
            completions (Liu et al. 2025).
        scale_rewards: TRL advantage scaling. ``none`` keeps the raw difference from the group
            mean, without dividing by the group's standard deviation (Liu et al. 2025).
        epsilon: Clipping range of the probability ratio; Schulman et al. 2017 Table 1.
            Safe range: > 0.
        num_iterations: Optimizer passes per generated batch; Shao et al. 2024 §4.2 use one.
            Safe range: >= 1.
        learning_rate: Adapter learning rate: Shao et al.'s full fine-tuning 1e-6 times the
            ~10x LoRA factor measured for RL (Schulman et al. 2025). Safe range: > 0.
        lr_scheduler_type: Transformers scheduler name; constant, as QLoRA (Dettmers et al. 2023
            App. B.2).
        warmup_steps: Warmup steps, or a fraction of ``max_steps`` when below 1. Safe: >= 0.
        optim: Transformers optimizer name; paged AdamW absorbs memory spikes under gradient
            checkpointing (Dettmers et al. 2023 §3).
        max_grad_norm: Gradient clipping norm; Dettmers et al. 2023 App. B.2. Safe range: > 0.
        gradient_checkpointing: Recompute activations to save memory; the gradients are the same.
        mask_truncated_completions: Leave completions that hit the cap out of the loss. Off, so
            the reward still scores a summary that runs to the cap.
        temperature: Rollout sampling temperature; 1.0 samples the policy unchanged (Liu et al.
            2025). Safe range: > 0.
        top_p: Rollout nucleus sampling; 1.0 disables it. TRL applies top-p to generation only,
            not to the log-probabilities in the loss. Safe range: (0, 1].
        top_k: Rollout top-k; 0 disables it, for the same reason. Safe range: >= 0.
        repetition_penalty: Rollout repetition penalty; 1.0 disables it. Safe range: > 0.
        lora_rank: LoRA rank r; SCRPO §4.2 (Hu et al. 2025). Safe range: >= 1.
        lora_alpha: LoRA scaling alpha; SCRPO §4.2. Safe range: >= 1.
        lora_dropout: Dropout on the LoRA input. 0: dropout gives the same input different
            log-probabilities, which adds noise to GRPO's probability ratio. Safe range: [0, 1).
        compute_dtype: Dtype the 4-bit base computes in; bf16, as QLoRA (Dettmers et al. 2023).
        lora_target_modules: Modules the adapter wraps. All linear layers: LoRA on query and
            value only underperforms (Dettmers et al. 2023 §4; AGENTS.md §12.1).
        quant_type: 4-bit format of the base; NF4, the format serving uses (D-008, D-041).
        double_quant: Quantize the quantization constants too, as serving does (D-008, D-041).
        adapter_root: Where adapters and checkpoints are written, one folder per run id. Outside
            the repository: weights never go into the committed ``results/`` (D-053). ``~`` is
            expanded.
        force_full_completions: Measurement only: no completion may end before
            ``sampling.max_tokens``, so every step holds the longest completions training can
            produce (D-059). Off for training.
        multi_objective_aggregation: How TRL combines several reward terms.
            ``sum_then_normalize`` (TRL's default; identical to a single reward) sums the
            weighted terms, then scales as ``scale_rewards`` says. ``normalize_then_sum`` is
            GDPO (Liu et al. 2026): each term is normalized within its group, the weighted sum
            is normalized again (D-134). TRL then divides by standard deviations whatever
            ``scale_rewards`` says, so it requires ``scale_rewards: group`` to state that.
        prompt_template: Name of the Generator template training renders, for a prompt other
            than the dataset's T0 template, such as ASQA without worked examples (D-136). None
            trains on the dataset's T0 template, as T5 does on XSum (D-053).
        worked_examples_seed: Seed the worked examples of a few-shot template are drawn from,
            once per run, as evaluation draws them (D-118). None draws from ``run.seed``.
            Safe range: >= 0.
        worked_examples_draw: ``per_run`` sends one draw with every question, as evaluation
            does (D-118). ``per_question`` draws afresh for each question, from a seed derived
            from ``worked_examples_seed`` and the question's ID, so training covers the pairs
            every evaluation seed draws (D-139).
    """

    num_generations: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    save_steps: int
    beta: float = 0.04
    loss_type: str = "dr_grpo"
    scale_rewards: str = "none"
    epsilon: float = 0.2
    num_iterations: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: float = 0.0
    optim: str = "paged_adamw_32bit"
    max_grad_norm: float = 0.3
    gradient_checkpointing: bool = True
    mask_truncated_completions: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    compute_dtype: str = "bfloat16"
    lora_target_modules: str = "all-linear"
    quant_type: str = "nf4"
    double_quant: bool = True
    adapter_root: str = "~/.cache/agent_evals/adapters"
    force_full_completions: bool = False
    multi_objective_aggregation: str = "sum_then_normalize"
    prompt_template: str | None = None
    worked_examples_seed: int | None = None
    worked_examples_draw: str = "per_run"

    def __post_init__(self) -> None:
        group = "training"
        _check_text(
            group,
            self,
            (
                "loss_type",
                "scale_rewards",
                "lr_scheduler_type",
                "optim",
                "compute_dtype",
                "lora_target_modules",
                "quant_type",
                "adapter_root",
            ),
        )
        _check_positive(
            group,
            self,
            (
                "num_iterations",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "max_steps",
                "save_steps",
                "lora_rank",
                "lora_alpha",
            ),
        )
        _check_finite(
            group,
            self,
            (
                "beta",
                "epsilon",
                "learning_rate",
                "max_grad_norm",
                "temperature",
                "top_p",
                "repetition_penalty",
            ),
            positive=True,
        )
        _check_finite(group, self, ("warmup_steps", "lora_dropout"), positive=False)
        if type(self.num_generations) is not int or self.num_generations < 2:
            raise ConfigError(
                f"{group}.num_generations must be an integer >= 2: GRPO compares completions "
                f"within a group, got {self.num_generations!r}"
            )
        if type(self.top_k) is not int or self.top_k < 0:
            raise ConfigError(f"{group}.top_k must be an integer >= 0, got {self.top_k!r}")
        if self.top_p > 1:
            raise ConfigError(f"{group}.top_p must lie in (0, 1], got {self.top_p!r}")
        if self.warmup_steps < 0:
            raise ConfigError(f"{group}.warmup_steps must be >= 0, got {self.warmup_steps!r}")
        if not 0 <= self.lora_dropout < 1:
            raise ConfigError(f"{group}.lora_dropout must lie in [0, 1), got {self.lora_dropout!r}")
        if self.compute_dtype not in _COMPUTE_DTYPES:
            raise ConfigError(
                f"{group}.compute_dtype must be one of {', '.join(_COMPUTE_DTYPES)}, "
                f"got {self.compute_dtype!r}"
            )
        for name in (
            "gradient_checkpointing",
            "mask_truncated_completions",
            "double_quant",
            "force_full_completions",
        ):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"{group}.{name} must be a boolean")
        self._check_prompt_and_aggregation(group)

    def _check_prompt_and_aggregation(self, group: str) -> None:
        if self.multi_objective_aggregation not in _AGGREGATIONS:
            raise ConfigError(
                f"{group}.multi_objective_aggregation must be one of {', '.join(_AGGREGATIONS)}, "
                f"got {self.multi_objective_aggregation!r}"
            )
        if (
            self.multi_objective_aggregation == "normalize_then_sum"
            and self.scale_rewards != "group"
        ):
            raise ConfigError(
                f"{group}.multi_objective_aggregation normalize_then_sum divides by group "
                "standard deviations whatever scale_rewards says; set "
                f"{group}.scale_rewards to group (D-134)"
            )
        if self.prompt_template is not None and (
            not isinstance(self.prompt_template, str) or not self.prompt_template.strip()
        ):
            raise ConfigError(f"{group}.prompt_template must be None or a template name")
        seed = self.worked_examples_seed
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ConfigError(f"{group}.worked_examples_seed must be None or an integer >= 0")
        if self.worked_examples_draw not in _EXAMPLE_DRAWS:
            raise ConfigError(
                f"{group}.worked_examples_draw must be one of {', '.join(_EXAMPLE_DRAWS)}, "
                f"got {self.worked_examples_draw!r}"
            )
