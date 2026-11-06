"""The GRPO memory measurement grid (Q-004): which backbones, prompt caps and group sizes."""

from __future__ import annotations

from dataclasses import dataclass

from .validation import ConfigError


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    """The cells of the GRPO memory measurement (D-055).

    Every cell is one short training run on the longest training articles whose T0 prompt fits
    the cap under every backbone's tokenizer: the worst case for memory at that cap. Each cell
    records peak VRAM and seconds per step in its own run directory.

    Attributes:
        backbones: ``model_registry.csv`` names, measured in this order. Required.
        prompt_caps: Prompt lengths in tokens, chat template included. Required; each >= 1.
        group_sizes: Group sizes G, measured in ascending order; a cell that runs out of memory
            skips the larger G for the same backbone and cap. Required; each >= 2.
        typical_within_cap: Also time typical steps: freeze the articles whose prompt is
            closest to the median prompt among those within this many tokens, the training
            pool's cap. None skips the timing cells; set together with ``typical_group_size``.
        typical_group_size: Group size G of the timing cells. Safe range: >= 2.
        forced_cap: Also rerun the longest articles of this cap at ``typical_group_size`` with
            every completion forced to the completion cap: the most memory any training step can
            need (D-059). None skips these cells; must be one of ``prompt_caps``.
    """

    backbones: tuple[str, ...]
    prompt_caps: tuple[int, ...]
    group_sizes: tuple[int, ...]
    typical_within_cap: int | None = None
    typical_group_size: int | None = None
    forced_cap: int | None = None

    def __post_init__(self) -> None:
        group = "measurement"
        if not self.backbones or any(not isinstance(b, str) or not b for b in self.backbones):
            raise ConfigError(f"{group}.backbones must be a non-empty list of registry names")
        for name, floor in (("prompt_caps", 1), ("group_sizes", 2)):
            values = getattr(self, name)
            if not values or any(type(v) is not int or v < floor for v in values):
                raise ConfigError(f"{group}.{name} must be a non-empty list of integers >= {floor}")
            if len(set(values)) != len(values):
                raise ConfigError(f"{group}.{name} must not repeat a value")
        if tuple(sorted(self.group_sizes)) != self.group_sizes:
            raise ConfigError(f"{group}.group_sizes must be ascending, so an OOM skips larger G")
        cap, size = self.typical_within_cap, self.typical_group_size
        if (cap is None) != (size is None):
            raise ConfigError(
                f"{group}.typical_within_cap and {group}.typical_group_size go together"
            )
        if cap is not None and (type(cap) is not int or cap < 1):
            raise ConfigError(f"{group}.typical_within_cap must be an integer >= 1")
        if size is not None and (type(size) is not int or size < 2):
            raise ConfigError(f"{group}.typical_group_size must be an integer >= 2")
        if self.forced_cap is not None and (
            size is None or self.forced_cap not in self.prompt_caps
        ):
            raise ConfigError(
                f"{group}.forced_cap must be one of {group}.prompt_caps, with "
                f"{group}.typical_group_size set"
            )
