"""Seeding: the single call site that makes 'three seeds, mean and std' meaningful."""

from __future__ import annotations

import random

import pytest
from agent_evals.seeding import MAX_SEED, make_sampling_rng, set_seed


def _draws(seed: int) -> list[float]:
    set_seed(seed)
    return [random.random() for _ in range(5)]


class TestDeterminism:
    def test_same_seed_gives_the_same_draws(self):
        assert _draws(0) == _draws(0)

    def test_different_seeds_give_different_draws(self):
        assert _draws(0) != _draws(1)

    def test_the_three_protocol_seeds_are_all_distinct(self):
        assert len({tuple(_draws(s)) for s in (0, 1, 2)}) == 3


class TestValidation:
    @pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
    def test_out_of_range_seed_rejected(self, seed):
        with pytest.raises(ValueError, match="seed must be in"):
            set_seed(seed)

    @pytest.mark.parametrize("seed", [0, 1, 2, MAX_SEED])
    def test_in_range_seeds_accepted(self, seed):
        assert set_seed(seed)["seed"] == seed


class TestSamplingStream:
    def test_private_factory_is_repeatable_and_preserves_global_state(self):
        before = random.getstate()
        first = make_sampling_rng(123)
        second = make_sampling_rng(123)
        assert first.sample(range(20), 8) == second.sample(range(20), 8)
        assert random.getstate() == before

    @pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, True, 1.5])
    def test_invalid_sampling_seed_raises(self, seed):
        with pytest.raises(ValueError, match="seed"):
            make_sampling_rng(seed)


class TestManifestRecord:
    def test_records_what_it_actually_seeded(self):
        record = set_seed(0)
        assert record["seed"] == 0
        assert record["python"] is True
        assert set(record) == {"seed", "python", "numpy", "torch", "torch_cuda", "deterministic"}

    def test_absent_libraries_are_none_not_omitted(self):
        # A manifest must distinguish "seeded numpy" from "numpy was not installed" rather than
        # implying a determinism that was never established.
        record = set_seed(0)
        for key in ("numpy", "torch"):
            assert key in record

    def test_deterministic_defaults_to_off(self):
        assert set_seed(0)["deterministic"] is False
