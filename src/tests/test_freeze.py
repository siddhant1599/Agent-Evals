"""Freezing the evaluation sample: bootstrap, provenance, and refusal to silently redraw."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from agent_evals.config import ConfigError, DataConfig, load_config
from agent_evals.datasets import DatasetError, XSumAdapter
from agent_evals.datasets.freeze import freeze_sample, main, render_sample_yaml
from agent_evals.registry import DATASETS

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "xsum.json"

#: Name the offline adapter registers under, so freezing never reaches the network in tests.
OFFLINE_NAME = "test_offline_xsum"


class OfflineXSum(XSumAdapter):
    """Serve the eight-row fixture through the registry, as ``freeze_sample`` resolves by name."""

    def _load_raw(self) -> Dataset:
        return Dataset.from_list(json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


@pytest.fixture(autouse=True)
def _offline_adapter():
    """Register the stub only for this module, so the shipped-registry test stays honest."""
    with DATASETS.temporary(OFFLINE_NAME, OfflineXSum):
        yield


@pytest.fixture
def workspace(tmp_path):
    """A config directory whose dataset config composes a not-yet-written sample file."""
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump({"name": "base", "run": {"owner": "sameer", "seed": 0}}),
        encoding="utf-8",
    )
    config = tmp_path / "data.yaml"
    config.write_text(
        "defaults: [base, data_sample]\n"
        "name: data\n"
        "data:\n"
        f"  name: {OFFLINE_NAME}\n"
        "  n_samples: 5\n"
        "  min_source_words: 1\n",
        encoding="utf-8",
    )
    return config, tmp_path / "data_sample.yaml"


class TestRender:
    def test_yaml_is_parseable_and_preserves_order(self):
        ids = ("b", "a", "c")
        parsed = yaml.safe_load(render_sample_yaml(DataConfig(), ids, "2026-09-11"))
        assert tuple(parsed["data"]["sample_ids"]) == ids

    def test_ids_are_quoted_so_numeric_ids_stay_strings(self):
        parsed = yaml.safe_load(render_sample_yaml(DataConfig(), ("27701195",), "2026-09-11"))
        assert parsed["data"]["sample_ids"] == ["27701195"]

    def test_header_records_the_draw_provenance(self):
        cfg = DataConfig(
            sample_seed=123, min_source_words=20, hf_config="FAKE_config", dedupe_sources=True
        )
        text = render_sample_yaml(cfg, ("x",), "2026-09-11")
        for expected in (cfg.hf_revision, cfg.split, "123", "20", "2026-09-11"):
            assert expected in text
        assert "# hf_config:        FAKE_config" in text
        assert "# dedupe_sources:   True" in text


class TestFreeze:
    def test_bootstraps_a_missing_sample_file_then_fills_it(self, workspace):
        config, sample = workspace
        assert not sample.exists()
        ids = freeze_sample(config, sample)
        assert len(ids) == 5
        assert tuple(yaml.safe_load(sample.read_text())["data"]["sample_ids"]) == ids

    def test_frozen_ids_round_trip_through_config_composition(self, workspace):
        config, sample = workspace
        ids = freeze_sample(config, sample)
        assert load_config(config).data.sample_ids == ids

    def test_refuses_to_redraw_once_frozen(self, workspace):
        config, sample = workspace
        freeze_sample(config, sample)
        with pytest.raises(DatasetError, match="refusing to redraw"):
            freeze_sample(config, sample)

    def test_overwrite_redraws_and_is_reproducible(self, workspace):
        config, sample = workspace
        first = freeze_sample(config, sample)
        assert freeze_sample(config, sample, overwrite=True) == first

    def test_overrides_are_applied_to_the_draw(self, workspace):
        config, sample = workspace
        assert len(freeze_sample(config, sample, ["data.n_samples=3"])) == 3

    def test_missing_data_group_is_rejected(self, tmp_path):
        config = tmp_path / "no_data.yaml"
        config.write_text("name: bare\nrun:\n  owner: sameer\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="no data group"):
            freeze_sample(config, tmp_path / "sample.yaml")


class TestCli:
    def test_main_writes_the_sample_and_exits_zero(self, workspace):
        config, sample = workspace
        code = main(["--config", str(config), "--out", str(sample), "--set", "data.n_samples=4"])
        assert code == 0
        assert len(yaml.safe_load(sample.read_text())["data"]["sample_ids"]) == 4

    def test_main_propagates_the_redraw_refusal(self, workspace):
        config, sample = workspace
        main(["--config", str(config), "--out", str(sample)])
        with pytest.raises(DatasetError):
            main(["--config", str(config), "--out", str(sample)])
