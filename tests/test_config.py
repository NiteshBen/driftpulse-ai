"""Tests for ThresholdConfig."""

import os
import tempfile
import pytest
from driftpulse.core.config import ThresholdConfig
from driftpulse.core.exceptions import ConfigurationError


def test_config_defaults_and_validation():
    cfg = ThresholdConfig()
    cfg.validate()
    assert cfg.p_value_alpha == 0.05
    assert cfg.psi_warning == 0.10
    assert cfg.psi_critical == 0.25


def test_config_invalid_alpha():
    cfg = ThresholdConfig(p_value_alpha=0.9)
    with pytest.raises(ConfigurationError):
        cfg.validate()


def test_config_invalid_psi():
    cfg = ThresholdConfig(psi_warning=0.3, psi_critical=0.2)
    with pytest.raises(ConfigurationError):
        cfg.validate()


def test_feature_overrides():
    cfg = ThresholdConfig(
        feature_overrides={
            "risk_score": {"psi_critical": 0.15},
        }
    )
    val = cfg.get_feature_threshold("risk_score", "psi_critical", 0.25)
    assert val == 0.15

    default_val = cfg.get_feature_threshold("other_feat", "psi_critical", 0.25)
    assert default_val == 0.25


def test_config_json_serialization():
    cfg = ThresholdConfig(psi_warning=0.12)
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "thresh.json")
        cfg.save_json(fpath)

        loaded = ThresholdConfig.from_json_file(fpath)
        assert loaded.psi_warning == 0.12
