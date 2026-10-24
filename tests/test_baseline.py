"""Tests for BaselineStore."""

import os
import tempfile
import pytest
from driftpulse.monitoring.baseline import BaselineStore


def test_baseline_create_save_load():
    store = BaselineStore()
    dataset = {
        "score": [10, 20, 30, 40, 50],
        "tier": ["SILVER", "GOLD", "GOLD", "PLATINUM", "SILVER"],
    }
    baseline_doc = store.create_baseline(dataset, baseline_name="TestBaseline", version="1.2.0")

    assert baseline_doc["baseline_name"] == "TestBaseline"
    assert baseline_doc["total_features"] == 2
    assert "score" in baseline_doc["features"]

    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "baseline.json")
        store.save(baseline_doc, fpath)
        assert os.path.exists(fpath)

        loaded = store.load(fpath)
        assert loaded["baseline_name"] == "TestBaseline"

        profiles = store.reconstruct_profiles(loaded)
        assert len(profiles) == 2
        assert profiles["score"].mean == 30.0


def test_baseline_not_found():
    store = BaselineStore()
    with pytest.raises(FileNotFoundError):
        store.load("non_existent_baseline_file.json")
