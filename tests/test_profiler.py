"""Tests for DataProfiler and feature type inference."""

import pytest
from driftpulse.core.exceptions import ProfilingError
from driftpulse.core.profiler import DataProfiler, infer_feature_type, is_number
from driftpulse.core.types import FeatureType


def test_is_number():
    assert is_number(42) is True
    assert is_number(3.14) is True
    assert is_number(True) is False
    assert is_number(False) is False
    assert is_number("42") is False
    assert is_number(float("nan")) is False
    assert is_number(float("inf")) is False


def test_infer_feature_type():
    assert infer_feature_type([1.5, 2.7, 3.9, 4.1]) == FeatureType.NUMERICAL
    assert infer_feature_type(["apple", "banana", "orange"]) == FeatureType.CATEGORICAL
    assert infer_feature_type([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]) == FeatureType.EMBEDDING
    long_texts = ["This is a very long sentence representing an LLM generated paragraph with many tokens and detailed vocabulary." for _ in range(10)]
    assert infer_feature_type(long_texts) == FeatureType.TEXT


def test_profile_numerical():
    profiler = DataProfiler(n_histogram_bins=5)
    data = [10.0, 20.0, 30.0, 40.0, 50.0, None]
    profile = profiler.profile_series("test_num", data)

    assert profile.name == "test_num"
    assert profile.feature_type == FeatureType.NUMERICAL
    assert profile.count == 6
    assert profile.missing_count == 1
    assert round(profile.missing_rate, 4) == round(1 / 6, 4)
    assert profile.mean == 30.0
    assert profile.min_val == 10.0
    assert profile.max_val == 50.0
    assert "p50" in profile.quantiles
    assert len(profile.histogram["bin_edges"]) == 6
    assert sum(profile.histogram["counts"]) == 5


def test_profile_categorical():
    profiler = DataProfiler()
    data = ["RED", "BLUE", "RED", "GREEN", "RED", "BLUE", None]
    profile = profiler.profile_series("color", data)

    assert profile.name == "color"
    assert profile.feature_type == FeatureType.CATEGORICAL
    assert profile.count == 7
    assert profile.missing_count == 1
    assert profile.frequencies["RED"] == 0.5  # 3/6
    assert profile.frequencies["BLUE"] == round(2 / 6, 4)


def test_profile_empty_error():
    profiler = DataProfiler()
    with pytest.raises(ProfilingError):
        profiler.profile_series("empty", [])


def test_profile_dataset():
    profiler = DataProfiler()
    dataset = {
        "num": [1, 2, 3, 4, 5],
        "cat": ["A", "B", "A", "B", "A"],
    }
    profiles = profiler.profile_dataset(dataset)
    assert "num" in profiles
    assert "cat" in profiles
    assert profiles["num"].feature_type == FeatureType.NUMERICAL
    assert profiles["cat"].feature_type == FeatureType.CATEGORICAL
