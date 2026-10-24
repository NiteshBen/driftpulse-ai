"""Tests for TextDriftDetector."""

import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.detectors.text_detector import TextDriftDetector


def test_text_no_drift():
    ref = ["Short sentence one.", "Short sentence two.", "Short sentence three."] * 20
    curr = ["Short sentence four.", "Short sentence five.", "Short sentence six."] * 20

    detector = TextDriftDetector(alpha=0.01)
    res = detector.compare_texts(ref, curr)

    assert res.has_drift is False
    assert res.word_count_drift_status in (DriftStatus.NO_DRIFT, DriftStatus.LOW_DRIFT)


def test_text_severe_drift():
    ref = ["Hello world"] * 50
    curr = [
        "This is a massive paragraph containing multiple detailed explanations, code blocks, technical documentation, and complex verbiage."
    ] * 50

    detector = TextDriftDetector()
    res = detector.compare_texts(ref, curr)

    assert res.has_drift is True
    assert res.word_count_drift_status in (DriftStatus.MODERATE_DRIFT, DriftStatus.CRITICAL_DRIFT)
    assert res.avg_words_curr > res.avg_words_ref


def test_text_empty_error():
    detector = TextDriftDetector()
    with pytest.raises(StatisticalTestError):
        detector.compare_texts([], ["text"])
