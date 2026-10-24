"""Tests for EmbeddingDriftDetector."""

import random
import pytest
from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus
from driftpulse.detectors.embedding_detector import EmbeddingDriftDetector


def test_embedding_no_drift():
    random.seed(42)
    dim = 16
    base_vec = [1.0] * dim
    ref = [[base_vec[i] + random.gauss(0, 0.05) for i in range(dim)] for _ in range(100)]
    curr = [[base_vec[i] + random.gauss(0, 0.05) for i in range(dim)] for _ in range(100)]

    detector = EmbeddingDriftDetector(cosine_threshold=0.15)
    res = detector.compare_embeddings(ref, curr)

    assert res.has_drift is False
    assert res.centroid_drift_status in (DriftStatus.NO_DRIFT, DriftStatus.LOW_DRIFT)
    assert res.dimension == dim


def test_embedding_severe_drift():
    random.seed(42)
    dim = 16
    ref = [[random.gauss(0, 1) for _ in range(dim)] for _ in range(100)]
    curr = [[random.gauss(2, 1) for _ in range(dim)] for _ in range(100)]  # Clear shift

    detector = EmbeddingDriftDetector(cosine_threshold=0.10)
    res = detector.compare_embeddings(ref, curr)

    assert res.has_drift is True
    assert res.centroid_drift_status in (DriftStatus.MODERATE_DRIFT, DriftStatus.CRITICAL_DRIFT)
    assert len(res.top_shifted_dimensions) > 0


def test_embedding_dim_mismatch_error():
    detector = EmbeddingDriftDetector()
    with pytest.raises(StatisticalTestError):
        detector.compare_embeddings([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
