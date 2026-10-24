"""High-dimensional embedding and latent vector drift detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType
from driftpulse.statistical.statistics import cosine_distance_vectors


@dataclass
class EmbeddingDriftResult:
    """Detailed diagnosis for embedding space distribution shift."""
    cosine_centroid_distance: float
    centroid_drift_status: DriftStatus
    dimension: int
    sample_count_ref: int
    sample_count_curr: int
    mean_norm_ref: float
    mean_norm_curr: float
    norm_shift_ratio: float
    top_shifted_dimensions: List[Dict[str, Any]] = field(default_factory=list)
    has_drift: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cosine_centroid_distance": round(self.cosine_centroid_distance, 6),
            "centroid_drift_status": self.centroid_drift_status.value,
            "dimension": self.dimension,
            "sample_count_ref": self.sample_count_ref,
            "sample_count_curr": self.sample_count_curr,
            "mean_norm_ref": round(self.mean_norm_ref, 4),
            "mean_norm_curr": round(self.mean_norm_curr, 4),
            "norm_shift_ratio": round(self.norm_shift_ratio, 4),
            "top_shifted_dimensions": self.top_shifted_dimensions,
            "has_drift": self.has_drift,
        }


class EmbeddingDriftDetector:
    """Monitors vector embeddings from LLM retrievals, RAG contexts, and vector databases."""

    def __init__(self, cosine_threshold: float = 0.12, norm_shift_threshold: float = 0.20):
        self.cosine_threshold = cosine_threshold
        self.norm_shift_threshold = norm_shift_threshold

    def compare_embeddings(
        self,
        embeddings_ref: Sequence[Sequence[float]],
        embeddings_curr: Sequence[Sequence[float]],
        top_k_dimensions: int = 5,
    ) -> EmbeddingDriftResult:
        """Evaluates drift between reference and production vector sets."""
        n_ref = len(embeddings_ref)
        n_curr = len(embeddings_curr)

        if n_ref == 0 or n_curr == 0:
            raise StatisticalTestError("Both embedding collections must contain at least 1 vector.")

        dim = len(embeddings_ref[0])
        if len(embeddings_curr[0]) != dim:
            raise StatisticalTestError(f"Embedding dimension mismatch: ref={dim}, curr={len(embeddings_curr[0])}")

        # 1. Compute Centroids and Norms
        centroid_ref = [0.0] * dim
        centroid_curr = [0.0] * dim
        norms_ref = []
        norms_curr = []

        dim_means_ref = [0.0] * dim
        dim_means_curr = [0.0] * dim

        for vec in embeddings_ref:
            norm_sq = 0.0
            for i in range(dim):
                v = float(vec[i])
                centroid_ref[i] += v
                norm_sq += v * v
            norms_ref.append(math.sqrt(norm_sq))

        for vec in embeddings_curr:
            norm_sq = 0.0
            for i in range(dim):
                v = float(vec[i])
                centroid_curr[i] += v
                norm_sq += v * v
            norms_curr.append(math.sqrt(norm_sq))

        centroid_ref = [c / n_ref for c in centroid_ref]
        centroid_curr = [c / n_curr for c in centroid_curr]

        mean_norm_ref = sum(norms_ref) / n_ref
        mean_norm_curr = sum(norms_curr) / n_curr

        norm_shift_ratio = (
            abs(mean_norm_curr - mean_norm_ref) / mean_norm_ref
            if mean_norm_ref > 1e-9 else 0.0
        )

        # 2. Cosine Distance between centroids
        cos_dist = cosine_distance_vectors(centroid_ref, centroid_curr)

        # 3. Identify top shifted dimensions
        dim_diffs = []
        for i in range(dim):
            diff = abs(centroid_curr[i] - centroid_ref[i])
            dim_diffs.append((i, diff, centroid_ref[i], centroid_curr[i]))

        dim_diffs.sort(key=lambda x: x[1], reverse=True)
        top_shifted = [
            {
                "dimension_index": item[0],
                "shift": round(item[1], 5),
                "ref_mean": round(item[2], 5),
                "curr_mean": round(item[3], 5),
            }
            for item in dim_diffs[:top_k_dimensions]
        ]

        has_drift = cos_dist >= self.cosine_threshold or norm_shift_ratio >= self.norm_shift_threshold

        if cos_dist >= self.cosine_threshold * 2.0:
            status = DriftStatus.CRITICAL_DRIFT
        elif cos_dist >= self.cosine_threshold:
            status = DriftStatus.MODERATE_DRIFT
        elif cos_dist >= self.cosine_threshold * 0.5:
            status = DriftStatus.LOW_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return EmbeddingDriftResult(
            cosine_centroid_distance=cos_dist,
            centroid_drift_status=status,
            dimension=dim,
            sample_count_ref=n_ref,
            sample_count_curr=n_curr,
            mean_norm_ref=mean_norm_ref,
            mean_norm_curr=mean_norm_curr,
            norm_shift_ratio=norm_shift_ratio,
            top_shifted_dimensions=top_shifted,
            has_drift=has_drift,
        )
