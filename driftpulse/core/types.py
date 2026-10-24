"""Core types, data models, and enumerations for DriftPulse-AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FeatureType(str, Enum):
    """Supported feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"


class DriftStatus(str, Enum):
    """Categorical classification of distribution drift severity."""
    NO_DRIFT = "no_drift"
    LOW_DRIFT = "low_drift"
    MODERATE_DRIFT = "moderate_drift"
    CRITICAL_DRIFT = "critical_drift"

    @property
    def is_drifted(self) -> bool:
        return self in (DriftStatus.MODERATE_DRIFT, DriftStatus.CRITICAL_DRIFT)

    @property
    def badge_color(self) -> str:
        if self == DriftStatus.NO_DRIFT:
            return "#10b981"  # Emerald green
        if self == DriftStatus.LOW_DRIFT:
            return "#3b82f6"  # Blue
        if self == DriftStatus.MODERATE_DRIFT:
            return "#f59e0b"  # Amber
        return "#ef4444"      # Crimson red


class MetricType(str, Enum):
    """Statistical drift distance and significance testing metrics."""
    KS_TEST = "ks_test"
    PSI = "psi"
    WASSERSTEIN = "wasserstein"
    JENSEN_SHANNON = "jensen_shannon"
    CHI_SQUARE = "chi_square"
    TVD = "total_variation_distance"
    COSINE_DISTANCE = "cosine_distance"
    Z_SCORE_SHIFT = "z_score_shift"
    LEXICAL_DIVERSITY = "lexical_diversity"


class AlertLevel(str, Enum):
    """Severity tier for monitoring notifications."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class MetricResult:
    """Individual statistical test outcome for a single metric."""
    metric_type: MetricType
    value: float
    threshold: float
    p_value: Optional[float] = None
    has_drift: bool = False
    status: DriftStatus = DriftStatus.NO_DRIFT
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_type": self.metric_type.value,
            "value": round(self.value, 6),
            "threshold": self.threshold,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "has_drift": self.has_drift,
            "status": self.status.value,
            "details": self.details,
        }


@dataclass
class FeatureProfile:
    """Statistical summary profile for a single feature distribution."""
    name: str
    feature_type: FeatureType
    count: int
    missing_count: int = 0
    missing_rate: float = 0.0
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    quantiles: Dict[str, float] = field(default_factory=dict)  # e.g., "p25", "p50", "p75"
    frequencies: Dict[str, float] = field(default_factory=dict)  # categorical frequency distribution
    histogram: Dict[str, Any] = field(default_factory=dict)  # bin edges and counts for charting
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "count": self.count,
            "missing_count": self.missing_count,
            "missing_rate": round(self.missing_rate, 4),
            "mean": round(self.mean, 4) if self.mean is not None else None,
            "std": round(self.std, 4) if self.std is not None else None,
            "min_val": round(self.min_val, 4) if self.min_val is not None else None,
            "max_val": round(self.max_val, 4) if self.max_val is not None else None,
            "quantiles": {k: round(v, 4) for k, v in self.quantiles.items()},
            "frequencies": {k: round(v, 4) for k, v in self.frequencies.items()},
            "histogram": self.histogram,
            "extra": self.extra,
        }


@dataclass
class FeatureDriftResult:
    """Consolidated drift diagnosis for an individual feature."""
    feature_name: str
    feature_type: FeatureType
    drift_status: DriftStatus
    primary_metric: MetricType
    primary_value: float
    p_value: Optional[float] = None
    all_metrics: Dict[str, MetricResult] = field(default_factory=dict)
    baseline_profile: Optional[FeatureProfile] = None
    current_profile: Optional[FeatureProfile] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_drifted(self) -> bool:
        return self.drift_status.is_drifted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "feature_type": self.feature_type.value,
            "drift_status": self.drift_status.value,
            "primary_metric": self.primary_metric.value,
            "primary_value": round(self.primary_value, 6),
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "is_drifted": self.is_drifted,
            "all_metrics": {k: v.to_dict() for k, v in self.all_metrics.items()},
            "baseline_profile": self.baseline_profile.to_dict() if self.baseline_profile else None,
            "current_profile": self.current_profile.to_dict() if self.current_profile else None,
            "warnings": self.warnings,
        }


@dataclass
class DatasetDriftReport:
    """Holistic drift report evaluating an entire dataset against a baseline."""
    dataset_name: str
    timestamp: str
    total_features: int
    drifted_features: int
    drift_share: float
    overall_status: DriftStatus
    feature_results: Dict[str, FeatureDriftResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_dataset_drift(self) -> bool:
        return self.overall_status.is_drifted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_share": round(self.drift_share, 4),
            "overall_status": self.overall_status.value,
            "has_dataset_drift": self.has_dataset_drift,
            "feature_results": {k: v.to_dict() for k, v in self.feature_results.items()},
            "metadata": self.metadata,
            "alerts": self.alerts,
        }
