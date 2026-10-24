"""Configuration management and configurable statistical thresholds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from driftpulse.core.exceptions import ConfigurationError
from driftpulse.core.types import MetricType


@dataclass
class ThresholdConfig:
    """Configurable thresholds for statistical tests and alert triggers."""

    # Statistical significance level (alpha) for p-value based hypothesis tests (KS-test, Chi-Square)
    p_value_alpha: float = 0.05

    # Population Stability Index (PSI) boundaries
    psi_warning: float = 0.10
    psi_critical: float = 0.25

    # Distance metric thresholds
    wasserstein_threshold: float = 0.15
    jensen_shannon_threshold: float = 0.10
    tvd_threshold: float = 0.15
    cosine_distance_threshold: float = 0.12
    z_score_threshold: float = 2.0

    # Categorical novelty: Out-of-vocabulary rate threshold
    oov_rate_threshold: float = 0.05

    # Dataset level: proportion of drifted features to trigger dataset-level alert
    dataset_drift_share_threshold: float = 0.33

    # Missing value rate jump threshold (e.g., +10% missing values triggers warning)
    missing_rate_delta_threshold: float = 0.10

    # Default statistical metric to prioritize for numerical features
    default_numerical_metric: MetricType = MetricType.PSI

    # Default statistical metric to prioritize for categorical features
    default_categorical_metric: MetricType = MetricType.CHI_SQUARE

    # Custom feature-specific overrides: {"feature_name": {"psi_critical": 0.20}}
    feature_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def validate(self) -> None:
        """Validates all threshold values are within sound statistical bounds."""
        if not (0.0 < self.p_value_alpha < 0.5):
            raise ConfigurationError(f"p_value_alpha must be in (0, 0.5), got {self.p_value_alpha}")
        if self.psi_warning <= 0 or self.psi_critical <= self.psi_warning:
            raise ConfigurationError("psi_warning must be > 0 and psi_critical > psi_warning")
        if not (0.0 < self.dataset_drift_share_threshold <= 1.0):
            raise ConfigurationError("dataset_drift_share_threshold must be in (0, 1]")

    def get_feature_threshold(self, feature_name: str, metric_key: str, default_val: float) -> float:
        """Retrieves a metric threshold, applying feature-specific overrides if configured."""
        if feature_name in self.feature_overrides:
            return self.feature_overrides[feature_name].get(metric_key, default_val)
        return default_val

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["default_numerical_metric"] = self.default_numerical_metric.value
        data["default_categorical_metric"] = self.default_categorical_metric.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThresholdConfig":
        clean = dict(data)
        if "default_numerical_metric" in clean:
            clean["default_numerical_metric"] = MetricType(clean["default_numerical_metric"])
        if "default_categorical_metric" in clean:
            clean["default_categorical_metric"] = MetricType(clean["default_categorical_metric"])
        config = cls(**clean)
        config.validate()
        return config

    @classmethod
    def from_json_file(cls, filepath: str) -> "ThresholdConfig":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_json(self, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
