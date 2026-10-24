"""Scikit-learn model wrapper and inference telemetry interceptor."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from driftpulse.core.config import ThresholdConfig
from driftpulse.core.types import DatasetDriftReport
from driftpulse.detectors.tabular_detector import TabularDriftDetector
from driftpulse.integrations.pandas_adapter import to_standard_dict


class SklearnDriftMonitor:
    """Wraps an ML estimator to automatically log inference batches and check for covariate drift."""

    def __init__(
        self,
        estimator: Any,
        baseline_data: Any,
        feature_names: Optional[Sequence[str]] = None,
        config: Optional[ThresholdConfig] = None,
        buffer_size: int = 100,
    ):
        self.estimator = estimator
        self.baseline_dict = to_standard_dict(baseline_data)
        self.feature_names = list(feature_names) if feature_names else list(self.baseline_dict.keys())
        self.config = config or ThresholdConfig()
        self.detector = TabularDriftDetector(config=self.config)
        self.buffer_size = buffer_size

        self._inference_buffer: Dict[str, List[Any]] = {f: [] for f in self.feature_names}
        self.reports: List[DatasetDriftReport] = []

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Executes model inference and captures input feature distributions."""
        self._record_inputs(X)
        return self.estimator.predict(X, **kwargs)

    def predict_proba(self, X: Any, **kwargs: Any) -> Any:
        """Executes model probability inference and captures input feature distributions."""
        self._record_inputs(X)
        return self.estimator.predict_proba(X, **kwargs)

    def _record_inputs(self, X: Any) -> None:
        """Appends observations into buffer and runs evaluation when buffer fills."""
        try:
            curr_dict = to_standard_dict(X)
            for f in self.feature_names:
                if f in curr_dict:
                    self._inference_buffer[f].extend(curr_dict[f])
        except Exception:
            # Fallback if 2D list/array
            if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], (list, tuple)):
                for row in X:
                    for idx, val in enumerate(row):
                        if idx < len(self.feature_names):
                            self._inference_buffer[self.feature_names[idx]].append(val)

        # Check if buffer reached threshold
        sample_len = len(self._inference_buffer[self.feature_names[0]]) if self.feature_names else 0
        if sample_len >= self.buffer_size:
            self.check_drift()

    def check_drift(self) -> DatasetDriftReport:
        """Forces an immediate drift evaluation on buffered observations and flushes buffer."""
        curr_dict = {f: list(vals) for f, vals in self._inference_buffer.items() if len(vals) > 0}
        report = self.detector.compare_datasets(
            ref_data=self.baseline_dict,
            curr_data=curr_dict,
            dataset_name="Inference_Buffer_Batch",
        )
        self.reports.append(report)
        # Reset buffer
        self._inference_buffer = {f: [] for f in self.feature_names}
        return report
