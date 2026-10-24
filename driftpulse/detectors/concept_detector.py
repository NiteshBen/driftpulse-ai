"""Concept, prediction, and unlabelled model confidence degradation detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType
from driftpulse.statistical.ks_test import kolmogorov_smirnov_2sample
from driftpulse.statistical.psi import compute_numerical_psi, evaluate_psi_drift


@dataclass
class ConceptDriftResult:
    """Diagnosis of model output degradation and confidence shift."""
    prediction_drift_status: DriftStatus
    prediction_psi: float
    ks_p_value: float
    mean_confidence_ref: float
    mean_confidence_curr: float
    confidence_decay: float
    mean_entropy_ref: float
    mean_entropy_curr: float
    entropy_increase: float
    has_drift: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_drift_status": self.prediction_drift_status.value,
            "prediction_psi": round(self.prediction_psi, 6),
            "ks_p_value": round(self.ks_p_value, 6),
            "mean_confidence_ref": round(self.mean_confidence_ref, 4),
            "mean_confidence_curr": round(self.mean_confidence_curr, 4),
            "confidence_decay": round(self.confidence_decay, 4),
            "mean_entropy_ref": round(self.mean_entropy_ref, 4),
            "mean_entropy_curr": round(self.mean_entropy_curr, 4),
            "entropy_increase": round(self.entropy_increase, 4),
            "has_drift": self.has_drift,
            "details": self.details,
        }


class ConceptDriftDetector:
    """Monitors model inference outputs, confidence margins, and prediction entropy."""

    def __init__(
        self,
        psi_threshold: float = 0.10,
        confidence_decay_threshold: float = 0.08,
    ):
        self.psi_threshold = psi_threshold
        self.confidence_decay_threshold = confidence_decay_threshold

    def compare_predictions(
        self,
        predictions_ref: Sequence[float],
        predictions_curr: Sequence[float],
    ) -> ConceptDriftResult:
        """Evaluates drift in continuous scalar predictions (e.g. regression or class probability)."""
        clean_ref = [float(x) for x in predictions_ref if x is not None and not math.isnan(float(x))]
        clean_curr = [float(x) for x in predictions_curr if x is not None and not math.isnan(float(x))]

        if not clean_ref or not clean_curr:
            raise StatisticalTestError("Both prediction series must contain valid numeric outputs.")

        # 1. Prediction distribution PSI & KS
        psi_dict = compute_numerical_psi(clean_ref, clean_curr)
        psi_val = psi_dict["psi"]
        _, ks_p_val = kolmogorov_smirnov_2sample(clean_ref, clean_curr)

        # 2. Confidence metrics
        mean_ref = sum(clean_ref) / len(clean_ref)
        mean_curr = sum(clean_curr) / len(clean_curr)
        conf_decay = max(0.0, mean_ref - mean_curr)

        # Shannon Entropy proxy for binary/scalar probabilities in [0, 1]
        def mean_entropy(probs: List[float]) -> float:
            ent_sum = 0.0
            count = 0
            for p in probs:
                p_clamped = max(1e-6, min(1.0 - 1e-6, p))
                h = - (p_clamped * math.log2(p_clamped) + (1.0 - p_clamped) * math.log2(1.0 - p_clamped))
                ent_sum += h
                count += 1
            return ent_sum / max(count, 1)

        ent_ref = mean_entropy(clean_ref)
        ent_curr = mean_entropy(clean_curr)
        ent_inc = max(0.0, ent_curr - ent_ref)

        psi_res = evaluate_psi_drift(psi_val, warning_threshold=self.psi_threshold)
        has_drift = psi_res.has_drift or (conf_decay >= self.confidence_decay_threshold)

        if psi_res.status == DriftStatus.CRITICAL_DRIFT or conf_decay >= self.confidence_decay_threshold * 2.0:
            status = DriftStatus.CRITICAL_DRIFT
        elif has_drift:
            status = DriftStatus.MODERATE_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return ConceptDriftResult(
            prediction_drift_status=status,
            prediction_psi=psi_val,
            ks_p_value=ks_p_val,
            mean_confidence_ref=mean_ref,
            mean_confidence_curr=mean_curr,
            confidence_decay=conf_decay,
            mean_entropy_ref=ent_ref,
            mean_entropy_curr=ent_curr,
            entropy_increase=ent_inc,
            has_drift=has_drift,
            details={
                "psi_interpretation": "Prediction distribution has significantly shifted" if psi_res.has_drift else "Prediction distribution is stable",
                "entropy_alert": "Model uncertainty has escalated" if ent_inc > 0.15 else "Confidence entropy stable",
            },
        )
