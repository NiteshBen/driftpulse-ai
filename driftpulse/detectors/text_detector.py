"""LLM prompt, retrieval, and response text distribution drift detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from driftpulse.core.exceptions import StatisticalTestError
from driftpulse.core.types import DriftStatus, MetricResult, MetricType
from driftpulse.statistical.ks_test import kolmogorov_smirnov_2sample


@dataclass
class TextDriftResult:
    """Consolidated drift diagnosis for unstructured text streams."""
    word_count_drift_status: DriftStatus
    ks_statistic: float
    p_value: float
    avg_words_ref: float
    avg_words_curr: float
    ttr_ref: float
    ttr_curr: float
    ttr_shift: float
    vocabulary_overlap_jaccard: float
    has_drift: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_count_drift_status": self.word_count_drift_status.value,
            "ks_statistic": round(self.ks_statistic, 6),
            "p_value": round(self.p_value, 6),
            "avg_words_ref": round(self.avg_words_ref, 2),
            "avg_words_curr": round(self.avg_words_curr, 2),
            "ttr_ref": round(self.ttr_ref, 4),
            "ttr_curr": round(self.ttr_curr, 4),
            "ttr_shift": round(self.ttr_shift, 4),
            "vocabulary_overlap_jaccard": round(self.vocabulary_overlap_jaccard, 4),
            "has_drift": self.has_drift,
            "details": self.details,
        }


class TextDriftDetector:
    """Monitors linguistic distribution shifts in GenAI and LLM prompt/response payloads."""

    def __init__(self, alpha: float = 0.05, ttr_shift_threshold: float = 0.15):
        self.alpha = alpha
        self.ttr_shift_threshold = ttr_shift_threshold

    def compare_texts(
        self,
        texts_ref: Sequence[str],
        texts_curr: Sequence[str],
    ) -> TextDriftResult:
        """Evaluates textual characteristics and structural shifts."""
        clean_ref = [str(t) for t in texts_ref if t is not None and str(t).strip() != ""]
        clean_curr = [str(t) for t in texts_curr if t is not None and str(t).strip() != ""]

        if not clean_ref or not clean_curr:
            raise StatisticalTestError("Both text corpora must contain non-empty strings.")

        # 1. Word Count Distributions
        wc_ref = [len(t.split()) for t in clean_ref]
        wc_curr = [len(t.split()) for t in clean_curr]

        ks_stat, p_val = kolmogorov_smirnov_2sample(wc_ref, wc_curr)
        avg_w_ref = sum(wc_ref) / len(wc_ref)
        avg_w_curr = sum(wc_curr) / len(wc_curr)

        # 2. Vocabulary & Type-Token Ratio
        vocab_ref = set()
        total_tokens_ref = 0
        for t in clean_ref:
            tokens = t.lower().split()
            vocab_ref.update(tokens)
            total_tokens_ref += len(tokens)

        vocab_curr = set()
        total_tokens_curr = 0
        for t in clean_curr:
            tokens = t.lower().split()
            vocab_curr.update(tokens)
            total_tokens_curr += len(tokens)

        ttr_ref = len(vocab_ref) / max(total_tokens_ref, 1)
        ttr_curr = len(vocab_curr) / max(total_tokens_curr, 1)
        ttr_shift = abs(ttr_curr - ttr_ref)

        # 3. Jaccard Vocabulary Overlap
        intersection = len(vocab_ref & vocab_curr)
        union = len(vocab_ref | vocab_curr)
        jaccard = intersection / max(union, 1)

        has_drift = p_val < self.alpha or ttr_shift >= self.ttr_shift_threshold

        if p_val < self.alpha / 5.0 or ttr_shift >= self.ttr_shift_threshold * 2.0:
            status = DriftStatus.CRITICAL_DRIFT
        elif has_drift:
            status = DriftStatus.MODERATE_DRIFT
        elif p_val < self.alpha * 2.0:
            status = DriftStatus.LOW_DRIFT
        else:
            status = DriftStatus.NO_DRIFT

        return TextDriftResult(
            word_count_drift_status=status,
            ks_statistic=ks_stat,
            p_value=p_val,
            avg_words_ref=avg_w_ref,
            avg_words_curr=avg_w_curr,
            ttr_ref=ttr_ref,
            ttr_curr=ttr_curr,
            ttr_shift=ttr_shift,
            vocabulary_overlap_jaccard=jaccard,
            has_drift=has_drift,
            details={
                "vocab_size_ref": len(vocab_ref),
                "vocab_size_curr": len(vocab_curr),
                "jaccard_overlap": round(jaccard, 4),
            },
        )
