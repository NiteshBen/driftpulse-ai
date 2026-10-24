"""Dataset profiling and statistical summary extraction."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from driftpulse.core.exceptions import ProfilingError
from driftpulse.core.types import FeatureProfile, FeatureType


def is_number(val: Any) -> bool:
    """Checks whether a value is numeric and non-boolean."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return not (math.isnan(val) or math.isinf(val))
    return False


def infer_feature_type(values: Sequence[Any], max_category_cardinality: int = 20) -> FeatureType:
    """Infers the FeatureType for a sequence of observations."""
    clean = [v for v in values if v is not None and v != ""]
    if not clean:
        return FeatureType.CATEGORICAL

    # Check for embedding (list/tuple of floats)
    if isinstance(clean[0], (list, tuple)) and len(clean[0]) > 0 and isinstance(clean[0][0], (int, float)):
        return FeatureType.EMBEDDING

    # Check numeric
    numeric_count = sum(1 for v in clean if is_number(v))
    if numeric_count / len(clean) >= 0.90:
        # Check if low-cardinality integers are actually categorical codes
        unique_vals = set(clean)
        if len(unique_vals) <= 2 and all(v in (0, 1, 0.0, 1.0) for v in unique_vals):
            return FeatureType.CATEGORICAL
        return FeatureType.NUMERICAL

    # Check text vs categorical
    text_lengths = [len(str(v)) for v in clean]
    avg_len = sum(text_lengths) / len(text_lengths)
    unique_ratio = len(set(clean)) / len(clean)

    if avg_len > 40 or (unique_ratio > 0.8 and len(clean) > 50 and avg_len > 25):
        return FeatureType.TEXT

    return FeatureType.CATEGORICAL


class DataProfiler:
    """High-performance statistical profiler for raw and tabular data."""

    def __init__(self, n_histogram_bins: int = 10, max_category_cardinality: int = 25):
        self.n_histogram_bins = n_histogram_bins
        self.max_category_cardinality = max_category_cardinality

    def profile_series(
        self,
        name: str,
        values: Sequence[Any],
        forced_type: Optional[FeatureType] = None,
    ) -> FeatureProfile:
        """Profiles a single 1D sequence of observations."""
        total_count = len(values)
        if total_count == 0:
            raise ProfilingError(f"Cannot profile empty feature series: '{name}'")

        # Missingness computation
        missing_count = sum(
            1 for v in values
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) or v == ""
        )
        missing_rate = missing_count / total_count
        clean_values = [
            v for v in values
            if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) and v != ""
        ]

        feature_type = forced_type or infer_feature_type(clean_values, self.max_category_cardinality)

        if not clean_values:
            return FeatureProfile(
                name=name,
                feature_type=feature_type,
                count=total_count,
                missing_count=missing_count,
                missing_rate=missing_rate,
            )

        if feature_type == FeatureType.NUMERICAL:
            return self._profile_numerical(name, clean_values, total_count, missing_count, missing_rate)
        elif feature_type == FeatureType.CATEGORICAL:
            return self._profile_categorical(name, clean_values, total_count, missing_count, missing_rate)
        elif feature_type == FeatureType.TEXT:
            return self._profile_text(name, clean_values, total_count, missing_count, missing_rate)
        else:  # EMBEDDING
            return self._profile_embedding(name, clean_values, total_count, missing_count, missing_rate)

    def _profile_numerical(
        self,
        name: str,
        clean_values: List[Any],
        total_count: int,
        missing_count: int,
        missing_rate: float,
    ) -> FeatureProfile:
        nums = sorted([float(x) for x in clean_values])
        n = len(nums)
        mean_val = sum(nums) / n
        variance = sum((x - mean_val) ** 2 for x in nums) / max(n - 1, 1)
        std_val = math.sqrt(variance)

        min_val = nums[0]
        max_val = nums[-1]

        def get_quantile(p: float) -> float:
            idx = p * (n - 1)
            lower = int(idx)
            upper = min(lower + 1, n - 1)
            weight = idx - lower
            return nums[lower] * (1.0 - weight) + nums[upper] * weight

        quantiles = {
            "p01": get_quantile(0.01),
            "p05": get_quantile(0.05),
            "p25": get_quantile(0.25),
            "p50": get_quantile(0.50),
            "p75": get_quantile(0.75),
            "p95": get_quantile(0.95),
            "p99": get_quantile(0.99),
        }

        # Histogram binning
        histogram: Dict[str, Any] = {"bin_edges": [], "counts": []}
        if min_val == max_val:
            histogram["bin_edges"] = [min_val - 0.5, max_val + 0.5]
            histogram["counts"] = [n]
        else:
            step = (max_val - min_val) / self.n_histogram_bins
            edges = [min_val + i * step for i in range(self.n_histogram_bins + 1)]
            counts = [0] * self.n_histogram_bins
            for val in nums:
                bin_idx = int((val - min_val) / step)
                if bin_idx >= self.n_histogram_bins:
                    bin_idx = self.n_histogram_bins - 1
                counts[bin_idx] += 1
            histogram["bin_edges"] = [round(e, 4) for e in edges]
            histogram["counts"] = counts

        return FeatureProfile(
            name=name,
            feature_type=FeatureType.NUMERICAL,
            count=total_count,
            missing_count=missing_count,
            missing_rate=missing_rate,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            quantiles=quantiles,
            histogram=histogram,
        )

    def _profile_categorical(
        self,
        name: str,
        clean_values: List[Any],
        total_count: int,
        missing_count: int,
        missing_rate: float,
    ) -> FeatureProfile:
        counts: Dict[str, int] = {}
        for v in clean_values:
            s = str(v)
            counts[s] = counts.get(s, 0) + 1

        n_clean = len(clean_values)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        frequencies: Dict[str, float] = {}
        top_cats = sorted_counts[: self.max_category_cardinality]
        for cat, cnt in top_cats:
            frequencies[cat] = round(cnt / n_clean, 4)

        if len(sorted_counts) > self.max_category_cardinality:
            other_cnt = sum(cnt for _, cnt in sorted_counts[self.max_category_cardinality :])
            frequencies["__OTHER__"] = round(other_cnt / n_clean, 4)

        return FeatureProfile(
            name=name,
            feature_type=FeatureType.CATEGORICAL,
            count=total_count,
            missing_count=missing_count,
            missing_rate=missing_rate,
            frequencies=frequencies,
            extra={"unique_categories": len(counts), "raw_counts": counts},
        )

    def _profile_text(
        self,
        name: str,
        clean_values: List[Any],
        total_count: int,
        missing_count: int,
        missing_rate: float,
    ) -> FeatureProfile:
        lengths = [len(str(v).split()) for v in clean_values]  # Word counts
        char_lengths = [len(str(v)) for v in clean_values]
        n = len(lengths)
        avg_words = sum(lengths) / n
        avg_chars = sum(char_lengths) / n

        # Lexical diversity sample
        vocab = set()
        total_tokens = 0
        for v in clean_values[:500]:
            words = str(v).lower().split()
            vocab.update(words)
            total_tokens += len(words)
        ttr = len(vocab) / max(total_tokens, 1)

        return FeatureProfile(
            name=name,
            feature_type=FeatureType.TEXT,
            count=total_count,
            missing_count=missing_count,
            missing_rate=missing_rate,
            mean=avg_words,
            min_val=float(min(lengths)),
            max_val=float(max(lengths)),
            extra={
                "avg_character_length": round(avg_chars, 2),
                "type_token_ratio": round(ttr, 4),
                "vocabulary_size": len(vocab),
            },
        )

    def _profile_embedding(
        self,
        name: str,
        clean_values: List[Any],
        total_count: int,
        missing_count: int,
        missing_rate: float,
    ) -> FeatureProfile:
        dim = len(clean_values[0])
        # Compute centroid vector
        centroid = [0.0] * dim
        norms = []
        for vec in clean_values:
            norm_sq = 0.0
            for i in range(min(dim, len(vec))):
                val = float(vec[i])
                centroid[i] += val
                norm_sq += val * val
            norms.append(math.sqrt(norm_sq))

        n = len(clean_values)
        centroid = [round(c / n, 6) for c in centroid]
        avg_norm = sum(norms) / n

        return FeatureProfile(
            name=name,
            feature_type=FeatureType.EMBEDDING,
            count=total_count,
            missing_count=missing_count,
            missing_rate=missing_rate,
            mean=avg_norm,
            extra={
                "dimension": dim,
                "centroid": centroid,
                "mean_vector_norm": round(avg_norm, 4),
            },
        )

    def profile_dataset(
        self,
        data: Union[Dict[str, Sequence[Any]], Any],
        feature_types: Optional[Dict[str, FeatureType]] = None,
    ) -> Dict[str, FeatureProfile]:
        """Profiles all features in a dictionary or DataFrame."""
        profiles = {}
        # Support pandas DataFrame without mandatory import
        if hasattr(data, "columns") and hasattr(data, "to_dict"):
            for col in data.columns:
                series = data[col].tolist()
                f_type = feature_types.get(str(col)) if feature_types else None
                profiles[str(col)] = self.profile_series(str(col), series, forced_type=f_type)
        elif isinstance(data, dict):
            for col, series in data.items():
                f_type = feature_types.get(str(col)) if feature_types else None
                profiles[str(col)] = self.profile_series(str(col), series, forced_type=f_type)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # List of row dictionaries
            cols = list(data[0].keys())
            for col in cols:
                series = [row.get(col) for row in data]
                f_type = feature_types.get(str(col)) if feature_types else None
                profiles[str(col)] = self.profile_series(str(col), series, forced_type=f_type)
        else:
            raise ProfilingError(f"Unsupported data format: {type(data)}")

        return profiles
