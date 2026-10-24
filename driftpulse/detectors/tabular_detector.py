"""Tabular dataset drift detector."""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from driftpulse.core.config import ThresholdConfig
from driftpulse.core.exceptions import IncompatibleSchemaError
from driftpulse.core.profiler import DataProfiler
from driftpulse.core.types import (
    DatasetDriftReport,
    DriftStatus,
    FeatureDriftResult,
    FeatureProfile,
    FeatureType,
    MetricResult,
    MetricType,
)
from driftpulse.statistical import (
    chi_square_goodness_of_fit,
    compute_categorical_psi,
    compute_numerical_psi,
    compute_oov_rate,
    compute_z_score_shift,
    evaluate_chi_square_drift,
    evaluate_ks_drift,
    evaluate_psi_drift,
    evaluate_wasserstein_drift,
    total_variation_distance,
)


class TabularDriftDetector:
    """High-performance drift detector for structured tabular datasets."""

    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
        profiler: Optional[DataProfiler] = None,
    ):
        self.config = config or ThresholdConfig()
        self.profiler = profiler or DataProfiler()

    def compare_features(
        self,
        feature_name: str,
        ref_values: Sequence[Any],
        curr_values: Sequence[Any],
        feature_type: Optional[FeatureType] = None,
    ) -> FeatureDriftResult:
        """Evaluates distribution drift between two raw sequences for a single feature."""
        # Clean values
        clean_ref = [v for v in ref_values if v is not None and v != ""]
        clean_curr = [v for v in curr_values if v is not None and v != ""]

        ref_profile = self.profiler.profile_series(feature_name, ref_values, forced_type=feature_type)
        curr_profile = self.profiler.profile_series(feature_name, curr_values, forced_type=ref_profile.feature_type)

        f_type = ref_profile.feature_type
        all_metrics: Dict[str, MetricResult] = {}
        warnings: List[str] = []

        # Check missing rate jump
        missing_delta = curr_profile.missing_rate - ref_profile.missing_rate
        if missing_delta >= self.config.missing_rate_delta_threshold:
            warnings.append(
                f"Missing rate increased by {round(missing_delta * 100, 1)}% (Ref: {round(ref_profile.missing_rate*100, 1)}%, Curr: {round(curr_profile.missing_rate*100, 1)}%)"
            )

        if f_type == FeatureType.NUMERICAL:
            ref_nums = [float(x) for x in clean_ref]
            curr_nums = [float(x) for x in clean_curr]

            # 1. KS-Test
            ks_res = evaluate_ks_drift(ref_nums, curr_nums, alpha=self.config.p_value_alpha)
            all_metrics["ks_test"] = ks_res

            # 2. PSI
            psi_dict = compute_numerical_psi(ref_nums, curr_nums)
            psi_res = evaluate_psi_drift(
                psi_dict["psi"],
                warning_threshold=self.config.psi_warning,
                critical_threshold=self.config.psi_critical,
            )
            all_metrics["psi"] = psi_res

            # 3. Wasserstein
            wass_thresh = self.config.get_feature_threshold(feature_name, "wasserstein_threshold", self.config.wasserstein_threshold)
            wass_res = evaluate_wasserstein_drift(ref_nums, curr_nums, threshold=wass_thresh)
            all_metrics["wasserstein"] = wass_res

            # 4. Z-Score Mean Shift
            if ref_profile.mean is not None and ref_profile.std is not None and curr_profile.mean is not None:
                z_shift = compute_z_score_shift(ref_profile.mean, ref_profile.std, curr_profile.mean)
                all_metrics["z_score_shift"] = MetricResult(
                    metric_type=MetricType.Z_SCORE_SHIFT,
                    value=z_shift,
                    threshold=self.config.z_score_threshold,
                    has_drift=z_shift >= self.config.z_score_threshold,
                    status=DriftStatus.MODERATE_DRIFT if z_shift >= self.config.z_score_threshold else DriftStatus.NO_DRIFT,
                )

            # Determine primary metric
            if self.config.default_numerical_metric == MetricType.KS_TEST:
                primary_res = ks_res
            else:
                primary_res = psi_res

        elif f_type == FeatureType.CATEGORICAL:
            ref_counts: Dict[str, int] = ref_profile.extra.get("raw_counts", {})
            curr_counts: Dict[str, int] = curr_profile.extra.get("raw_counts", {})

            # 1. Chi-Square Test
            chi_res = evaluate_chi_square_drift(ref_counts, curr_counts, alpha=self.config.p_value_alpha)
            all_metrics["chi_square"] = chi_res

            # 2. Categorical PSI
            psi_cat = compute_categorical_psi(ref_counts, curr_counts)
            psi_res = evaluate_psi_drift(
                psi_cat["psi"],
                warning_threshold=self.config.psi_warning,
                critical_threshold=self.config.psi_critical,
            )
            all_metrics["psi"] = psi_res

            # 3. Total Variation Distance (TVD)
            tvd_val = total_variation_distance(ref_counts, curr_counts)
            all_metrics["tvd"] = MetricResult(
                metric_type=MetricType.TVD,
                value=tvd_val,
                threshold=self.config.tvd_threshold,
                has_drift=tvd_val >= self.config.tvd_threshold,
                status=DriftStatus.MODERATE_DRIFT if tvd_val >= self.config.tvd_threshold else DriftStatus.NO_DRIFT,
            )

            # 4. Out-of-Vocabulary (OOV) Rate
            ref_keys = set(ref_counts.keys())
            oov_rate, new_cats = compute_oov_rate(ref_keys, [str(x) for x in clean_curr])
            if oov_rate >= self.config.oov_rate_threshold:
                warnings.append(f"High Out-of-Vocabulary rate ({round(oov_rate*100, 1)}%): new categories {new_cats[:3]}")

            # Primary metric
            if self.config.default_categorical_metric == MetricType.PSI:
                primary_res = psi_res
            else:
                primary_res = chi_res
        else:
            # Fallback for text/embeddings in tabular mode
            primary_res = MetricResult(
                metric_type=MetricType.PSI,
                value=0.0,
                threshold=self.config.psi_warning,
                has_drift=False,
                status=DriftStatus.NO_DRIFT,
            )

        # Consolidate status: prioritize worst severity among key metrics
        max_severity = primary_res.status
        for m in all_metrics.values():
            if m.status == DriftStatus.CRITICAL_DRIFT:
                max_severity = DriftStatus.CRITICAL_DRIFT
                break
            elif m.status == DriftStatus.MODERATE_DRIFT and max_severity in (DriftStatus.NO_DRIFT, DriftStatus.LOW_DRIFT):
                max_severity = DriftStatus.MODERATE_DRIFT

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type=f_type,
            drift_status=max_severity,
            primary_metric=primary_res.metric_type,
            primary_value=primary_res.value,
            p_value=primary_res.p_value,
            all_metrics=all_metrics,
            baseline_profile=ref_profile,
            current_profile=curr_profile,
            warnings=warnings,
        )

    def compare_datasets(
        self,
        ref_data: Any,
        curr_data: Any,
        dataset_name: str = "Production_Dataset",
        feature_types: Optional[Dict[str, FeatureType]] = None,
    ) -> DatasetDriftReport:
        """Compares full reference and current datasets and returns a complete DatasetDriftReport."""
        # Extract columns
        ref_cols = self._extract_columns(ref_data)
        curr_cols = self._extract_columns(curr_data)

        common_cols = [c for c in ref_cols if c in curr_cols]
        missing_in_curr = [c for c in ref_cols if c not in curr_cols]
        extra_in_curr = [c for c in curr_cols if c not in ref_cols]

        if not common_cols:
            raise IncompatibleSchemaError(f"No shared features between datasets! Ref: {ref_cols}, Curr: {curr_cols}")

        feature_results: Dict[str, FeatureDriftResult] = {}
        drifted_count = 0

        for col in common_cols:
            ref_vals = self._get_column_values(ref_data, col)
            curr_vals = self._get_column_values(curr_data, col)
            forced_t = feature_types.get(col) if feature_types else None

            res = self.compare_features(col, ref_vals, curr_vals, feature_type=forced_t)
            feature_results[col] = res
            if res.is_drifted:
                drifted_count += 1

        total_features = len(common_cols)
        drift_share = drifted_count / total_features

        # Dataset level status
        if drift_share >= self.config.dataset_drift_share_threshold:
            overall_status = DriftStatus.CRITICAL_DRIFT
        elif drift_share >= self.config.dataset_drift_share_threshold / 2.0:
            overall_status = DriftStatus.MODERATE_DRIFT
        elif drifted_count > 0:
            overall_status = DriftStatus.LOW_DRIFT
        else:
            overall_status = DriftStatus.NO_DRIFT

        # Build alerts list
        alerts = []
        if overall_status.is_drifted:
            alerts.append({
                "level": "CRITICAL" if overall_status == DriftStatus.CRITICAL_DRIFT else "WARNING",
                "message": f"Dataset drift threshold breached! {drifted_count}/{total_features} features ({round(drift_share*100, 1)}%) drifted.",
            })

        for col, r in feature_results.items():
            if r.drift_status == DriftStatus.CRITICAL_DRIFT:
                alerts.append({
                    "level": "CRITICAL",
                    "feature": col,
                    "message": f"Critical drift in '{col}' ({r.primary_metric.value}={r.primary_value})",
                })
            for w in r.warnings:
                alerts.append({
                    "level": "WARNING",
                    "feature": col,
                    "message": w,
                })

        return DatasetDriftReport(
            dataset_name=dataset_name,
            timestamp=datetime.datetime.now().isoformat(),
            total_features=total_features,
            drifted_features=drifted_count,
            drift_share=drift_share,
            overall_status=overall_status,
            feature_results=feature_results,
            metadata={
                "missing_in_current": missing_in_curr,
                "extra_in_current": extra_in_curr,
                "threshold_config": self.config.to_dict(),
            },
            alerts=alerts,
        )

    def _extract_columns(self, data: Any) -> List[str]:
        if hasattr(data, "columns"):
            return [str(c) for c in data.columns]
        if isinstance(data, dict):
            return [str(k) for k in data.keys()]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return [str(k) for k in data[0].keys()]
        raise IncompatibleSchemaError(f"Cannot extract columns from data format: {type(data)}")

    def _get_column_values(self, data: Any, col: str) -> List[Any]:
        if hasattr(data, "__getitem__"):
            col_data = data[col]
            if hasattr(col_data, "tolist"):
                return col_data.tolist()
            if isinstance(col_data, (list, tuple)):
                return list(col_data)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return [row.get(col) for row in data]
        raise IncompatibleSchemaError(f"Cannot retrieve column '{col}' from data format")
