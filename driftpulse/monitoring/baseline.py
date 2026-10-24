"""Baseline statistical profile storage and serialization."""

from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, Optional

from driftpulse.core.exceptions import ProfilingError
from driftpulse.core.profiler import DataProfiler
from driftpulse.core.types import FeatureProfile, FeatureType


class BaselineStore:
    """Manages creation, serialization, and versioning of baseline feature profiles."""

    def __init__(self, profiler: Optional[DataProfiler] = None):
        self.profiler = profiler or DataProfiler()

    def create_baseline(
        self,
        dataset: Any,
        baseline_name: str = "default_baseline",
        version: str = "1.0.0",
        feature_types: Optional[Dict[str, FeatureType]] = None,
    ) -> Dict[str, Any]:
        """Profiles a reference dataset and wraps it in a versioned baseline document."""
        profiles = self.profiler.profile_dataset(dataset, feature_types=feature_types)
        return {
            "baseline_name": baseline_name,
            "version": version,
            "created_at": datetime.datetime.now().isoformat(),
            "total_features": len(profiles),
            "features": {name: p.to_dict() for name, p in profiles.items()},
        }

    def save(self, baseline_doc: Dict[str, Any], filepath: str) -> None:
        """Saves a baseline profile document to disk as JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(baseline_doc, f, indent=2)

    def load(self, filepath: str) -> Dict[str, Any]:
        """Loads a baseline document from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Baseline file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def reconstruct_profiles(self, baseline_doc: Dict[str, Any]) -> Dict[str, FeatureProfile]:
        """Reconstructs strongly-typed FeatureProfile objects from a saved baseline document."""
        profiles = {}
        for name, data in baseline_doc.get("features", {}).items():
            profiles[name] = FeatureProfile(
                name=data["name"],
                feature_type=FeatureType(data["feature_type"]),
                count=data["count"],
                missing_count=data.get("missing_count", 0),
                missing_rate=data.get("missing_rate", 0.0),
                mean=data.get("mean"),
                std=data.get("std"),
                min_val=data.get("min_val"),
                max_val=data.get("max_val"),
                quantiles=data.get("quantiles", {}),
                frequencies=data.get("frequencies", {}),
                histogram=data.get("histogram", {}),
                extra=data.get("extra", {}),
            )
        return profiles
