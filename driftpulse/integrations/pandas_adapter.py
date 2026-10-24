"""Data format adapter with zero mandatory external dependency."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Optional


def load_csv_as_dict(filepath: str) -> Dict[str, List[Any]]:
    """Loads a CSV file into a column-oriented dictionary of values with automatic type casting."""
    data: Dict[str, List[Any]] = {}

    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col, raw_val in row.items():
                if col not in data:
                    data[col] = []

                if raw_val is None or raw_val.strip() == "":
                    data[col].append(None)
                    continue

                v = raw_val.strip()
                # Try int
                try:
                    casted = int(v)
                    data[col].append(casted)
                    continue
                except ValueError:
                    pass

                # Try float
                try:
                    casted = float(v)
                    data[col].append(casted)
                    continue
                except ValueError:
                    pass

                # Fallback to string
                data[col].append(v)

    return data


def to_standard_dict(data: Any) -> Dict[str, List[Any]]:
    """Normalizes DataFrame, list of dicts, or CSV path to a standard column-dictionary."""
    if isinstance(data, str) and data.endswith(".csv"):
        return load_csv_as_dict(data)

    if hasattr(data, "to_dict"):
        # Pandas DataFrame
        return {str(col): data[col].tolist() for col in data.columns}

    if isinstance(data, dict):
        return {str(k): list(v) for k, v in data.items()}

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        cols = list(data[0].keys())
        res: Dict[str, List[Any]] = {str(c): [] for c in cols}
        for row in data:
            for c in cols:
                res[str(c)].append(row.get(c))
        return res

    raise ValueError(f"Cannot convert data of type {type(data)} to column dictionary")
