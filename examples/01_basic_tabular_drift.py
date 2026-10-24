"""Example 01: Basic Tabular Drift Detection."""

import random
from driftpulse import TabularDriftDetector, format_terminal_summary

# 1. Simulate baseline reference dataset
random.seed(42)
baseline_data = {
    "age": [random.gauss(38, 7) for _ in range(400)],
    "salary": [random.gauss(55000, 12000) for _ in range(400)],
    "department": random.choices(["ENGINEERING", "SALES", "MARKETING"], weights=[0.5, 0.3, 0.2], k=400),
}

# 2. Simulate production inference window with covariate drift in salary
production_data = {
    "age": [random.gauss(38.5, 7.1) for _ in range(400)],
    "salary": [random.gauss(78000, 19000) for _ in range(400)],  # Significant shift
    "department": random.choices(["ENGINEERING", "SALES", "MARKETING"], weights=[0.52, 0.28, 0.2], k=400),
}

# 3. Detect drift
detector = TabularDriftDetector()
report = detector.compare_datasets(baseline_data, production_data, dataset_name="HR_Salary_Model_Batch")

# 4. Print formatted terminal summary
print(format_terminal_summary(report))
