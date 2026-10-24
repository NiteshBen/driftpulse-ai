"""Example 03: Continuous Monitoring Pipeline with Alerts."""

import random
from driftpulse import MonitoringPipeline

random.seed(42)

baseline_stream = {
    "latency_ms": [random.gauss(120, 15) for _ in range(300)],
    "error_rate": [random.betavariate(1, 50) for _ in range(300)],
}

def on_alert(alert):
    print(f"\n[TRIGGERED ALERT]: {alert.title}")
    print(f"Details: {alert.message}")

pipeline = MonitoringPipeline(baseline_data=baseline_stream, on_drift_alert=on_alert)

# Evaluate stable batch
stable_batch = {
    "latency_ms": [random.gauss(121, 14.8) for _ in range(100)],
    "error_rate": [random.betavariate(1, 48) for _ in range(100)],
}
r1 = pipeline.evaluate_batch(stable_batch, dataset_name="Batch_001_Morning")
print(f"Batch 001 Status: {r1.overall_status.value}")

# Evaluate degraded batch
degraded_batch = {
    "latency_ms": [random.gauss(240, 45) for _ in range(100)],  # Severe latency spike
    "error_rate": [random.betavariate(5, 20) for _ in range(100)],  # Severe error spike
}
r2 = pipeline.evaluate_batch(degraded_batch, dataset_name="Batch_002_Evening_Incident")
print(f"Batch 002 Status: {r2.overall_status.value}")

print("\nPipeline Telemetry Summary:", pipeline.get_summary_metrics())
