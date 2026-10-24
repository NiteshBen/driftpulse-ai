"""Script to generate a full-featured executive dark-mode HTML report."""

import os
import random
from driftpulse import TabularDriftDetector, generate_html_report

def main():
    random.seed(1337)
    n = 600

    # Realistic FinTech Credit Underwriting Dataset
    baseline = {
        "applicant_age": [random.gauss(42, 9) for _ in range(n)],
        "annual_income": [random.gauss(72000, 18000) for _ in range(n)],
        "debt_to_income": [random.betavariate(2, 5) * 60 for _ in range(n)],
        "loan_amount": [random.gauss(25000, 8000) for _ in range(n)],
        "employment_status": random.choices(["SALARIED", "SELF_EMPLOYED", "CONTRACT"], weights=[0.70, 0.20, 0.10], k=n),
        "home_ownership": random.choices(["MORTGAGE", "RENT", "OWN"], weights=[0.55, 0.35, 0.10], k=n),
    }

    # Production inference batch under macroeconomic regime shift
    production = {
        "applicant_age": [random.gauss(42.3, 9.1) for _ in range(n)],  # Stable
        "annual_income": [random.gauss(89000, 24000) for _ in range(n)],  # Drifted (Inflation/Shift)
        "debt_to_income": [random.betavariate(3, 4) * 65 for _ in range(n)],  # Drifted
        "loan_amount": [random.gauss(34000, 11000) for _ in range(n)],  # Drifted (Higher loans)
        "employment_status": random.choices(["SALARIED", "SELF_EMPLOYED", "CONTRACT", "GIG_WORKER"], weights=[0.50, 0.25, 0.15, 0.10], k=n),  # Novel category
        "home_ownership": random.choices(["MORTGAGE", "RENT", "OWN"], weights=[0.53, 0.37, 0.10], k=n),  # Stable
    }

    detector = TabularDriftDetector()
    report = detector.compare_datasets(baseline, production, dataset_name="FinTech_CreditUnderwriting_Q4")

    html_content = generate_html_report(report, title="FinTech Credit Model & Covariate Drift Sentinel")
    out_file = os.path.join(os.path.dirname(__file__), "..", "driftpulse_executive_report.html")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[+] Sample Executive HTML Report written to: {os.path.abspath(out_file)}")

if __name__ == "__main__":
    main()
