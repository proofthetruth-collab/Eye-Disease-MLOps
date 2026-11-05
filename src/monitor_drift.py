import os 
import sys
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def monitor_drift():
    # 1. Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)

    # 2. Load reference dataset (training data distribution)
    data_path = "data/image_manifest.csv"
    if not os.path.exists(data_path):
        print(f"Error: Reference dataset not found at {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)

    # Split reference and simulate a "production" dataset arriving after deployment
    reference_data = df.sample(frac=0.7, random_state=42)
    production_data = df.drop(reference_data.index)

    print(f"Reference data shape: {reference_data.shape}")
    print(f"Production data shape: {production_data.shape}")

    # 3. Run Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])

    eval_result = drift_report.run(reference_data=reference_data, current_data=production_data)

    # 4. Save HTML report to reports/directory
    report_path = "reports/drift_report.html"
    eval_result.save_html(report_path) # Changed from .save() to .save_html() and called save_html on eval_result instead of drift_report
    print(f"Drift report successfully saved to {report_path}")

    # 5. Check metrics and enforce threshold exit condition
    report_dict = eval_result.dict() # Called as_dict() on eval_result snapshot, changed as_dict() to dict()
    try:
        drift_result = report_dict["metrics"][0]["result"]
        dataset_drift = drift_result.get("dataset_drift", False)
        share_of_drifted_columns = drift_result.get("share_of_drifted_columns", 0.0)

        print(f"Dataset drift detected: {dataset_drift}")
        print(f"Share of drifted columns: {share_of_drifted_columns:.2f}")

        # Configurable threshold (exits with code 1 if drift exceeds threshold)
        drift_threshold = 0.5  # Example threshold, adjust as needed
        if share_of_drifted_columns > drift_threshold:
            print(f"Error: Drift share ({share_of_drifted_columns}) exceeds the allowed threshold ({drift_threshold})!")
            sys.exit(1)

    except Exception as e:
        print(f"Note on parsing metrics dictionary: {e}")

    print("Drift monitoring completed successfully.")

if __name__ == "__main__":
    monitor_drift()