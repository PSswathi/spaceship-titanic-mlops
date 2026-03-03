"""
monitor.py — Phase 6: Evidently Drift Monitoring
Spaceship Titanic MLOps Pipeline

Generates:
  - Data drift report  (HTML)
  - Model performance report (HTML)
  - Drift summary (JSON) for CI/CD alerting
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.report import Report

# ── Logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import cfg

# ── Paths ─────────────────────────────────────
REFERENCE_DATA_PATH = Path("data/processed/train_engineered.csv")
CURRENT_DATA_PATH   = Path("data/processed/test_engineered.csv")
REPORTS_DIR         = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference (train) and current (test) engineered datasets."""
    logger.info("Loading reference data from %s", REFERENCE_DATA_PATH)
    reference = pd.read_csv(REFERENCE_DATA_PATH)

    logger.info("Loading current data from %s", CURRENT_DATA_PATH)
    current = pd.read_csv(CURRENT_DATA_PATH)

    logger.info("Reference shape: %s | Current shape: %s", reference.shape, current.shape)
    return reference, current


# ─────────────────────────────────────────────
# 2. Column Mapping
# ─────────────────────────────────────────────
def get_column_mapping(reference: pd.DataFrame, current: pd.DataFrame) -> ColumnMapping:
    """Define which columns are target, prediction, numerical, categorical."""
    target = cfg.data.target_column  # "Transported"

    # Identify feature columns
    exclude = {target, "PassengerId", "Name", "prediction", "prediction_proba"}
    numerical_features = [
        c for c in reference.select_dtypes(include=["number"]).columns
        if c not in exclude
    ]
    categorical_features = [
        c for c in reference.select_dtypes(include=["object", "bool", "category"]).columns
        if c not in exclude
    ]

    logger.info("Numerical features  : %s", numerical_features)
    logger.info("Categorical features: %s", categorical_features)

    # Only set target if present in BOTH datasets
    has_target = target in reference.columns and target in current.columns
    has_prediction = "prediction" in reference.columns and "prediction" in current.columns

    return ColumnMapping(
        target=target if has_target else None,
        prediction="prediction" if has_prediction else None,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )


# ─────────────────────────────────────────────
# 3. Data Drift Report
# ─────────────────────────────────────────────
def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> dict:
    """Generate HTML data drift report and return drift summary."""
    logger.info("Running data drift report...")

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    # Save HTML report
    output_path = REPORTS_DIR / "data_drift_report.html"
    report.save_html(str(output_path))
    logger.info("Data drift report saved to %s", output_path)

    # Extract drift summary
    report_dict = report.as_dict()
    drift_metrics = report_dict.get("metrics", [{}])[0].get("result", {})

    summary = {
        "share_of_drifted_columns": drift_metrics.get("share_of_drifted_columns", None),
        "number_of_drifted_columns": drift_metrics.get("number_of_drifted_columns", None),
        "number_of_columns": drift_metrics.get("number_of_columns", None),
        "dataset_drift": drift_metrics.get("dataset_drift", None),
    }

    logger.info("Drift summary: %s", summary)
    return summary


# ─────────────────────────────────────────────
# 4. Model Performance Report (if predictions exist)
# ─────────────────────────────────────────────
def run_model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    column_mapping: ColumnMapping,
) -> None:
    """Generate model performance report if prediction column exists."""
    if "prediction" not in reference.columns:
        logger.info("No 'prediction' column found — skipping model performance report.")
        logger.info("To enable: add model predictions to train_engineered.csv")
        return

    logger.info("Running model performance report...")

    report = Report(metrics=[ClassificationPreset()])
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )

    output_path = REPORTS_DIR / "model_performance_report.html"
    report.save_html(str(output_path))
    logger.info("Model performance report saved to %s", output_path)


# ─────────────────────────────────────────────
# 5. Drift Alert
# ─────────────────────────────────────────────
def check_drift_threshold(summary: dict, threshold: float = None) -> bool:
    """Return True if drift exceeds threshold — triggers alert in CI/CD."""
    threshold = threshold or cfg.monitoring.drift_share_threshold
    share = summary.get("share_of_drifted_columns")

    if share is None:
        logger.warning("Could not determine drift share.")
        return False

    if share > threshold:
        logger.warning(
            "⚠️  DRIFT ALERT: %.1f%% of features drifted (threshold: %.1f%%)",
            share * 100, threshold * 100,
        )
        return True
    else:
        logger.info(
            "✅ No significant drift: %.1f%% of features drifted (threshold: %.1f%%)",
            share * 100, threshold * 100,
        )
        return False


# ─────────────────────────────────────────────
# 6. Save Summary JSON
# ─────────────────────────────────────────────
def save_drift_summary(summary: dict, drift_detected: bool) -> None:
    """Save drift summary as JSON for CI/CD or dashboards."""
    summary["drift_detected"] = drift_detected
    output_path = REPORTS_DIR / "drift_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Drift summary JSON saved to %s", output_path)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run_monitoring() -> None:
    logger.info("=" * 50)
    logger.info(" Phase 6: Evidently Drift Monitoring")
    logger.info("=" * 50)

    # 1. Load data
    reference, current = load_data()

    # 2. Column mapping
    column_mapping = get_column_mapping(reference, current)

    # 3. Data drift report
    drift_summary = run_data_drift_report(reference, current, column_mapping)

    # 4. Model performance report (optional)
    run_model_performance_report(reference, current, column_mapping)

    # 5. Check drift threshold
    drift_detected = check_drift_threshold(drift_summary)

    # 6. Save summary
    save_drift_summary(drift_summary, drift_detected)

    logger.info("=" * 50)
    logger.info(" Monitoring complete!")
    logger.info(" Reports saved to: %s/", REPORTS_DIR)
    logger.info("=" * 50)


if __name__ == "__main__":
    run_monitoring()