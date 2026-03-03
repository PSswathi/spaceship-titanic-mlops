"""
model.py — Phase 2: Model Training & Evaluation
Spaceship Titanic Classification Pipeline

- XGBoost classifier with hyperparameters from config.py
- Stratified K-Fold cross-validation
- Full evaluation: Accuracy, ROC-AUC, F1, Confusion Matrix, Classification Report
- MLflow experiment tracking + model registry
- Model saved locally as .pkl artifact
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from config import cfg
from feature_engg import run_feature_engineering, save_engineered
from data_loader import load_all

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# 1. Load & Prepare Data
# ─────────────────────────────────────────────

def load_engineered_data() -> tuple:
    """
    Load engineered train CSV if it exists, otherwise run feature engineering.

    Returns:
        X (pd.DataFrame), y (pd.Series)
    """
    train_path = cfg.data.processed_data_dir / "train_engineered.csv"

    if train_path.exists():
        logger.info(f"Loading engineered train data from {train_path}")
        df = pd.read_csv(train_path)
    else:
        logger.info("Engineered data not found — running feature engineering...")
        train_df, _, _ = load_all()
        df = run_feature_engineering(train_df, is_train=True)
        save_engineered(df, "train_engineered.csv")

    if cfg.data.target_column not in df.columns:
        raise ValueError(f"Target column '{cfg.data.target_column}' not found in engineered data.")

    X = df.drop(columns=[cfg.data.target_column])
    y = df[cfg.data.target_column].astype(int)

    logger.info(f"Features: {X.shape} | Target distribution:\n{y.value_counts().to_string()}")
    return X, y


# ─────────────────────────────────────────────
# 2. Build XGBoost Model
# ─────────────────────────────────────────────

def build_model() -> XGBClassifier:
    """Instantiate XGBClassifier using hyperparameters from config.py."""
    model = XGBClassifier(
        n_estimators=cfg.model.n_estimators,
        learning_rate=cfg.model.learning_rate,
        max_depth=cfg.model.max_depth,
        subsample=cfg.model.subsample,
        colsample_bytree=cfg.model.colsample_bytree,
        reg_alpha=cfg.model.reg_alpha,
        reg_lambda=cfg.model.reg_lambda,
        scale_pos_weight=cfg.model.scale_pos_weight,
        eval_metric=cfg.model.eval_metric,
        random_state=cfg.model.random_state,
        use_label_encoder=False,
        verbosity=0,
    )
    logger.info(f"Built XGBClassifier with {cfg.model.n_estimators} estimators")
    return model


# ─────────────────────────────────────────────
# 3. Cross-Validation
# ─────────────────────────────────────────────

def cross_validate(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run Stratified K-Fold CV and return mean/std of metrics.

    Returns:
        dict with mean and std of accuracy, roc_auc, f1
    """
    skf = StratifiedKFold(n_splits=cfg.model.cv_folds, shuffle=True, random_state=cfg.model.random_state)

    acc_scores, auc_scores, f1_scores = [], [], []

    logger.info(f"\nRunning {cfg.model.cv_folds}-Fold Stratified Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model()
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        preds      = model.predict(X_val)
        probs      = model.predict_proba(X_val)[:, 1]
        acc        = accuracy_score(y_val, preds)
        auc        = roc_auc_score(y_val, probs)
        f1         = f1_score(y_val, preds)

        acc_scores.append(acc)
        auc_scores.append(auc)
        f1_scores.append(f1)

        logger.info(f"  Fold {fold} — Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")

    cv_results = {
        "cv_accuracy_mean":  float(np.mean(acc_scores)),
        "cv_accuracy_std":   float(np.std(acc_scores)),
        "cv_roc_auc_mean":   float(np.mean(auc_scores)),
        "cv_roc_auc_std":    float(np.std(auc_scores)),
        "cv_f1_mean":        float(np.mean(f1_scores)),
        "cv_f1_std":         float(np.std(f1_scores)),
    }

    logger.info(f"\nCV Results:")
    logger.info(f"  Accuracy : {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
    logger.info(f"  ROC-AUC  : {cv_results['cv_roc_auc_mean']:.4f} ± {cv_results['cv_roc_auc_std']:.4f}")
    logger.info(f"  F1 Score : {cv_results['cv_f1_mean']:.4f} ± {cv_results['cv_f1_std']:.4f}")

    return cv_results


# ─────────────────────────────────────────────
# 4. Evaluation on Hold-out Val Set
# ─────────────────────────────────────────────

def evaluate_model(model: XGBClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """
    Evaluate trained model on validation set.

    Returns:
        dict of all evaluation metrics
    """
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]

    acc    = accuracy_score(y_val, preds)
    auc    = roc_auc_score(y_val, probs)
    f1     = f1_score(y_val, preds)
    cm     = confusion_matrix(y_val, preds)
    report = classification_report(y_val, preds, target_names=["Not Transported", "Transported"])

    logger.info(f"\n{'─'*40}")
    logger.info("Validation Set Evaluation")
    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  ROC-AUC   : {auc:.4f}")
    logger.info(f"  F1 Score  : {f1:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"{'─'*40}\n")

    return {
        "val_accuracy":  float(acc),
        "val_roc_auc":   float(auc),
        "val_f1":        float(f1),
        "confusion_matrix": cm,
        "classification_report": report,
    }


# ─────────────────────────────────────────────
# 5. Confusion Matrix Plot
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, output_dir: Path) -> Path:
    """Save confusion matrix as a PNG artifact for MLflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Transported", "Transported"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Validation Set")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    logger.info(f"Confusion matrix saved → {path}")
    return path


# ─────────────────────────────────────────────
# 6. Feature Importance Plot
# ─────────────────────────────────────────────

def plot_feature_importance(model: XGBClassifier, feature_names: list, output_dir: Path) -> Path:
    """Save top-20 feature importance plot as PNG artifact for MLflow."""
    output_dir.mkdir(parents=True, exist_ok=True)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.head(20).plot(kind="barh", ax=ax, color="steelblue")
    ax.invert_yaxis()
    ax.set_title("Top 20 Feature Importances (XGBoost)")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = output_dir / "feature_importance.png"
    plt.savefig(path)
    plt.close()
    logger.info(f"Feature importance plot saved → {path}")
    return path


# ─────────────────────────────────────────────
# 7. Save Model Locally
# ─────────────────────────────────────────────

def save_model(model: XGBClassifier) -> Path:
    """Save trained model as .pkl to cfg.model.model_path."""
    cfg.model.model_output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model.model_path)
    logger.info(f"Model saved → {cfg.model.model_path}")
    return cfg.model.model_path


# ─────────────────────────────────────────────
# 8. MLflow Training Run
# ─────────────────────────────────────────────

def train_with_mlflow() -> XGBClassifier:
    """
    Full training pipeline logged to MLflow.

    Steps:
        1. Load engineered data
        2. Train/val split
        3. Cross-validation
        4. Final model training
        5. Evaluation on val set
        6. Log params, metrics, plots, model to MLflow
        7. Register model in MLflow Model Registry
        8. Save model locally

    Returns:
        Trained XGBClassifier
    """
    # ── Setup MLflow ──────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    plots_dir = Path("reports/plots")

    # ── Load data ─────────────────────────────
    X, y = load_engineered_data()

    # ── Train/val split ───────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.model.test_size,
        random_state=cfg.model.random_state,
        stratify=y,
    )
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")

    with mlflow.start_run(run_name=cfg.mlflow.run_name) as run:
        logger.info(f"\nMLflow Run ID : {run.info.run_id}")
        logger.info(f"Experiment    : {cfg.mlflow.experiment_name}")

        # ── Log hyperparameters ───────────────
        params = {
            "model_type":        cfg.model.model_type,
            "n_estimators":      cfg.model.n_estimators,
            "learning_rate":     cfg.model.learning_rate,
            "max_depth":         cfg.model.max_depth,
            "subsample":         cfg.model.subsample,
            "colsample_bytree":  cfg.model.colsample_bytree,
            "reg_alpha":         cfg.model.reg_alpha,
            "reg_lambda":        cfg.model.reg_lambda,
            "scale_pos_weight":  cfg.model.scale_pos_weight,
            "eval_metric":       cfg.model.eval_metric,
            "random_state":      cfg.model.random_state,
            "test_size":         cfg.model.test_size,
            "cv_folds":          cfg.model.cv_folds,
        }
        mlflow.log_params(params)

        # ── Cross-validation ──────────────────
        cv_results = cross_validate(X_train, y_train)
        mlflow.log_metrics(cv_results)

        # ── Final model training ──────────────
        logger.info("\nTraining final model on full training set...")
        model = build_model()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # ── Evaluate ──────────────────────────
        eval_results = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics({
            "val_accuracy": eval_results["val_accuracy"],
            "val_roc_auc":  eval_results["val_roc_auc"],
            "val_f1":       eval_results["val_f1"],
        })

        # ── Log plots ─────────────────────────
        cm_path = plot_confusion_matrix(eval_results["confusion_matrix"], plots_dir)
        fi_path = plot_feature_importance(model, X_train.columns.tolist(), plots_dir)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(fi_path))

        # ── Log classification report as text ─
        report_path = plots_dir / "classification_report.txt"
        report_path.write_text(eval_results["classification_report"])
        mlflow.log_artifact(str(report_path))

       # ── Log & register model ──────────────
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg.mlflow.registered_model_name,
        )
        logger.info(f"Model registered as '{cfg.mlflow.registered_model_name}'")
        
        # ── Save locally ──────────────────────
        save_model(model)

        logger.info(f"\n✓ Training complete.")
        logger.info(f"  Val Accuracy : {eval_results['val_accuracy']:.4f}")
        logger.info(f"  Val ROC-AUC  : {eval_results['val_roc_auc']:.4f}")
        logger.info(f"  Val F1       : {eval_results['val_f1']:.4f}")
        logger.info(f"  MLflow UI    : {cfg.mlflow.tracking_uri}")

    return model


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model = train_with_mlflow()