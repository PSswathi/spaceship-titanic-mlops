"""
config.py — Centralized configuration for the Full MLOps Pipeline
Covers all 7 phases: Setup, Train & Track, Serve, Dockerize, Deploy, Monitor, Wrap Up
"""

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────
# Phase 1: Setup & Data
# ─────────────────────────────────────────────
@dataclass
class DataConfig:
    # Kaggle dataset
    kaggle_dataset: str = "username/your-kaggle-dataset-name"   # e.g. "titanic" or "playground-series-s4e1"
    kaggle_json_path: str = "~/.kaggle/kaggle.json"

    # Local paths
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    train_file: str = "train.csv"
    test_file: str = "test.csv"
    target_column: str = "Transported"                               # ← change to your label column

    # Data versioning (DVC or simple hash)
    data_version: str = "v1.0"
    enable_dvc: bool = False


# ─────────────────────────────────────────────
# Phase 2: Train & Track — MLflow
# ─────────────────────────────────────────────
@dataclass
class MLflowConfig:
    tracking_uri: str = "mlruns"                # local; swap for remote URI in prod
    experiment_name: str = "kaggle-classification"
    run_name: str = "baseline-run"
    registered_model_name: str = "kaggle-clf-model"
    model_stage: str = "Production"                             # Staging | Production | Archived
    artifact_location: str = "mlruns"


# ─────────────────────────────────────────────
# Phase 2: Model Hyperparameters
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    model_type: str = "xgboost"                                 # xgboost | lightgbm | sklearn-rf
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # XGBoost / LightGBM shared params
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1                                      # L1
    reg_lambda: float = 1.0                                     # L2
    scale_pos_weight: float = 1.0                               # for imbalanced classes

    # Evaluation metric
    eval_metric: str = "auc"                                    # auc | logloss | f1 | accuracy

    # Saved model artifact (used by FastAPI)
    model_output_dir: Path = Path("models")
    model_filename: str = "model.pkl"

    @property
    def model_path(self) -> Path:
        return self.model_output_dir / self.model_filename


# ─────────────────────────────────────────────
# Phase 3: Serve — FastAPI
# ─────────────────────────────────────────────
@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    predict_endpoint: str = "/predict"
    health_endpoint: str = "/health"
    log_level: str = "info"
    workers: int = 1


# ─────────────────────────────────────────────
# Phase 4: Dockerize
# ─────────────────────────────────────────────
@dataclass
class DockerConfig:
    image_name: str = "kaggle-clf-app"
    image_tag: str = "latest"
    container_port: int = 8000
    host_port: int = 8000
    dockerfile_path: str = "Dockerfile"

    @property
    def full_image_name(self) -> str:
        return f"{self.image_name}:{self.image_tag}"

    @property
    def run_command(self) -> str:
        return (
            f"docker run -d -p {self.host_port}:{self.container_port} "
            f"--name {self.image_name} {self.full_image_name}"
        )


# ─────────────────────────────────────────────
# Phase 5: Deploy — AWS EC2
# ─────────────────────────────────────────────
@dataclass
class AWSConfig:
    region: str = "us-east-1"
    ec2_instance_type: str = "t2.micro"
    ec2_ami_id: str = "ami-xxxxxxxxxxxxxxxxx"                   # ← fill in your AMI
    key_pair_name: str = "your-key-pair"
    security_group_id: str = "sg-xxxxxxxxxxxxxxxxx"
    ec2_user: str = "ubuntu"
    ec2_public_ip: str = ""                                     # set after launch

    # ECR (optional — if pushing image to ECR instead of Docker Hub)
    ecr_repo_uri: str = ""                                      # e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/kaggle-clf

    # Git repo to pull on EC2
    github_repo_url: str = "https://github.com/your-username/your-repo.git"
    repo_branch: str = "main"

    @property
    def ssh_command(self) -> str:
        return f"ssh -i {self.key_pair_name}.pem {self.ec2_user}@{self.ec2_public_ip}"


# ─────────────────────────────────────────────
# Phase 6: Monitor — Evidently
# ─────────────────────────────────────────────
@dataclass
class MonitoringConfig:
    reference_data_path: Path = Path("data/processed/train.csv")
    current_data_path: Path = Path("data/processed/test.csv")
    drift_report_output_dir: Path = Path("reports")
    drift_report_filename: str = "drift_report.html"
    enable_target_drift: bool = True
    enable_data_drift: bool = True
    drift_share_threshold: float = 0.2                          # flag if >20% features drift

    @property
    def drift_report_path(self) -> Path:
        return self.drift_report_output_dir / self.drift_report_filename


# ─────────────────────────────────────────────
# Phase 7: Wrap Up
# ─────────────────────────────────────────────
@dataclass
class ProjectConfig:
    project_name: str = "kaggle-mlops-pipeline"
    readme_path: str = "README.md"
    screenshots_dir: Path = Path("screenshots")
    github_repo: str = "https://github.com/your-username/your-repo"


# ─────────────────────────────────────────────
# Master Config — single import point
# ─────────────────────────────────────────────
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    docker: DockerConfig = field(default_factory=DockerConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    project: ProjectConfig = field(default_factory=ProjectConfig)


# ─────────────────────────────────────────────
# Singleton — import this everywhere
# ─────────────────────────────────────────────
cfg = Config()


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Project       : {cfg.project.project_name}")
    print(f"MLflow URI    : {cfg.mlflow.tracking_uri}")
    print(f"Model path    : {cfg.model.model_path}")
    print(f"API           : http://{cfg.api.host}:{cfg.api.port}{cfg.api.predict_endpoint}")
    print(f"Docker image  : {cfg.docker.full_image_name}")
    print(f"AWS region    : {cfg.aws.region}")
    print(f"Drift report  : {cfg.monitoring.drift_report_path}")