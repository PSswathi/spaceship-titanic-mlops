# Spaceship Titanic - Full MLOps Pipeline

A production-grade MLOps pipeline built on the Kaggle Spaceship Titanic dataset.
Predicts whether a passenger was transported to an alternate dimension.


## Project Structure
```
spaceship-titanic-mlops/
├── src/
│   ├── config.py               # Centralized config for all phases
│   ├── data_loader.py          # Load raw CSVs from Kaggle
│   ├── feature_engg.py         # Feature engineering pipeline
│   ├── model.py                # XGBoost training + MLflow logging
│   ├── app.py                  # FastAPI serving app
│   └── monitor.py              # Evidently drift monitoring
├── data/
│   ├── raw/                    # Raw Kaggle CSVs (not committed)
│   └── processed/              # Engineered features
├── models/
│   └── model.pkl               # Trained model (not committed)
├── reports/
│   ├── data_drift_report.html  # Evidently drift report
│   └── drift_summary.json      # Drift summary for CI/CD
├── terraform/
│   ├── main.tf                 # EC2 + Security Group + EIP
│   ├── variables.tf            # Input variables
│   ├── outputs.tf              # Output values
│   ├── terraform.tfvars        # Your values (not committed)
│   └── user_data.sh            # EC2 bootstrap script
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── .gitignore
```

## Pipeline Overview
```
Phase 1: Setup & Data    ->  GitHub Repo + Kaggle Data
Phase 2: Train & Track   ->  XGBoost + MLflow
Phase 3: Serve           ->  FastAPI /predict /health
Phase 4: Dockerize       ->  Docker Build & Run
Phase 5: Deploy          ->  AWS EC2 (Terraform IaC)
Phase 6: Monitor         ->  Evidently Drift Report
```

## Model Results

| Metric | Score |
|---|---|
| Validation Accuracy | 81.31% |
| ROC-AUC | 0.9094 |
| F1 Score | 0.8135 |
| CV Accuracy | 0.8056 +/- 0.0069 |

---

## Prerequisites

| Tool | Install |
|---|---|
| Python 3.10+ | python.org |
| Docker Desktop | docker.com |
| Terraform | brew install terraform |
| AWS CLI | brew install awscli |

---

## End-to-End Execution Guide

### Phase 1 - Setup & Data

**Step 1: Clone the repository**
```bash
git clone https://github.com/PSswathi/spaceship-titanic-mlops.git
cd spaceship-titanic-mlops
```

**Step 2: Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see (venv) in your terminal prompt after activation.

**Step 3: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Set up Kaggle credentials**
```bash
mkdir -p ~/.kaggle
cp /path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 5: Download dataset**
```bash
python src/data_loader.py
```

---

### Phase 2 - Train & Track with MLflow

**Step 6: Run data loader**
```bash
python src/data_loader.py
```

**Step 7: Run feature engineering**
```bash
python src/feature_engg.py
```

**Step 8: Run model training**
```bash
python src/model.py
```

This will train XGBoost, log metrics/params/artifacts to MLflow, and save models/model.pkl.

**Step 9: View MLflow UI**

Open a new terminal tab:
```bash
source venv/bin/activate
mlflow ui --backend-store-uri mlruns --port 5001
```

Open in browser: http://localhost:5001

> Screenshot: MLflow_metrics.png, mlflow_training_runs.png and mlflow_training.png

---

### Phase 3 - Serve with FastAPI

**Step 10: Start the FastAPI server**
```bash
source venv/bin/activate
python src/app.py
```

**Step 11: Test the health endpoint**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status":"ok","model_loaded":true,"model_path":"models/model.pkl","version":"1.0.0"}
```

**Step 12: Test a prediction**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PassengerId": "0001_01",
    "HomePlanet": "Europa",
    "CryoSleep": false,
    "Cabin": "B/0/P",
    "Destination": "TRAPPIST-1e",
    "Age": 39.0,
    "VIP": false,
    "RoomService": 0.0,
    "FoodCourt": 0.0,
    "ShoppingMall": 0.0,
    "Spa": 0.0,
    "VRDeck": 0.0,
    "Name": "Maham Ofracculy"
  }'
```

Expected response:
```json
{
  "PassengerId": "0001_01",
  "Transported": true,
  "probability": 0.823,
  "confidence": "High"
}
```

Or open Swagger UI in browser: http://localhost:8000/docs

> Screenshot: FastAPI Swagger UI and prediction response

The screenshots in the folder fastapi_predictions.png

---

### Phase 4 - Dockerize

**Step 13: Open Docker Desktop**

Launch Docker Desktop from Applications and wait for the whale icon to appear in the menu bar.

**Step 14: Build the Docker image**
```bash
docker build -t spaceship-titanic-app:latest .
```

**Step 15: Run the Docker container**
```bash
docker run -d -p 8000:8000 --name spaceship-titanic-app spaceship-titanic-app:latest
```

**Step 16: Verify container is running**
```bash
curl http://localhost:8000/health
```

**Step 17: View container logs**
```bash
docker logs spaceship-titanic-app
```

**Step 18: Open Swagger UI from Docker container**

Open in browser: http://localhost:8000/docs

> Screenshot: Docker Desktop showing running container and health check response

The screenshots are placed in screenshots/docker folder


---

### Phase 5 - Deploy to AWS EC2 with Terraform

**Step 19: Configure AWS credentials**
```bash
aws configure
```

Enter when prompted:
- AWS Access Key ID: from AWS Console -> account name -> Security credentials -> Access keys
- AWS Secret Access Key: same page (only shown once at creation)
- Default region: us-east-1
- Default output format: json

Verify credentials work:
```bash
aws sts get-caller-identity
```

**Step 20: Move key pair to ~/.ssh/**
```bash
mv ~/Downloads/your-key.pem ~/.ssh/
chmod 400 ~/.ssh/your-key.pem
```

**Step 21: Fill in terraform.tfvars**
```bash
cd terraform
```

Edit terraform.tfvars:
```hcl
aws_region      = "us-east-1"
app_name        = "spaceship-titanic-app"
instance_type   = "t2.micro"
key_pair_name   = "my-key"
github_repo_url = "https://github.com/PSswathi/spaceship-titanic-mlops.git"
repo_branch     = "main"
```

**Step 22: Deploy infrastructure**
```bash
terraform init
terraform plan
terraform apply
```

Type yes when prompted. After completion you will see:
```
ec2_public_ip    = "xx.xx.xx.xx"
fastapi_docs_url = "http://xx.xx.xx.xx:8000/docs"
ssh_command      = "ssh -i my-key.pem ubuntu@xx.xx.xx.xx"
```

**Step 23: Copy model and data to EC2**

Run from your LOCAL terminal (not EC2):
```bash
scp -i ~/.ssh/my-key.pem models/model.pkl ubuntu@<EC2_IP>:/home/ubuntu/spaceship-titanic-mlops/models/
scp -i ~/.ssh/my-key.pem -r data/processed ubuntu@<EC2_IP>:/home/ubuntu/spaceship-titanic-mlops/data/
```

**Step 24: SSH into EC2**
```bash
ssh -i ~/.ssh/my-key.pem ubuntu@<EC2_IP>
```

**Step 25: Build and run Docker on EC2**
```bash
cd /home/ubuntu/spaceship-titanic-mlops
git pull origin main
docker build -t spaceship-titanic-app:latest .
docker run -d -p 8000:8000 --name spaceship-titanic-app --restart unless-stopped spaceship-titanic-app:latest
```

**Step 26: Verify EC2 deployment**
```bash
curl http://localhost:8000/health
```

Or open in browser: http://<EC2_IP>:8000/docs

> Screenshot: Swagger UI running on EC2 public IP

The screenshots are placed under screenshots/fastapi_ec2 folder

---

### Phase 6 - Monitor with Evidently

**Step 27: Run drift monitoring (on local machine)**
```bash
cd /Users/swathi/Downloads/spaceship-titanic/spaceship-titanic-mlops
source venv/bin/activate
python src/monitor.py
```

Expected output:
```
INFO | Loading reference data from data/processed/train_engineered.csv
INFO | Loading current data from data/processed/test_engineered.csv
INFO | Reference shape: (8693, 27) | Current shape: (4277, 26)
INFO | Running data drift report...
INFO | Data drift report saved to reports/data_drift_report.html
INFO | No significant drift: 0.0% of features drifted (threshold: 20.0%)
INFO | Drift summary JSON saved to reports/drift_summary.json
INFO | Monitoring complete!
```

**Step 28: View drift report in browser**
```bash
open reports/data_drift_report.html
```

**Drift Results:**

| Metric | Value |
|---|---|
| Features monitored | 26 |
| Drifted features | 0 |
| Share of drift | 0.0% |
| Threshold | 20.0% |
| Dataset drift | None detected |

**Step 29: View drift summary JSON**
```bash
cat reports/drift_summary.json
```

Expected output:
```json
{
  "share_of_drifted_columns": 0.0,
  "number_of_drifted_columns": 0,
  "number_of_columns": 26,
  "dataset_drift": false,
  "drift_detected": false
}
```

> Screenshot: Evidently drift report showing feature distributions in browser

Monitoring results are placed under the reports folder.


## Screenshots

| Step | Screenshot |
|---|---|
| MLflow training runs | screenshots/mlflow_*.png |
| FastAPI Swagger UI (local) | screenshots/fastapi_predictions/fastapi_*.png |
| Docker Desktop container | screenshots/docker/docker_*.png |
| Terraform apply output | screenshots/terraform_apply.png |
| FastAPI Swagger UI (EC2) | screenshots/fastapi_ec2/fastapi_*.png |
| Evidently drift report | screenshots/reports/data_drift_report.html |

---

## EC2- FASTAPI-INFERENCE

ec2_public_ip = "98.83.175.104"

fastapi_docs_url = "http://98.83.175.104:8000/docs"

## Author

Swathi PS
GitHub: https://github.com/PSswathi/spaceship-titanic-mlops