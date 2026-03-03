"""
app.py — Phase 3: FastAPI Serving
Spaceship Titanic Classification Pipeline

Endpoints:
    GET  /health   → health check + model status
    POST /predict  → single passenger prediction
    POST /predict/batch → batch predictions
"""

import logging
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import cfg
from feature_engg import run_feature_engineering

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────

app = FastAPI(
    title="Spaceship Titanic — Transport Predictor",
    description="Predicts whether a passenger was transported to another dimension.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# Load Model at Startup
# ─────────────────────────────────────────────

model = None
model_loaded_at = None

def load_model():
    """Load the trained XGBoost model from disk."""
    global model, model_loaded_at
    model_path = cfg.model.model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run model.py first to train and save the model."
        )

    model = joblib.load(model_path)
    model_loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Model loaded from {model_path} at {model_loaded_at}")


@app.on_event("startup")
def startup_event():
    load_model()


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class PassengerFeatures(BaseModel):
    """Raw input features — same as Kaggle dataset columns."""
    PassengerId:  str            = Field(..., example="0001_01")
    HomePlanet:   Optional[str]  = Field(None, example="Europa")
    CryoSleep:    Optional[bool] = Field(None, example=False)
    Cabin:        Optional[str]  = Field(None, example="B/0/P")
    Destination:  Optional[str]  = Field(None, example="TRAPPIST-1e")
    Age:          Optional[float]= Field(None, example=39.0)
    VIP:          Optional[bool] = Field(None, example=False)
    RoomService:  Optional[float]= Field(None, example=0.0)
    FoodCourt:    Optional[float]= Field(None, example=0.0)
    ShoppingMall: Optional[float]= Field(None, example=0.0)
    Spa:          Optional[float]= Field(None, example=0.0)
    VRDeck:       Optional[float]= Field(None, example=0.0)
    Name:         Optional[str]  = Field(None, example="Maham Ofracculy")

    class Config:
        json_schema_extra = {
            "example": {
                "PassengerId":  "0001_01",
                "HomePlanet":   "Europa",
                "CryoSleep":    False,
                "Cabin":        "B/0/P",
                "Destination":  "TRAPPIST-1e",
                "Age":          39.0,
                "VIP":          False,
                "RoomService":  0.0,
                "FoodCourt":    0.0,
                "ShoppingMall": 0.0,
                "Spa":          0.0,
                "VRDeck":       0.0,
                "Name":         "Maham Ofracculy"
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response for a single passenger."""
    PassengerId:    str
    Transported:    bool
    probability:    float = Field(..., description="Probability of being transported (class 1)")
    confidence:     str   = Field(..., description="High / Medium / Low")


class BatchPredictionResponse(BaseModel):
    predictions:    List[PredictionResponse]
    total:          int
    transported:    int
    not_transported: int


class HealthResponse(BaseModel):
    status:         str
    model_loaded:   bool
    model_path:     str
    model_loaded_at: Optional[str]
    version:        str


# ─────────────────────────────────────────────
# Helper: Confidence Band
# ─────────────────────────────────────────────

def get_confidence(prob: float) -> str:
    if prob >= 0.75 or prob <= 0.25:
        return "High"
    elif prob >= 0.60 or prob <= 0.40:
        return "Medium"
    else:
        return "Low"


# ─────────────────────────────────────────────
# Helper: Preprocess input for model
# ─────────────────────────────────────────────

def preprocess(passengers: List[PassengerFeatures]) -> pd.DataFrame:
    """Convert list of PassengerFeatures → engineered feature DataFrame."""
    raw_df = pd.DataFrame([p.dict() for p in passengers])
    engineered_df = run_feature_engineering(raw_df, is_train=False)

    # Ensure column order matches training features
    expected_cols = [c for c in engineered_df.columns if c != cfg.data.target_column]
    engineered_df = engineered_df[expected_cols]

    return engineered_df


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get(cfg.api.health_endpoint, response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API and model are running correctly."""
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        model_path=str(cfg.model.model_path),
        model_loaded_at=model_loaded_at,
        version="1.0.0",
    )


@app.post(cfg.api.predict_endpoint, response_model=PredictionResponse, tags=["Prediction"])
def predict(passenger: PassengerFeatures):
    """
    Predict whether a single passenger was transported.

    - Input  : raw passenger features (same as Kaggle dataset)
    - Output : Transported (bool), probability, confidence level
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again shortly.")

    try:
        df = preprocess([passenger])
        prob  = float(model.predict_proba(df)[0][1])
        pred  = prob >= 0.5

        return PredictionResponse(
            PassengerId=passenger.PassengerId,
            Transported=pred,
            probability=round(prob, 4),
            confidence=get_confidence(prob),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(passengers: List[PassengerFeatures]):
    """
    Predict for a batch of passengers.

    - Input  : list of passenger features
    - Output : list of predictions + summary stats
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if len(passengers) == 0:
        raise HTTPException(status_code=400, detail="Empty passenger list.")

    if len(passengers) > 1000:
        raise HTTPException(status_code=400, detail="Batch size must be <= 1000.")

    try:
        df    = preprocess(passengers)
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(bool)

        results = [
            PredictionResponse(
                PassengerId=p.PassengerId,
                Transported=bool(pred),
                probability=round(float(prob), 4),
                confidence=get_confidence(float(prob)),
            )
            for p, pred, prob in zip(passengers, preds, probs)
        ]

        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            transported=int(preds.sum()),
            not_transported=int((~preds).sum()),
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=cfg.api.host,
        port=cfg.api.port,
        log_level=cfg.api.log_level,
        reload=False,
    )