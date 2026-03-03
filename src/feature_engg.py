"""
feature_engineering.py — Phase 2: Feature Engineering
Spaceship Titanic Classification Pipeline

Features:
    - PassengerId  : extract Group ID + Group Size
    - Cabin        : extract Deck, CabinNum, Side
    - Name         : extract FirstName, LastName (then drop)
    - CryoSleep    : bool → int; if CryoSleep=True, all amenity spend must be 0
    - Amenities    : RoomService, FoodCourt, ShoppingMall, Spa, VRDeck
                     → TotalSpend, SpendPerAmenity, HasSpent flag
    - Age          : AgeBin (child / teen / adult / senior)
    - Categorical  : Label encoding for tree models
    - Missing      : median imputation (numerical), mode imputation (categorical)
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import cfg
from data_loader import load_all

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ── Amenity columns ────────────────────────────────────────────────────────────
AMENITY_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

# ── Categorical columns to encode ─────────────────────────────────────────────
CAT_COLS = ["HomePlanet", "Destination", "Deck", "Side", "AgeBin"]


# ─────────────────────────────────────────────
# 1. PassengerId → Group features
# ─────────────────────────────────────────────

def extract_passenger_id_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split PassengerId (gggg_pp) into:
        - GroupId    : group the passenger belongs to
        - GroupSize  : how many people share the same group
        - IsAlone    : 1 if travelling solo
    """
    df["GroupId"]   = df["PassengerId"].str.split("_").str[0].astype(int)
    df["GroupSize"] = df.groupby("GroupId")["GroupId"].transform("count")
    df["IsAlone"]   = (df["GroupSize"] == 1).astype(int)
    logger.info("Extracted PassengerId features: GroupId, GroupSize, IsAlone")
    return df


# ─────────────────────────────────────────────
# 2. Cabin → Deck, CabinNum, Side
# ─────────────────────────────────────────────

def extract_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split Cabin (deck/num/side) into:
        - Deck      : A–G, T
        - CabinNum  : cabin number (int)
        - Side      : P (Port) or S (Starboard)
    """
    cabin_split = df["Cabin"].str.split("/", expand=True)
    df["Deck"]     = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"]     = cabin_split[2]
    logger.info("Extracted Cabin features: Deck, CabinNum, Side")
    return df


# ─────────────────────────────────────────────
# 3. Name → drop (not predictive)
# ─────────────────────────────────────────────

def drop_name(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Name column — not predictive for this problem."""
    df = df.drop(columns=["Name"], errors="ignore")
    logger.info("Dropped: Name")
    return df


# ─────────────────────────────────────────────
# 4. CryoSleep — fix amenity leakage
# ─────────────────────────────────────────────

def fix_cryo_sleep(df: pd.DataFrame) -> pd.DataFrame:
    """
    Passengers in CryoSleep cannot spend on amenities.
    Fill missing amenity values with 0 where CryoSleep=True.
    Encodes CryoSleep as explicit 0/1 int.
    """
    cryo_mask = df["CryoSleep"] == True  # noqa: E712
    for col in AMENITY_COLS:
        df.loc[cryo_mask, col] = df.loc[cryo_mask, col].fillna(0)
    # True/False/'True'/'False'/NaN → 0/1 int
    df["CryoSleep"] = df["CryoSleep"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    logger.info("Fixed CryoSleep amenity leakage | CryoSleep encoded as 0/1 int")
    return df


# ─────────────────────────────────────────────
# 4b. VIP → 0/1 int
# ─────────────────────────────────────────────

def encode_vip(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode VIP boolean column (True/False/'True'/'False') → 0/1 int.
    Missing values default to 0 (not VIP).
    """
    df["VIP"] = df["VIP"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    logger.info("Encoded VIP as 0/1 int")
    return df


# ─────────────────────────────────────────────
# 5. Amenity spend features
# ─────────────────────────────────────────────

def create_spend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive spending features:
        - TotalSpend        : sum of all 5 amenity columns
        - SpendPerAmenity   : TotalSpend / 5
        - HasSpent          : 1 if TotalSpend > 0
        - LogTotalSpend     : log1p(TotalSpend) — reduces skew
    """
    df["TotalSpend"]      = df[AMENITY_COLS].sum(axis=1)
    df["SpendPerAmenity"] = df["TotalSpend"] / len(AMENITY_COLS)
    df["HasSpent"]        = (df["TotalSpend"] > 0).astype(int)
    df["LogTotalSpend"]   = np.log1p(df["TotalSpend"])

    # Log-transform individual amenity cols to reduce skew
    for col in AMENITY_COLS:
        df[f"Log{col}"] = np.log1p(df[col].fillna(0))

    logger.info("Created spend features: TotalSpend, SpendPerAmenity, HasSpent, LogTotalSpend, Log<Amenity>")
    return df


# ─────────────────────────────────────────────
# 6. Age binning
# ─────────────────────────────────────────────

def bin_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin Age into:
        child  : 0–12
        teen   : 13–17
        adult  : 18–60
        senior : 61+
    """
    bins   = [-1, 12, 17, 60, np.inf]
    labels = ["child", "teen", "adult", "senior"]
    df["AgeBin"] = pd.cut(df["Age"], bins=bins, labels=labels).astype(str)
    logger.info("Created AgeBin feature")
    return df


# ─────────────────────────────────────────────
# 7. Missing value imputation
# ─────────────────────────────────────────────

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
        - Numerical  : median
        - Categorical: mode (most frequent)
    """
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        if df[col].isnull().any():
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.info(f"  Imputed {col} with median={median:.2f}")

    for col in cat_cols:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col] = df[col].fillna(mode).infer_objects(copy=False)
            logger.info(f"  Imputed {col} with mode='{mode}'")

    logger.info("Missing value imputation complete")
    return df


# ─────────────────────────────────────────────
# 8. Label Encoding
# ─────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode categorical columns.
    Suitable for tree-based models (XGBoost, LightGBM).
    """
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"  Label encoded: {col}")
    return df


# ─────────────────────────────────────────────
# 9. Drop raw / ID columns
# ─────────────────────────────────────────────

def drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original columns that have been replaced by engineered features."""
    cols_to_drop = ["PassengerId", "Cabin"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    logger.info(f"Dropped raw columns: {cols_to_drop}")
    return df


# ─────────────────────────────────────────────
# Master Pipeline
# ─────────────────────────────────────────────

def run_feature_engineering(
    df: pd.DataFrame,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline to a DataFrame.

    Args:
        df       : raw DataFrame (train or test)
        is_train : if True, separates and re-attaches the target column

    Returns:
        Engineered DataFrame ready for model training / inference
    """
    logger.info(f"\n{'─'*40}")
    logger.info(f"Running feature engineering | is_train={is_train} | shape={df.shape}")

    # Separate target before engineering (train only)
    target = None
    if is_train and cfg.data.target_column in df.columns:
        target = df[cfg.data.target_column].astype(int)
        df = df.drop(columns=[cfg.data.target_column])

    df = extract_passenger_id_features(df)
    df = extract_cabin_features(df)
    df = drop_name(df)
    df = fix_cryo_sleep(df)
    df = encode_vip(df)
    df = create_spend_features(df)
    df = bin_age(df)
    df = impute_missing(df)
    df = encode_categoricals(df)
    df = drop_raw_columns(df)

    # Re-attach target
    if is_train and target is not None:
        df[cfg.data.target_column] = target.values

    logger.info(f"Feature engineering complete | output shape={df.shape}")
    logger.info(f"{'─'*40}\n")
    return df


# ─────────────────────────────────────────────
# Save Engineered Data
# ─────────────────────────────────────────────

def save_engineered(df: pd.DataFrame, filename: str) -> Path:
    """Save engineered DataFrame to cfg.data.processed_data_dir."""
    out_path = cfg.data.processed_data_dir / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_df, test_df, sample_df = load_all()

    train_eng = run_feature_engineering(train_df, is_train=True)
    test_eng  = run_feature_engineering(test_df,  is_train=False)

    save_engineered(train_eng, "train_engineered.csv")
    save_engineered(test_eng,  "test_engineered.csv")

    print(f"\nTrain engineered : {train_eng.shape}")
    print(f"Test  engineered : {test_eng.shape}")
    print(f"\nColumns:\n{train_eng.columns.tolist()}")
    print(f"\nSample:\n{train_eng.head(3)}")