# agriscore/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib
from pathlib import Path
from config import FEATURE_COLUMNS

import os

#get the parent directory of the current file
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


# get the path to the data folder
data_path = os.path.join(
    parent_dir,
    "Data sets"
)

model_path = os.path.join(
    parent_dir ,
    "Models"
)



def create_synthetic_target(df: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(42)
    score = (
        (df['yield_per_ha'] > df['yield_per_ha'].median()).astype(int) +
        (df['balance_stability'] > df['balance_stability'].median()).astype(int) +
        (df['climate_risk_score'] < df['climate_risk_score'].median()).astype(int)
    )
    df['good_borrower'] = np.where(
        (score + np.random.normal(0, 0.5, len(df))) > 2, 1, 0
    )
    return df


def fit_encoder_and_save(df: pd.DataFrame, save_path: str = os.path.join(model_path,"encoder.pkl")):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
    enc.fit(df[categorical_cols])
    
    meta = {
        "categorical_cols": categorical_cols,
        "encoder": enc,
        "feature_names_out": enc.get_feature_names_out(categorical_cols).tolist()
    }
    joblib.dump(meta, save_path)
    
    # Save exact training columns
    X_train = preprocess_for_scoring(df, is_training=True)
    joblib.dump(X_train.columns.tolist(), os.path.join(model_path,"feature_columns.pkl"))
    
    print(f" Encoder saved | Training features: {len(X_train.columns)}")
    return meta


def preprocess_for_scoring(raw_data: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    df = raw_data.copy()
    
    # Encode all categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    try:
        meta = joblib.load('model_assets/encoder.pkl')
        encoder = meta.get("encoder")
        trained_cats = meta.get("categorical_cols", [])
    except Exception:
        encoder = None
        trained_cats = []

    if encoder and trained_cats:
        for col in trained_cats:
            if col not in df.columns:
                df[col] = "missing"
        
        encoded = encoder.transform(df[trained_cats])
        encoded_df = pd.DataFrame(encoded, columns=meta["feature_names_out"], index=df.index)
        df = pd.concat([df.drop(columns=trained_cats, errors='ignore'), encoded_df], axis=1)

    # === CRITICAL: Force ALL columns to numeric ===
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                df[col] = df[col].astype('category').cat.codes

    # Fill any NaNs created during conversion
    df = df.fillna(0)

    # Force exact feature columns from training
    target_cols = FEATURE_COLUMNS if FEATURE_COLUMNS else df.columns.tolist()
    
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0.0

    return df[target_cols]