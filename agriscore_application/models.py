# agriscore/model.py
import joblib
import pandas as pd
from pathlib import Path

import os


#get the parent directory of the current file
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


model_path = os.path.join(
    parent_dir ,
    "Models"
)


def load_model():
    """Load the trained LightGBM model"""

    if not os.path.exists(model_path) :
        raise FileNotFoundError(f"Model not found at {model_path}. Please train and save the model first.")
    return joblib.load(os.path.join(model_path , "lightgbm_model.pkl"))


def predict_pd(model, X: pd.DataFrame):
    """Return probability of default (bad borrower = 1)"""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    else:
        # Fallback for models without predict_proba
        return model.predict(X)