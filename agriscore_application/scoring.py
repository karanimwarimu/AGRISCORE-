# agriscore/scoring.py
import pandas as pd
from pathlib import Path
from preprocessing import preprocess_for_scoring
from models import load_model, predict_pd
from explainer import load_explainer, get_explanation
from config import RISK_THRESHOLDS, SCORE_SCALE


def score_farmer(raw_input):
    """
    Main scoring function for AgriScore.
    Supports single farmer or batch input.
    """
    # Convert input to DataFrame
    if isinstance(raw_input, dict):
        df = pd.DataFrame([raw_input])
        single = True
    elif isinstance(raw_input, list):
        df = pd.DataFrame(raw_input)
        single = len(raw_input) == 1
    elif isinstance(raw_input, pd.DataFrame):
        df = raw_input.copy()
        single = len(df) == 1
    else:
        raise ValueError("Input must be dict, list of dicts, or pandas DataFrame")

    if len(df) == 0:
        raise ValueError("Empty input data")

    # Preprocess
    X = preprocess_for_scoring(df)

    # Load assets
    model = load_model()
    explainer = load_explainer()

    # Predict
    pd_probs = predict_pd(model, X)
    explanations = get_explanation(explainer, X)

    results = []
    for i in range(len(df)):
        pd_prob = float(pd_probs[i])
        
        credit_score = int(SCORE_SCALE[0] + (1 - pd_prob) * (SCORE_SCALE[1] - SCORE_SCALE[0]))
        
        # Risk tier
        if pd_prob < RISK_THRESHOLDS.get('low', 0.15):
            risk_tier = "Low"
            recommendation = "Approve"
        elif pd_prob < RISK_THRESHOLDS.get('medium', 0.35):
            risk_tier = "Medium"
            recommendation = "Approve with Caution"
        else:
            risk_tier = "High"
            recommendation = "Review / Decline"

       
        if isinstance(explanations, list):
            exp = explanations[i]
        else:
            exp = explanations 

        # New Features
        loan_amount = calculate_recommended_loan(credit_score, risk_tier)
        confidence_score = calculate_confidence(pd_prob)

        result = {
            "credit_score": credit_score,
            "risk_tier": risk_tier,
            "pd_probability": round(pd_prob, 4),
            "confidence_score": round(confidence_score, 2),
            "recommended_loan_amount": loan_amount,
            "recommendation": recommendation,
            "top_positive_drivers": exp['top_positive'],
            "top_negative_drivers": exp['top_negative'],
            "explanation_text": generate_explanation_text(exp)
        }
        results.append(result)

    return results[0] if single else results

def calculate_recommended_loan(credit_score: int, risk_tier: str) -> int:
    """Recommend loan amount in KES based on score and risk"""
    base_amounts = {
        "Low": 150000,
        "Medium": 80000,
        "High": 25000
    }
    # Scale by credit score
    score_factor = (credit_score - 300) / (850 - 300)
    return int(base_amounts.get(risk_tier, 30000) * score_factor)


def calculate_confidence(pd_prob: float) -> float:
    """Confidence in the prediction (0-100)"""
    # Higher confidence when prediction is extreme (very safe or very risky)
    return round(100 * (1 - 2 * abs(pd_prob - 0.5)), 1)


def generate_explanation_text(exp):
    """Improved plain language explanation"""
    pos = exp['top_positive'][0]
    neg = exp['top_negative'][0]
    
    return (
        f"Strongest positive factor: {pos['feature'].replace('_', ' ')} "
        f"(+{pos['shap_value']:.3f}). Main risk factor: "
        f"{neg['feature'].replace('_', ' ')} ({neg['shap_value']:.3f})."
    )