# test_full.py
import pandas as pd
import joblib
from preprocessing import preprocess_for_scoring
from scoring import score_farmer
from explainer import load_explainer, get_explanation, plot_shap_summary, get_shap_interactions
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

print("=== AgriScore Full Test ===\n")

# ================== SAMPLE FARMER DATA ==================
my_sample = {
    'hh_size': 5,
    'hh_dependency_ratio': 1.2,
    'age_manager': 42,
    'hh_primary_education': 1,
    'hh_formal_education': 8,
    'farm_size': 2.5,
    'farm_work': 180,
    'livestock': 8,
    'inorganic_fertilizer': 120,
    'fertilizer_per_ha': 48,
    'yield_kg_ha': 1850,
    'yield_stability': 0.75,
    'yield_per_ha': 740,
    'rainfall_mm': 850,
    'soil_quality_index': 65,
    'soil_npk': 45,               # numeric now
    'drought_risk': 0.35,
    'climate_stress_index': 0.42,
    'crop_health_index': 0.78,
    'tx_frequency': 28,
    'total_volume': 145000,
    'avg_transaction': 5200,
    'input_purchase_ratio': 0.68,
    'cashflow_volatility': 0.22,
    'balance_stability': 0.81,
    'fraud_rate': 0.08,
    'input_efficiency': 1.45,
    'climate_risk_score': 0.38,
    'hh_shock': 0,
    'drought_shock': 1,
    'dist_market': 12,
    'country': 'Kenya',
    'main_crop_clean': 'maize'
}

# ================== SINGLE FARMER SCORING ==================
print("Scoring single farmer...")
result = score_farmer(my_sample)

print(f"Credit Score     : {result['credit_score']}")
print(f"Risk Tier        : {result['risk_tier']}")
print(f"PD Probability   : {result['pd_probability']:.1%}")
print(f"Recommendation   : {result['recommendation']}")
print(f"Explanation      : {result['explanation_text']}\n")

print("Top Positive Drivers:")
for d in result['top_positive_drivers']:
    print(f"   + {d['feature']:25} : {d['shap_value']:.4f}")

print("\nTop Negative Drivers:")
for d in result['top_negative_drivers']:
    print(f"   - {d['feature']:25} : {d['shap_value']:.4f}")

# ================== SHAP INTERACTIONS ==================
print("\n" + "="*60)
print("Computing SHAP Interaction Effects...")

explainer = load_explainer()
X_sample = preprocess_for_scoring(pd.DataFrame([my_sample]))

interactions = get_shap_interactions(explainer, X_sample, top_n=6)

print("\nTop Feature Interactions:")
for inter in interactions:
    print(f"   {inter['feature1']:20} ↔ {inter['feature2']:20} : {inter['interaction_strength']:.4f}")

# ================== GLOBAL SUMMARY PLOT ==================
print("\nGenerating SHAP Summary Plot...")
df_bg = pd.read_csv(os.path.join(data_path , "Final_dataset.csv")).sample(800, random_state=42)   # adjust path if needed
X_bg = preprocess_for_scoring(df_bg)

plot_path = plot_shap_summary(explainer, X_bg)
print(f"Summary plot saved to: {plot_path}")