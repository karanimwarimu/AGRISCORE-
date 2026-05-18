import joblib
import shap
import pandas as pd
from pathlib import Path
import os 
import matplotlib.pyplot as plt 

#get the parent directory of the current file
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


model_path = os.path.join(
    parent_dir ,
    "Models"
)

visuals_path =os.path.join(
    parent_dir ,
    "Visuals"
)


def load_explainer():
    explainer_path = os.path.join(model_path ,"agriscore_SHAP_explainer.pkl")
    if not os.path.exists(explainer_path):
        raise FileNotFoundError(f"SHAP explainer not found at {explainer_path}")
    
    explainer = joblib.load(explainer_path)
    print(" SHAP Explainer loaded")
    return explainer


def get_explanation(explainer, X_instance: pd.DataFrame):
    shap_values = explainer.shap_values(X_instance)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    vals = shap_values[0] if len(X_instance) == 1 and len(shap_values.shape) > 1 else shap_values[0]

    importance = pd.DataFrame({
        'feature': X_instance.columns.tolist(),
        'shap_value': vals
    }).sort_values(by='shap_value', ascending=False)

    return {
        'top_positive': importance.head(5).to_dict('records'),
        'top_negative': importance.tail(5).to_dict('records')
    }


def plot_shap_summary(explainer, X_background: pd.DataFrame, max_display=15, save_path= os.path.join(visuals_path , "shap_summary.png")):
    """Generate and save SHAP Summary Plot"""
    shap_values = explainer.shap_values(X_background)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Clear any previous figure
    plt.clf()
    
    # Create new figure properly
    fig = plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_background, max_display=max_display, show=False)
    
    plt.title("AgriScore - SHAP Feature Importance (Summary Plot)", fontsize=14)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f" SHAP Summary Plot saved to: {save_path}")
    return save_path


def get_shap_interactions(explainer, X_instance: pd.DataFrame, top_n=6):
    """Compute top SHAP Interaction Effects"""
    shap_interaction_values = explainer.shap_interaction_values(X_instance)
    
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]
    
    interaction_matrix = shap_interaction_values[0]
    features = X_instance.columns.tolist()
    
    interactions = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            value = abs(interaction_matrix[i, j])
            if value > 0.01:
                interactions.append({
                    'feature1': features[i],
                    'feature2': features[j],
                    'interaction_strength': round(float(value), 4)
                })
    
    interactions = sorted(interactions, key=lambda x: x['interaction_strength'], reverse=True)
    return interactions[:top_n]



