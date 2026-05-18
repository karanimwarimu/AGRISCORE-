import pandas as pd
import joblib
import shap
from preprocessing import preprocess_for_scoring
from models import load_model   
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

print("Loading training data for explainer...")
df = pd.read_csv(os.path.join(data_path , "Final_dataset.csv"))

# Preprocess to get correct feature matrix
X = preprocess_for_scoring(df, is_training=True)

# Load your model
model_cv = load_model()      

# Extract the best actual LightGBM model
best_model = model_cv.best_estimator_

# Create and save proper explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(best_model)

# Save the explainer object correctly
joblib.dump(explainer, os.path.join(model_path , "agriscore_SHAP_explainer.pkl"))
print(" SHAP Explainer saved successfully!")