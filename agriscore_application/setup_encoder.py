import pandas as pd
from preprocessing import create_synthetic_target, fit_encoder_and_save
import os 


#get the parent directory of the current file
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)


# get the path to the data folder
data_path = os.path.join(
    parent_dir,
    "Data sets"
)

print("Loading training data...")

# Change this to your actual training data file
df = pd.read_csv(os.path.join(data_path ,"Final_dataset.csv"))  

print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

# Create target
df = create_synthetic_target(df)

# Run encoder fitting
fit_encoder_and_save(df)   # No need to pass categorical_cols manually

print("✅ Encoder and feature columns saved successfully!")