import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle

# Step 1: Load CSV data into a pandas DataFrame
file_path = 'train.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Step 2: Prepare features (X) and target (y)
# Example: If 'target_column' is your target variable
if 'target_column' in df.columns:
    X = df.drop(columns=['target_column'])  # Features
    y = df['target_column']  # Target
else:
    raise ValueError("Column 'target_column' not found in the dataset.")

# Step 3: Train Gradient Boosting Regressor
model = GradientBoostingRegressor()
model.fit(X, y)

# Step 4: Save the model to a pickle file
with open('gradient_boosting_regressor.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Gradient Boosting Regressor model saved to gradient_boosting_regressor.pkl")
