import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load data from CSV file
file_path = 'train.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Check data types of columns
print(df.dtypes)

# Convert columns with mixed types to numeric (if necessary)
for column in df.columns:
    if df[column].dtype == 'object':
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce')  # coerce to NaN for non-numeric
        except ValueError:
            print(f"Column '{column}' could not be converted to numeric.")

# Select only numeric columns for scaling
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Check if there are any numerical columns to scale
if not numerical_columns.empty:
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Fit scaler to the numerical columns
    scaler.fit(df[numerical_columns])

    # Save scaler to a pickle file
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Scaler saved to scaler.pkl")
else:
    print("No numerical columns found to scale.")
