import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data from CSV file
file_path = 'train.csv'  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Check data types of columns
print(df.dtypes)

# Initialize LabelEncoders
label_encoders = {}

# Iterate over each categorical column
for column in df.select_dtypes(include=['object']).columns:
    # Convert column to string type if not already
    df[column] = df[column].astype(str)
    
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    
    # Fit and transform column
    df[column + '_encoded'] = label_encoder.fit_transform(df[column])
    
    # Store LabelEncoder in dictionary
    label_encoders[column] = label_encoder

# Save LabelEncoders to a pickle file
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("LabelEncoders saved to label_encoders.pkl")
