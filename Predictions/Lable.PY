import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

data_df = pd.read_csv('train.csv')
categorical_columns = ['categorical_column_1', 'categorical_column_2', 'categorical_column_3']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    le.fit(data_df[col])
    label_encoders[col] = le
    data_df[col] = le.transform(data_df[col])
current_dir = os.getcwd()
file_path = os.path.join(current_dir, 'label_encode.pkl')
joblib.dump(label_encoders, file_path)
print(f"LabelEncoders saved to '{file_path}' successfully.")
