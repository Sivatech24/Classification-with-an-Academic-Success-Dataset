import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

test_df = pd.read_csv('test.csv')

label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

for col in test_df.columns:
    if test_df[col].isnull().sum() > 0:
        if test_df[col].dtype == 'object':
            test_df[col].fillna(test_df[col].mode()[0], inplace=True)
        else:
            test_df[col].fillna(test_df[col].mean(), inplace=True)

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        if col in label_encoders:
            test_df[col] = label_encoders[col].transform(test_df[col])
        else:
            raise ValueError(f"Label encoder not found for column '{col}' in test data.")

test_df_scaled = scaler.transform(test_df)

best_model_gb = joblib.load('best_model_gb.pkl')

test_predictions_gb = best_model_gb.predict(test_df_scaled)

submission_gb = pd.DataFrame({
    'ID': test_df.index,  
    'Target': test_predictions_gb
})

label_encoders = {}
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        label_encoders[col] = LabelEncoder()
        train_df[col] = label_encoders[col].fit_transform(train_df[col])

joblib.dump(label_encoders, 'label_encoders.pkl')

submission_gb.to_csv('submission_gb.csv', index=False)

print("Predictions saved successfully.")
