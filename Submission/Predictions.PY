import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

test_df = pd.read_csv('test.csv')

label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

categorical_columns = ['categorical_column_1', 'categorical_column_2']
numerical_columns = ['numerical_column_1', 'numerical_column_2']

for col in categorical_columns:
    test_df[col] = label_encoders[col].transform(test_df[col])

test_df[numerical_columns] = scaler.transform(test_df[numerical_columns])

model = joblib.load('gradient_boosting_model.pkl')

predictions = model.predict(test_df)

submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'Prediction': predictions
})

submission_df.to_csv('submission.csv', index=False)

print("Predictions saved successfully to submission.csv")
