import pandas as pd
import cuml
from cuml.preprocessing import LabelEncoder, StandardScaler
from cuml.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

for col in train_df.columns:
    if train_df[col].isnull().sum() > 0:
        if train_df[col].dtype == 'object':
            train_df[col].fillna(train_df[col].mode()[0], inplace=True)
        else:
            train_df[col].fillna(train_df[col].mean(), inplace=True)

for col in test_df.columns:
    if test_df[col].isnull().sum() > 0:
        if test_df[col].dtype == 'object':
            test_df[col].fillna(test_df[col].mode()[0], inplace=True)
        else:
            test_df[col].fillna(test_df[col].mean(), inplace=True)

label_encoders = {}
for col in train_df.columns:
    if train_df[col].dtype == 'object':
        label_encoders[col] = LabelEncoder()
        train_df[col] = label_encoders[col].fit_transform(train_df[col])

for col in test_df.columns:
    if test_df[col].dtype == 'object':
        if col in label_encoders:
            test_df[col] = label_encoders[col].transform(test_df[col])
        else:
            label_encoders[col] = LabelEncoder()
            test_df[col] = label_encoders[col].fit_transform(test_df[col])

X = train_df.drop(columns=['Target'])
y = train_df['Target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_df_scaled = scaler.transform(test_df)

X_train = cuml.DataFrame.from_pandas(pd.DataFrame(X_train))
X_val = cuml.DataFrame.from_pandas(pd.DataFrame(X_val))
y_train = cuml.Series(y_train)
y_val = cuml.Series(y_val)
test_df_scaled = cuml.DataFrame.from_pandas(pd.DataFrame(test_df_scaled))

model = RandomForestClassifier(n_estimators=100, max_depth=20)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val.to_pandas(), y_pred.to_pandas())
f1 = f1_score(y_val.to_pandas(), y_pred.to_pandas(), average='weighted')

print(f"Random Forest - Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
print(classification_report(y_val.to_pandas(), y_pred.to_pandas()))

test_predictions = model.predict(test_df_scaled)

submission = pd.DataFrame({
    'ID': test_df.index,
    'Target': test_predictions.to_pandas()
})

submission.to_csv('submission.csv', index=False)
