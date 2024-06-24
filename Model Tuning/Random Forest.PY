import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

best_model_rf = grid_search_rf.best_estimator_
print(f"Best Random Forest Model: {best_model_rf}")

y_pred_rf = best_model_rf.predict(X_val)
print(f"Best Random Forest Model - Accuracy: {accuracy_score(y_val, y_pred_rf):.4f}, F1-score: {f1_score(y_val, y_pred_rf, average='weighted'):.4f}")
print(classification_report(y_val, y_pred_rf))

test_predictions_rf = best_model_rf.predict(test_df_scaled)
submission_rf = pd.DataFrame({
    'ID': test_df.index,  
    'Target': test_predictions_rf
})

submission_rf.to_csv('submission_rf.csv', index=False)
