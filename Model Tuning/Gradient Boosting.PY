import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search_gb = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_gb.fit(X_train, y_train)

best_model_gb = grid_search_gb.best_estimator_
print(f"Best Gradient Boosting Model: {best_model_gb}")

y_pred_gb = best_model_gb.predict(X_val)
print(f"Best Gradient Boosting Model - Accuracy: {accuracy_score(y_val, y_pred_gb):.4f}, F1-score: {f1_score(y_val, y_pred_gb, average='weighted'):.4f}")
print(classification_report(y_val, y_pred_gb))

test_predictions_gb = best_model_gb.predict(test_df_scaled)
submission_gb = pd.DataFrame({
    'ID': test_df.index,  
    'Target': test_predictions_gb
})

submission_gb.to_csv('submission_gb.csv', index=False)
