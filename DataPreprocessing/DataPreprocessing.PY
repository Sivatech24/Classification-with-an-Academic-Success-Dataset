import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print("Train DataFrame Info:")
train_df.info()
print("\nTest DataFrame Info:")
test_df.info()

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
print("\nPreprocessed Training Features (X_train):")
print(X_train[:5])
print("\nPreprocessed Validation Features (X_val):")
print(X_val[:5])
print("\nPreprocessed Test Features (test_df_scaled):")
print(test_df_scaled[:5])
print("\nPreprocessed Target (y_train):")
print(y_train[:5])
