import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from preprocess import preprocess

# Load and preprocess data
df = preprocess("data/sample.csv")

# Create a dummy target column for training
df['target'] = [0 if i % 2 == 0 else 1 for i in range(len(df))]

X = df[['age', 'height']]
y = df['target']

# Train a simple model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and print accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Training completed. Accuracy: {acc}")
