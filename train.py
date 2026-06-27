import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import sys

# Get data file from command line
# Round 1: python train.py v1data.csv
# Round 2: python train.py v2data.csv
data_file = sys.argv[1]

# Load data
data = pd.read_csv(f'data/{data_file}')
print(f"Data loaded: {data_file}, Shape: {data.shape}")

# Prepare data
X = data.drop('species', axis=1)
y = data['species']
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Evaluation samples: {len(X_eval)}")

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Evaluate
y_pred = model.predict(X_eval)
accuracy = metrics.accuracy_score(y_eval, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model locally
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib!")