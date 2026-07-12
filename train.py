# CI/CD pipeline verified for Week 4 assignment
# Model retraining and evaluation validated through automated GitHub Actions workflow
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import sys
import csv

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

# Save evaluation set for CI testing
eval_data = X_eval.copy()
eval_data['species'] = y_eval
eval_data.to_csv('eval_data.csv', index=False)
print("Saved eval_data.csv!")

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# Evaluate
y_pred = model.predict(X_eval)
accuracy = metrics.accuracy_score(y_eval, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Compute extra metrics
precision = metrics.precision_score(y_eval, y_pred, average='weighted')
recall = metrics.recall_score(y_eval, y_pred, average='weighted')

# Save metrics to CSV
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['accuracy', 'precision', 'recall'])
    writer.writerow([accuracy, precision, recall])
print("Saved metrics.csv!")

# Save model locally
joblib.dump(model, 'model.joblib')
print("Model saved as model.joblib!")