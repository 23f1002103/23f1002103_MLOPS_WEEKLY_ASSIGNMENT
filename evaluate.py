import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
import csv

# Point to MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from registry
model_name = "iris-model"
model_uri = f"models:/{model_name}/latest"
print(f"Loading model from MLflow Registry: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print(f"Model loaded successfully: {type(model).__name__}")

# Load same dataset used in training
data = pd.read_csv("data/v1data.csv")

# Same split as train.py
X = data.drop("species", axis=1)
y = data["species"]

X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42  # same random_state as train.py
)

# Evaluate on holdout only
predictions = model.predict(X_eval)

acc = accuracy_score(y_eval, predictions)
precision = precision_score(y_eval, predictions, average='weighted')
recall = recall_score(y_eval, predictions, average='weighted')

print(f"\n===== Evaluation Results =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_eval, y_eval, target_names=data["species"].unique()))

# Save metrics.csv — same data as MLflow
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["accuracy", "precision", "recall"])
    writer.writerow([acc, precision, recall])

print("Saved metrics.csv")