import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import sys
import csv
import mlflow
import mlflow.sklearn

# -------------------------------------------------
# MLflow Setup
# -------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("iris-experiment")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
data_file = sys.argv[1] if len(sys.argv) > 1 else "v1data.csv"

data = pd.read_csv(f"data/{data_file}")
print(f"Loaded dataset: {data_file}")
print(f"Dataset Shape: {data.shape}")

# -------------------------------------------------
# Split Data
# -------------------------------------------------
X = data.drop("species", axis=1)
y = data["species"]

X_train, X_eval, y_train, y_eval = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------------------------------
# Save Evaluation Dataset
# -------------------------------------------------
eval_data = X_eval.copy()
eval_data["species"] = y_eval
eval_data.to_csv("eval_data.csv", index=False)
print("Saved eval_data.csv")

# -------------------------------------------------
# Hyperparameter Configurations
# -------------------------------------------------
hyperparameter_configs = [
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 5, "min_samples_split": 4}
]

# -------------------------------------------------
# Variables for Best Model
# -------------------------------------------------
best_accuracy = 0
best_precision = 0
best_recall = 0
best_model = None
best_run_id = None

print("\nStarting MLflow Experiments...\n")

# -------------------------------------------------
# Training Loop — Log params/metrics only, NO GCS upload
# -------------------------------------------------
for config in hyperparameter_configs:

    print("=" * 60)
    print(f"Running Configuration: {config}")

    with mlflow.start_run():

        # Log Parameters
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("min_samples_split", config["min_samples_split"])
        mlflow.log_param("data_file", data_file)

        # Train Model
        print("Training model...")
        model = DecisionTreeClassifier(
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate Model
        print("Evaluating model...")
        predictions = model.predict(X_eval)

        accuracy = metrics.accuracy_score(y_eval, predictions)
        precision = metrics.precision_score(y_eval, predictions, average="weighted")
        recall = metrics.recall_score(y_eval, predictions, average="weighted")

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")

        # Log Metrics only (no GCS upload here)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Track Best Model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_precision = precision
            best_recall = recall
            best_model = model
            best_run_id = mlflow.active_run().info.run_id

# -------------------------------------------------
# Register ONLY Best Model to MLflow Registry (1 GCS upload)
# -------------------------------------------------
print("\nUploading best model to MLflow Registry...")
with mlflow.start_run(run_id=best_run_id):
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="iris-model"
    )
print("Best model uploaded successfully.")

# -------------------------------------------------
# Save Metrics
# -------------------------------------------------
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["accuracy", "precision", "recall"])
    writer.writerow([best_accuracy, best_precision, best_recall])

print("Saved metrics.csv")

# -------------------------------------------------
# Save Best Model Locally
# -------------------------------------------------
joblib.dump(best_model, "model.joblib")
print("Saved model.joblib")

# -------------------------------------------------
# Final Summary
# -------------------------------------------------
print("\n" + "=" * 60)
print("Training Completed Successfully")
print(f"Best Accuracy : {best_accuracy:.4f}")
print(f"Best Run ID   : {best_run_id}")
print("All experiments logged to MLflow.")
print("=" * 60)