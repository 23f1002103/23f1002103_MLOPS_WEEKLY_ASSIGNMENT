import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_mlsecops_poisoning")

DATASETS = {
    0: "iris_clean.csv",
    5: "iris_poisoned_5.csv",
    10: "iris_poisoned_10.csv",
    50: "iris_poisoned_50.csv",
}

for poison_level, filename in DATASETS.items():
    data = pd.read_csv(f"data/{filename}")
    print(f"\n--- Poison level: {poison_level}% | File: {filename} | Shape: {data.shape} ---")

    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name=f"poison_{poison_level}pct"):
        mlflow.log_param("poison_level_pct", poison_level)
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        mlflow.log_param("random_state", 42)

        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_eval)
        accuracy = metrics.accuracy_score(y_eval, y_pred)
        precision = metrics.precision_score(y_eval, y_pred, average='weighted', zero_division=0)
        recall = metrics.recall_score(y_eval, y_pred, average='weighted', zero_division=0)
        f1 = metrics.f1_score(y_eval, y_pred, average='weighted', zero_division=0)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Accuracy: {accuracy*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

print("\nAll runs logged to MLflow experiment 'iris_mlsecops_poisoning'.")
