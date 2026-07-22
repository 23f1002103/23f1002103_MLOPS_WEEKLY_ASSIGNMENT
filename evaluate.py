import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Point to MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from registry by name
model_name = "iris-model"
model_uri = f"models:/{model_name}/latest"

print(f"Loading model from MLflow Registry: {model_uri}")
model = mlflow.sklearn.load_model(model_uri)
print(f"Model loaded successfully: {type(model).__name__}")

# Load iris dataset with correct column names matching training
iris = load_iris()
X = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

# String labels to match training
label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
y = [label_map[i] for i in iris.target]

# Run predictions
y_pred = model.predict(X)

# Calculate metrics
acc = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')

print(f"\n===== Evaluation Results =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"\nClassification Report:")
print(classification_report(y, y_pred))
