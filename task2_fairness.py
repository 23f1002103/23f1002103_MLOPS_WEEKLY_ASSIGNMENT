import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from fairlearn.metrics import MetricFrame

# Load data with location column
df = pd.read_csv("data/iris_with_location.csv")

# Features WITHOUT location (location is NOT used for training)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']
sensitive = df['location']

# Split — keep sensitive attribute aligned with train/test split
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive, test_size=0.2, random_state=42, stratify=y
)

# Train model (normal training, no location used)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Overall accuracy
print("Overall accuracy:", accuracy_score(y_test, y_pred))

# Fairlearn: check metrics disaggregated by location
metrics = {
    'accuracy': accuracy_score,
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro', zero_division=0),
    'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
}

mf = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sens_test
)

print("\n--- Metrics by location group ---")
print(mf.by_group)

print("\n--- Overall metrics ---")
print(mf.overall)

# Save the model and test data for use in Task 3 (SHAP)
import joblib
joblib.dump(clf, "iris_rf_model.bin")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)