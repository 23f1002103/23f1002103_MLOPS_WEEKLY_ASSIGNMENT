git push -u origin week_1
git checkout main
git pull origin main
git rm inference.ipynb training.ipynb
git commit -m "Remove old Week 1 notebooks, keep main focused on current pipeline"
git push origin main
git checkout week_4
git pull origin week_4
cat README.md
cat > README.md << 'EOF'
# 23f1002103 MLOps Weekly Assignment

## Project Overview
This repository implements a reproducible, version-controlled, and continuously tested ML pipeline for IRIS flower classification — built incrementally across weekly assignments (DVC versioning, Feast feature store, and CI/CD automation).

## Week 2 — DVC Integration
Integrated Data Version Control (DVC) into the IRIS pipeline for reproducible, version-controlled data and model artifacts, backed by Google Cloud Storage (GCS).

**Approach:**
- Used DVC to version control the IRIS dataset and trained models
- Configured Google Cloud Storage as the DVC remote backend
- Created two versions of data and models to demonstrate versioning
- Demonstrated switching between versions using Git + DVC checkout

**Key files:**
- `train.py` → training script for the IRIS classification model
- `data/v1data.csv.dvc`, `data/v2data.csv.dvc` → DVC pointers to dataset versions
- `model.joblib.dvc` → DVC pointer to the trained model artifact
- `.dvc/config` → DVC remote configuration pointing to the GCS bucket
- `.dvcignore`, `.gitignore` → files excluded from DVC/Git tracking

**GCS remote storage:** `gs://mlops-iris-csv-pipeline-unique/dvc-storage`

**Switching versions:**
git checkout v1.0

nano README.md
[200~git add README.md
git commit -m "Update README to reflect DVC and CI/CD work"
git push origin week_4~
rm key.json
git add README.md
git commit -m "Update README to reflect DVC and CI/CD work"
git push origin week_4
git checkout week_4
git pull origin week_4
git merge main --allow-unrelated-histories
git checkout --ours README.md
git add README.md
git commit -m "Merge main into week_4, resolve README conflict"
git push origin week_4
git checkout week_4
sed -i '1i # CI/CD pipeline verified for Week 4 assignment' train.py
head -3 train.py
git add train.py
git commint -m ""Add verification comment to train.py"

git commint -m ""Add verification comment to train.py"
git commit -m "Add verification comment to train.py"
git push week_4
git push origin week_4
pwd
ls
ls
pwd
clear
clear
git checkout
clear
git checkout
git checkout
sed -i '1i #Test suite for IRIS data validation and model evalution - Week4' test_model.py
head -3 test_model.py
git add test_model.py
git commit -m "Add descriptive comment"
git push origin week_4
clear
ls
git checkout
sed -i '2i # Includes checks for missing values, schema consistency, and minimum accuracy thresholds' test_model.py
head -4 test_model.py
git add test_model.py
git commit -m "Add coverage summary comment to test suite"
git push origin week_4
git push origin week_4
git pull origin week_4
git config pull.rebase false
git pull origin week_4
cat test_model.py
nano test_model.py
git add test_model.py
git commit -m "Resolve merge conflict in test file header comment"
git push origin week_4
nano test_model.py
[200~grep -n "<<<<<<<\|=======\|>>>>>>>" test_model.py~
grep -n "<<<<<<<" test_model.py
git add test_model.py
git commit -m "Resolve merge conflict in test file header comment"
git push origin week_4
git status
head -10 train.py
clear
git checkout
sed -i '2i # Model retraining and evaluation validated through automated GitHub Actions workflow' train.py
head -4 train.py
git add train.py
git commit -m "Add CI/DC verification note to train.py"
git push origin week_4
pwd
ls
git checkout
ls
git checkout main
git pull
git checkout -b week_5
pip install mlflow
mlflow --version
gsutil mkdir gs://mlops-iris-csv-pipeline-unique/mlflow/
mlflow server   --backend-store-uri gs://mlops-iris-csv-pipeline-unique/mlflow/   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000
ps aux | grep mlflow
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000 &
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000   --serve-artifacts &
pkill -f mlflow
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000   --allowed-hosts "*" &
ps aux | grep train
python train.py v1data.csv
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root ./mlflow-artifacts   --host 0.0.0.0   --port 5000   --allowed-hosts "*" &
python train.py v1data.csv
python train.py v1data.csv
print("1")
import pandas as pd
print("2")
from sklearn.tree import DecisionTreeClassifier
print("3")
from sklearn.model_selection import train_test_split
print("4")
from sklearn import metrics
print("5")
import joblib
print("6")
import sys
print("7")
import csv
print("8")
import mlflow
print("9")
import mlflow.sklearn
print("10")
clear
python -u train.py v1data.csv
python - <<EOF
from mlflow import MlflowClient

client = MlflowClient(tracking_uri="http://34.135.49.194:5000")

exp = client.get_experiment_by_name("iris-experiment")
print("Experiment ID:", exp.experiment_id)

runs = client.search_runs([exp.experiment_id])

for r in runs:
    print("="*50)
    print("Run ID:", r.info.run_id)
    print("Metrics:", r.data.metrics)
    print("Params:", r.data.params)
EOF

ls data/
python train.py v1data.csv
pkill -f train
python -c "
import mlflow
mlflow.set_tracking_uri('http://34.135.49.194:5000')
mlflow.set_experiment('test')
print('MLflow connected!')
"
pkill -f train
python train.py v1data.csv
cat > train.py << 'EOF'
# CI/CD pipeline verified for Week 5 assignment
# MLflow experiment tracking and model registry integrated
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import sys
import csv
import mlflow
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("http://34.135.49.194:5000")
mlflow.set_experiment("iris-experiment")

data_file = sys.argv[1] if len(sys.argv) > 1 else "v1data.csv"

# Load data
data = pd.read_csv(f'data/{data_file}')
print(f"Data loaded: {data_file}, Shape: {data.shape}")

# Prepare data
X = data.drop('species', axis=1)
y = data['species']
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Save evaluation set
eval_data = X_eval.copy()
eval_data['species'] = y_eval
eval_data.to_csv('eval_data.csv', index=False)
print("Saved eval_data.csv!")

# Hyperparameter configurations to try
hyperparameter_configs = [
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 5, "min_samples_split": 4},
    {"max_depth": None, "min_samples_split": 2},
]

best_accuracy = 0
best_run_id = None

for config in hyperparameter_configs:
    with mlflow.start_run():
        max_depth = config["max_depth"]
        min_samples_split = config["min_samples_split"]

        # Log hyperparameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("data_file", data_file)

        # Train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_eval)
        accuracy = metrics.accuracy_score(y_eval, y_pred)
        precision = metrics.precision_score(y_eval, y_pred, average='weighted')
        recall = metrics.recall_score(y_eval, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"Config: {config} => Accuracy: {accuracy:.4f}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris-model"
        )

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = mlflow.active_run().info.run_id

# Save best metrics to CSV (for CI compatibility)
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['accuracy', 'precision', 'recall'])
    writer.writerow([best_accuracy, precision, recall])
print("Saved metrics.csv!")

# Also save model locally for CI fallback
joblib.dump(model, 'model.joblib')
print(f"Best accuracy: {best_accuracy:.4f}, Run ID: {best_run_id}")
print("All experiments logged to MLflow!")
EOF

cat > test_model.py << 'EOF'
# Test suite for IRIS - Week 5
# Fetches model from MLflow Model Registry
import pandas as pd
import csv
import os
import mlflow
import mlflow.sklearn

DATA_PATH = "data/v1data.csv"
EVAL_DATA_PATH = "eval_data.csv"
METRICS_PATH = "metrics.csv"

MLFLOW_TRACKING_URI = "http://34.135.49.194:5000"
MODEL_NAME = "iris-model"
MODEL_ALIAS = "models:/iris-model/latest"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ---------- Task 1: Data validation tests ----------

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Data file is missing"

def test_data_schema():
    data = pd.read_csv(DATA_PATH)
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_columns.issubset(set(data.columns))

def test_no_missing_values():
    data = pd.read_csv(DATA_PATH)
    assert data.isnull().sum().sum() == 0

def test_feature_types():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(data[col])

def test_value_ranges():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert (data[col] > 0).all()

# ---------- Task 2: Model evaluation tests (from MLflow) ----------

def test_model_loads_from_registry():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    assert model is not None, "Could not load model from MLflow registry"

def test_eval_data_exists():
    assert os.path.exists(EVAL_DATA_PATH), "eval_data.csv is missing"

def test_model_predicts():
    model = mlflow.sklearn.load_model(MODEL_ALIAS)
    data = pd.read_csv(EVAL_DATA_PATH)
    X = data.drop("species", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(X)

def test_accuracy_threshold():
    assert os.path.exists(METRICS_PATH), "metrics.csv is missing"
    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    accuracy = float(row["accuracy"])
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy}"
EOF

clear
python train.py v1data.csv
python train.py v1data.csv
gsutil ls gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/
echo "test" > /tmp/test.txt
gsutil cp /tmp/test.txt gs://mlops-iris-csv-pipeline-unique/mlflow/test.txt
pkill -f mlflow
pkill -f train
rm -f mlflow.db
mlflow server   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root gs://mlops-iris-csv-pipeline-unique/mlflow/artifacts/   --host 0.0.0.0   --port 5000   --allowed-hosts "*" &
pkill -f mlflow
curl http://34.135.49.194:5000/health
ps aux | grep train
pwd
ls
find . -type f \( -name "*.py" -o -name "dvc.yaml" -o -name "*.dvc" \) | grep -v __pycache__
cat train.py
cat test_model.py
cat dvc.yaml
dvc remove model.joblib.dvc
git rm model.joblib.dvc
cat > train.py << 'EOF'
# CI/CD pipeline verified for Week 5 assignment
# MLflow experiment tracking and model registry integrated
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import joblib
import sys
import csv
import mlflow
import mlflow.sklearn

# MLflow setup
mlflow.set_tracking_uri("http://34.135.49.194:5000")
mlflow.set_experiment("iris-experiment")

data_file = sys.argv[1] if len(sys.argv) > 1 else "v1data.csv"

# Load data
data = pd.read_csv(f'data/{data_file}')
print(f"Data loaded: {data_file}, Shape: {data.shape}")

# Prepare data
X = data.drop('species', axis=1)
y = data['species']
X_train, X_eval, y_train, y_eval = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Save evaluation set
eval_data = X_eval.copy()
eval_data['species'] = y_eval
eval_data.to_csv('eval_data.csv', index=False)
print("Saved eval_data.csv!")

# Hyperparameter configurations to try
hyperparameter_configs = [
    {"max_depth": 3, "min_samples_split": 2},
    {"max_depth": 5, "min_samples_split": 4},
    {"max_depth": None, "min_samples_split": 2},
]

best_accuracy = 0
best_run_id = None

for config in hyperparameter_configs:
    with mlflow.start_run():
        max_depth = config["max_depth"]
        min_samples_split = config["min_samples_split"]

        # Log hyperparameters
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("data_file", data_file)

        # Train model
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_eval)
        accuracy = metrics.accuracy_score(y_eval, y_pred)
        precision = metrics.precision_score(y_eval, y_pred, average='weighted')
        recall = metrics.recall_score(y_eval, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        print(f"Config: {config} => Accuracy: {accuracy:.4f}")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="iris-model"
        )

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_run_id = mlflow.active_run().info.run_id

# Save best metrics to CSV (for CI compatibility)
with open('metrics.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['accuracy', 'precision', 'recall'])
    writer.writerow([best_accuracy, precision, recall])
print("Saved metrics.csv!")

# Also save model locally for CI fallback
joblib.dump(model, 'model.joblib')
print(f"Best accuracy: {best_accuracy:.4f}, Run ID: {best_run_id}")
print("All experiments logged to MLflow!")
EOF

curl http://34.135.49.194:5000/health
tail -f mlflow.log
ps aux | grep train.py
tail -f mlflow.log
watch -n 5 'ps aux | grep train.py | grep -v grep && echo "---" && cat /proc/28891/net/dev | awk "NR>2 {print $1, \"RX:\", $2, \"TX:\", $10}"'
python3 -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()
for exp in experiments:
    print('Experiment:', exp.name)
    runs = client.search_runs(exp.experiment_id)
    print('Runs so far:', len(runs))
    for run in runs:
        print(' -', run.info.run_id, run.info.status, run.data.metrics)
"
# First run evaluate.py to update metrics.csv
python evaluate.py
# Check what metrics.csv now shows
cat metrics.csv
# Then push changes
git add evaluate.py test_model.py metrics.csv
git commit -m "fix: use holdout split in evaluate.py, update CI threshold to 0.80"
git push origin week_5
PROJECT_ID=project-b38b370e-25fb-420a-a54
for ROLE in roles/viewer roles/container.viewer; do   gcloud projects add-iam-policy-binding "$PROJECT_ID"     --member="user:da5014_1@study.iitm.ac.in"     --role="$ROLE"; done
gcloud auth login
