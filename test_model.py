# Test suite for IRIS data validation and model evaluation - Week 4
# Includes checks for missing values, schema consistency, and minimum accuracy thresholds
import joblib
import pandas as pd
import csv
import os

DATA_PATH = "data/v1data.csv"
EVAL_DATA_PATH = "eval_data.csv"
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.csv"

# ---------- Task 1: Data validation tests ----------

def test_data_file_exists():
    assert os.path.exists(DATA_PATH), "Data file is missing"

def test_data_schema():
    data = pd.read_csv(DATA_PATH)
    expected_columns = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert expected_columns.issubset(set(data.columns)), f"Missing columns: {expected_columns - set(data.columns)}"

def test_no_missing_values():
    data = pd.read_csv(DATA_PATH)
    assert data.isnull().sum().sum() == 0, "Data contains missing values"

def test_feature_types():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(data[col]), f"{col} is not numeric"

def test_value_ranges():
    data = pd.read_csv(DATA_PATH)
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in numeric_cols:
        assert (data[col] > 0).all(), f"{col} has non-positive values"

# ---------- Task 2: Model evaluation tests ----------

def test_model_file_exists():
    assert os.path.exists(MODEL_PATH), "Trained model file is missing"

def test_eval_data_exists():
    assert os.path.exists(EVAL_DATA_PATH), "eval_data.csv is missing — run train.py first"

def test_model_loads():
    model = joblib.load(MODEL_PATH)
    assert model is not None

def test_model_predicts():
    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(EVAL_DATA_PATH)
    X = data.drop("species", axis=1)
    predictions = model.predict(X)
    assert len(predictions) == len(X), "Prediction count doesn't match input rows"

def test_accuracy_threshold():
    assert os.path.exists(METRICS_PATH), "metrics.csv is missing — run train.py first"
    with open(METRICS_PATH) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    accuracy = float(row["accuracy"])
    assert accuracy >= 0.90, f"Accuracy too low: {accuracy}"

def test_eval_data_exists():
    assert os.path.exists(EVAL_DATA_PATH), "eval_data.csv is missing — run train.py first"
