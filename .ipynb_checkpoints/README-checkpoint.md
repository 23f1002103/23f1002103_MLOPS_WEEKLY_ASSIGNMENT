# 23f1002103 MLOps Weekly Assignment - Week 2

## DVC Integration with IRIS Pipeline

## Problem Statement
Integrate Data Version Control (DVC) into the IRIS machine learning pipeline to create a reproducible, version-controlled workflow for both data and model artifacts backed by Google Cloud Storage (GCS).

## Approach
- Used DVC to version control IRIS dataset and trained models
- Configured Google Cloud Storage as DVC remote backend
- Created two versions of data and models to demonstrate versioning
- Demonstrated switching between versions using Git + DVC checkout

## Repository Structure
- `train.py` → Training script for IRIS classification model
- `data/v1data.csv.dvc` → DVC pointer to version 1 of IRIS dataset (101 rows)
- `data/v2data.csv.dvc` → DVC pointer to version 2 of IRIS dataset
- `model.joblib.dvc` → DVC pointer to trained model artifact
- `.dvc/config` → DVC remote configuration pointing to GCS bucket
- `.dvcignore` → Files ignored by DVC
- `.gitignore` → Files ignored by Git (actual data and model files)

## GCS Remote Storage
Actual data and model files are stored in:
gs://mlops-iris-csv-pipeline-unique/dvc-storage

## How to Reproduce
1. Clone this repository
2. Install DVC: `pip install dvc dvc-gs`
3. Pull data from GCS: `dvc pull`
4. Train model: `python train.py v1data.csv`

## Version History
- v1.0 → Model trained on v1data.csv (101 rows)
- v2.0 → Model trained on v2data.csv

## Switching Versions
# Go to version 1
git checkout v1.0
dvc checkout

# Go back to latest
git checkout master
dvc checkout

## Learnings
- DVC extends Git to handle large files and ML models
- GCS can be used as DVC remote storage backend
- Version switching allows instant rollback to any data/model version

