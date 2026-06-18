# 23f1002103 MLOps Weekly Assignment - Week 1

## Files

### training.ipynb
- Fetches IRIS dataset from GCS bucket
- Trains a DecisionTreeClassifier model
- Saves model artifacts to GCS with timestamp folder

### inference.ipynb
- Fetches trained model from GCS artifacts folder
- Runs inference on evaluation set
- Prints accuracy and classification report

## GCS Bucket Structure
gs://mlops-iris-csv-pipeline-unique/
├── data/iris.csv
└── artifacts/
    ├── 20260617_164425/model.joblib
    └── 20260618_111618/model.joblib
