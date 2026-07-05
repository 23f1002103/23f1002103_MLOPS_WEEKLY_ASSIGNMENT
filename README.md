# 23f1002103 MLOps Weekly Assignment - Week 3

## Feast Feature Store Integration with IRIS Pipeline

### Files in this repo:

**feast.ipynb**
Main notebook containing all the code for this assignment. Includes:
- Feast installation and setup
- Feature definitions and feast apply
- Training model using offline store
- Materializing features to online store
- Making predictions using online store
- BigQuery backend setup (Task 6)

**iris_feast/iris_feature_repo/feature_repo/iris_features.py**
Defines the Feast feature store components:
- Entity: iris_id
- Data Source: IRIS dataset
- Feature View: sepal_length, sepal_width, petal_length, petal_width, species

**iris_feast/iris_feature_repo/feature_repo/feature_store.yaml**
Feast configuration file. Configures:
- Offline store: BigQuery
- Online store: Google Datastore
- Registry: GCS bucket
