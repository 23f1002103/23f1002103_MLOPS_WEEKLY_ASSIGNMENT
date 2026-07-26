from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.types import Float64, String

# DATA SOURCE — now pointing to BigQuery!
iris_source = BigQuerySource(
    table="project-b38b370e-25fb-420a-a54.feast_iris_store.iris_features",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# ENTITY
iris_entity = Entity(
    name="iris_id",
    join_keys=["iris_id"],
)

# FEATURE VIEW
iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width",  dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width",  dtype=Float64),
        Field(name="species",      dtype=String),
    ],
    source=iris_source,
)
