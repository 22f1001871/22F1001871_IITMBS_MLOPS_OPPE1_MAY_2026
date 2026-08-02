from feast import Entity
from feast import FeatureView
from feast import Field
from feast.types import Float32
from feast.types import Int64
from feast.infra.offline_stores.file_source import FileSource
from datetime import timedelta

stock_source = FileSource(
    path="data/train_v1.parquet",
    timestamp_field="timestamp",
)

stock = Entity(
    name="stock_name",
    join_keys=["stock_name"],
)

stock_features = FeatureView(
    name="stock_features",
    entities=[stock],
    ttl=timedelta(days=365),
    schema=[
        Field(name="rolling_avg_10", dtype=Float32),
        Field(name="volume_sum_10", dtype=Int64),
    ],
    source=stock_source,
    online=True,
)