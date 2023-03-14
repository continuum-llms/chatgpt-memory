from pydantic import BaseModel
from enum import Enum


class RedisIndexType(Enum):
    hnsw = "HNSW"
    flat = "FLAT"


class DataStoreConfig(BaseModel):
    host: str
    port: int
    password: str


class RedisDataStoreConfig(DataStoreConfig):
    index_type: str = RedisIndexType.hnsw.value
    vector_field_name: str
    vector_dimensions: int
    distance_metric: str = "L2"
