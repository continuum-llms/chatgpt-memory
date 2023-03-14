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
    distance_metric: = "L2"
    {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "BLOCK_SIZE":number_of_vectors }