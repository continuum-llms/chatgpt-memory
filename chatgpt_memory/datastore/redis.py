from typing import Any, Dict, List, Union
from uuid import uuid4
import numpy as np
import redis

from redis.commands.search.field import VectorField, TextField, TagField
from redis.commands.search.query import Query

from chatgpt_memory.datastore.config import RedisDataStoreConfig
from chatgpt_memory.datastore.datastore import DataStore


class RedisDataStore(DataStore):
    def __init__(self, config: RedisDataStoreConfig, do_flush_data: bool = False):
        super().__init__(config=config)
        self.config = config

        self.do_flush_data = do_flush_data

    def connect(self):
        """
        Connect to the Redis server.
        """
        connection_pool = redis.ConnectionPool(**self.config.dict())
        self.redis_connection = redis.Redis(connection_pool=connection_pool)
        
        # flush data only once after establishing connection
        if self.do_flush_data:
            self.redis_connection.flushall()
            self.do_flush_data = False

    def create_index(
        self,
        number_of_vectors: int,
        index_fields: Union[TagField, TextField],
        M=40,
        EF=200,
    ):
        """
        Creates a Redis index with a dense vector field.

        Args:
            number_of_vectors (int): Number of vectors to be indexed.
            index_fields (Union[TagField, TextField]): List of fields including the
            metadata to be stored in Redis.
            M (int, optional): Defaults to 40.
            EF (int, optional): Defaults to 200.
        """
        self.redis_connection.ft().create_index(
            [
                VectorField(
                    self.config.vector_field_name,
                    self.config.index_type,
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.config.vector_dimensions,
                        "DISTANCE_METRIC": self.config.distance_metric,
                        "INITIAL_CAP": number_of_vectors,
                        "M": M,
                        "EF_CONSTRUCTION": EF,
                    },
                ),
            ]
            + index_fields
        )

    def index_documents(self, documents: List[Dict]):
        """
        Indexes the set of documents.

        Args:
            documents (List[Dict]): _description_
        """
        redis_pipeline = self.redis_connection.pipeline(transaction=False)
        for document in documents:
            redis_pipeline.hset(uuid4().hex, mapping=document)
        redis_pipeline.execute()

    def search_documents(
        self,
        query_vector: np.ndarray,
        topk: int = 5,
        result_fields: List[str] = ["text", "vector_score"],
    ) -> List[Any]:
        """
        Searches the redis index using the query vector.

        Args:
            query_vector (np.ndarray): Embedded query vector.
            topk (int, optional): Number of results. Defaults to 5.
            result_fields (int, optional): Name of the fields that you want to be returned
            from the search result documents

        Returns:
            List[Any]: Search result documents.
        """
        query = (
            Query(
                f"*=>[KNN {topk} @{self.config.vector_field_name} $vec_param AS vector_score]"
            )
            .sort_by("vector_score")
            .paging(0, topk)
            .return_fields(
                # parse `result_fields` as strings separated by comma to pass as params
                eval(
                    " , ".join([f'"{result_field}"' for result_field in result_fields])
                )
            )
            .dialect(2)
        )
        params_dict = {"vec_param": query_vector}
        result_documents = (
            self.redis_connection.ft().search(query, query_params=params_dict).docs
        )

        return result_documents
