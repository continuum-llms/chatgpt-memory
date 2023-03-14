import redis
from chatgpt_memory.datastore.config import DataStoreConfig

from chatgpt_memory.datastore.datastore import DataStore


class RedisDataStore(DataStore):
    def __init__(self, config: DataStoreConfig):
        super().__init__(config=config)

    def connect(self) -> redis.Redis:
        connection_pool = redis.ConnectionPool(**self.config.dict())
        redi_datastore = redis.Redis(connection_pool=connection_pool)
        return redi_datastore

    def create_index(self):
        raise NotImplementedError

    def index_documents(self):
        raise NotImplementedError

    def search_documents(self):
        raise NotImplementedError