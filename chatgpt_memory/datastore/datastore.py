from abc import ABC, abstractmethod
from typing import Any, Dict, List

from chatgpt_memory.datastore.config import DataStoreConfig


class DataStore(ABC):
    """
    Abstract class for datastores.
    """

    def __init__(self, config: DataStoreConfig):
        self.config = config

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def create_index(self):
        raise NotImplementedError

    @abstractmethod
    def index_documents(self, documents: List[Dict]):
        raise NotImplementedError

    @abstractmethod
    def search_documents(self, query_vector: Any, conversation_id: str, topk: int) -> List[Any]:
        raise NotImplementedError
