from chatgpt_memory.datastore.config import RedisDataStoreConfig
from chatgpt_memory.datastore.redis import RedisDataStore
import pytest

from chatgpt_memory.environment import OPENAI_API_KEY
from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
from chatgpt_memory.llm_client.openai.embedding.embedding_client import (
    OpenAIEmbeddingClient,
)
from chatgpt_memory.environment import REDIS_HOST, REDIS_PASSWORD, REDIS_PORT


@pytest.fixture(scope="session")
def openai_embedding_client():
    llm_client = LLMClient(api_key=OPENAI_API_KEY)
    embedding_config = EmbeddingConfig()
    return OpenAIEmbeddingClient(llm_client=llm_client, config=embedding_config)


@pytest.fixture(scope="session")
def redis_datastore():
    redis_datastore_config = RedisDataStoreConfig(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        vector_field_name="embedding",
        vector_dimensions=1024,
    )
    redis_datastore = RedisDataStore(config=redis_datastore_config, do_flush_data=True)

    return redis_datastore
 