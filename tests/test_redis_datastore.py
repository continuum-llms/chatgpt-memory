import numpy as np
from chatgpt_memory.datastore.redis import RedisDataStore
from chatgpt_memory.environment import OPENAI_API_KEY
from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
from chatgpt_memory.llm_client.openai.embedding.embedding_client import (
    OpenAIEmbeddingClient,
)

SAMPLE_QUERIES = ["Where is Berlin?"]
SAMPLE_DOCUMENTS = [
    {"text": "Berlin is located in Germany.", "conversation_id": "1"},
    {"text": "Vienna is in Austria."},
]


def test_redis_datastore(redis_datastore: RedisDataStore):
    llm_client = LLMClient(api_key=OPENAI_API_KEY)
    embedding_config = EmbeddingConfig()
    openai_embedding_client = OpenAIEmbeddingClient(
        llm_client=llm_client, config=embedding_config
    )

    redis_datastore.connect()
    assert (
        redis_datastore.redis_connection.ping()
    ), "Redis connection failed,\
          double check your connection parameters"

    redis_datastore.create_index(number_of_vectors=2, index_fields=[])

    document_embeddings: np.ndarray = openai_embedding_client.embed_documents(
        SAMPLE_DOCUMENTS
    )
    for idx, embedding in enumerate(document_embeddings):
        SAMPLE_DOCUMENTS[idx]["embedding"] = embedding.astype(np.float32).tobytes()
    redis_datastore.index_documents(documents=SAMPLE_DOCUMENTS)

    query_embeddings: np.ndarray = openai_embedding_client.embed_queries(SAMPLE_QUERIES)
    query_vector = query_embeddings[0].astype(np.float32).tobytes()
    search_results = redis_datastore.search_documents(
        query_vector=query_vector, conversation_id="1", topk=1
    )
    assert len(search_results), "No documents returned, expected 1 document."

    assert (
        search_results[0].text == "Berlin is located in Germany."
    ), "Incorrect document returned as search result."
