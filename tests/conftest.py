import pytest

from chatgpt_memory.environment import OPENAI_API_KEY
from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
from chatgpt_memory.llm_client.openai.embedding.embedding_client import OpenAIEmbeddingClient


@pytest.fixture(scope="session")
def openai_embedding_client():
    llm_client = LLMClient(api_key=OPENAI_API_KEY)
    embedding_config = EmbeddingConfig()
    return OpenAIEmbeddingClient(llm_client=llm_client, config=embedding_config)
