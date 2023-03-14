from chatgpt_memory.llm_client.openai.embedding.embedding_client import (
    OpenAIEmbeddingClient,
)

SAMPLE_QUERIES = ["Where is Berlin?"]
SAMPLE_DOCUMENTS = [{"text": "Berlin is located in Germany."}]


def test_openai_embedding_client(openai_embedding_client: OpenAIEmbeddingClient):
    assert openai_embedding_client.embed_queries(SAMPLE_QUERIES).shape[0]

    assert openai_embedding_client.embed_documents(SAMPLE_DOCUMENTS).shape[0]
