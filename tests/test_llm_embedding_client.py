from chatgpt_memory.llm_client.openai.embedding.embedding_client import EmbeddingClient

SAMPLE_QUERIES = ["Where is Berlin?"]
SAMPLE_DOCUMENTS = [{"text": "Berlin is located in Germany."}]

EXPECTED_EMBEDDING_DIMENSIONS = (1, 1024)


def test_openai_embedding_client(openai_embedding_client: EmbeddingClient):
    assert (
        openai_embedding_client.embed_queries(SAMPLE_QUERIES).shape == EXPECTED_EMBEDDING_DIMENSIONS
    ), "Generated query embedding is of inconsistent dimension"

    assert (
        openai_embedding_client.embed_documents(SAMPLE_DOCUMENTS).shape == EXPECTED_EMBEDDING_DIMENSIONS
    ), "Generated document(s) embedding is of inconsistent dimension"
