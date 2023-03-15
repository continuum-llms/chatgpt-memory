import os

# Any remote API (OpenAI, Cohere etc.)
OPENAI_TIMEOUT = float(os.getenv("REMOTE_API_TIMEOUT_SEC", 30))
OPENAI_BACKOFF = float(os.getenv("REMOTE_API_BACKOFF_SEC", 10))
OPENAI_MAX_RETRIES = int(os.getenv("REMOTE_API_MAX_RETRIES", 5))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Cloud data store (Redis, Pinecone etc.)
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
