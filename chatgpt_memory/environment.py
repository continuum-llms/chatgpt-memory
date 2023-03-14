import os

# Any remote API (OpenAI, Cohere etc.)
OPENAI_TIMEOUT = float(os.environ.get("REMOTE_API_TIMEOUT_SEC", 30))
OPENAI_BACKOFF = float(os.environ.get("REMOTE_API_BACKOFF_SEC", 10))
OPENAI_MAX_RETRIES = int(os.environ.get("REMOTE_API_MAX_RETRIES", 5))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
