from enum import Enum

from chatgpt_memory.llm_client.config import LLMClientConfig


class EmbeddingModels(Enum):
    ada = "*-ada-*-001"
    babbage = "*-babbage-*-001"
    curie = "*-curie-*-001"
    davinci = "*-davinci-*-001"


class EmbeddingConfig(LLMClientConfig):
    url: str = "https://api.openai.com/v1/embeddings"
    batch_size: int = 64
    progress_bar: bool = False
    model: str = EmbeddingModels.ada.value
    max_seq_len: int = 8191
    use_tiktoken: bool = False
