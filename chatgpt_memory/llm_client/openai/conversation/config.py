from chatgpt_memory.llm_client.config import LLMClientConfig


class ChatGPTConfig(LLMClientConfig):
    temperature: float = 0
    model_name: str = "gpt-3.5-turbo"
    max_retries: int = 6
    max_tokens: int = 256
    verbose: bool = False
