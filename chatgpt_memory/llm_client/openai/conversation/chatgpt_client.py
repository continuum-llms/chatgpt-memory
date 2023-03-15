from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.conversation.config import ChatGPTConfig


class ChatGPTClient(LLMClient):
    def __init__(self, config: ChatGPTConfig):
        super().__init__(config=config)

    def converse(self):
        pass
