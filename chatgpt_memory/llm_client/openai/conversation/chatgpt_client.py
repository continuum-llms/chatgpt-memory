import logging
import uuid

from langchain import LLMChain, OpenAI, PromptTemplate

from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.conversation.config import ChatGPTConfig
from chatgpt_memory.memory.manager import MemoryManager
from chatgpt_memory.utils.openai_utils import get_prompt

logger = logging.getLogger(__name__)


class ChatGPTClient(LLMClient):
    def __init__(self, config: ChatGPTConfig, memory_manager: MemoryManager):
        super().__init__(config=config)
        prompt = PromptTemplate(input_variables=["prompt"], template="{prompt}")
        self.chatgpt_chain = LLMChain(
            llm=OpenAI(
                temperature=config.temperature,
                openai_api_key=self.api_key,
                model_name=config.model_name,
                max_retries=config.max_retries,
                max_tokens=config.max_tokens,
            ),
            prompt=prompt,
            verbose=config.verbose,
        )
        self.memory_manager = memory_manager

    def converse(self, message: str, conversation_id: str = None) -> str:
        if not conversation_id:
            conversation_id = uuid.uuid4().hex

        history = ""
        try:
            past_messages = self.memory_manager.get_messages(conversation_id=conversation_id, query=message)
            history = "\n".join([past_message.text for past_message in past_messages if getattr(past_message, "text")])
        except Exception:
            logger.warning(f"No previous chat history found for conversation_id: {conversation_id}.")
        prompt = get_prompt(message=message, history=history)
        chat_gpt_answer = self.chatgpt_chain.predict(prompt=prompt)

        if len(message.strip()) and len(chat_gpt_answer.strip()):
            self.memory_manager.add_message(conversation_id=conversation_id, human=message, assistant=chat_gpt_answer)

        return chat_gpt_answer
