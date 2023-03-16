import uuid
from typing import List, Union

from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatGeneration, HumanMessage, SystemMessage

from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.conversation.config import ChatGPTConfig
from chatgpt_memory.memory.manager import MemoryManager


class ChatGPTClient(LLMClient):
    def __init__(self, config: ChatGPTConfig, memory_manager: MemoryManager):
        super().__init__(config=config)
        self.chat = ChatOpenAI(
            temperature=config.temperature,
            openai_api_key=self.api_key,
            model_name=config.model_name,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
        )
        self.memory_manager = memory_manager

    def converse(self, messages: List[Union[SystemMessage, HumanMessage]], conversation_id: str = None):
        if not conversation_id:
            conversation_id = uuid.uuid4().hex

        user_messages: List[HumanMessage] = []
        for message_pair in messages:
            for message in message_pair:
                if isinstance(message, HumanMessage):
                    user_messages.append(message)
        generated_responses: List[ChatGeneration] = [
            generation[0] for generation in self.chat.generate(messages).generations
        ]
        assert len(user_messages) == len(
            generated_responses
        ), "Not enough number of responses generated for given input messages"
        for user_message, generated_response in zip(user_messages, generated_responses):
            self.memory_manager.add_message(
                conversation_id=conversation_id, user=user_message, system=generated_response.text  # type: ignore
            )


if __name__ == "__main__":
    # TODO: remove this only for dev purposes
    from chatgpt_memory.datastore.config import RedisDataStoreConfig
    from chatgpt_memory.datastore.redis import RedisDataStore
    from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
    from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
    from chatgpt_memory.llm_client.openai.embedding.embedding_client import EmbeddingClient

    embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)
    embed_client = EmbeddingClient(config=embedding_config)

    redis_datastore_config = RedisDataStoreConfig(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
    )
    redis_datastore = RedisDataStore(config=redis_datastore_config, do_flush_data=True)
    redis_datastore.connect()
    redis_datastore.create_index()

    memory_manager = MemoryManager(datastore=redis_datastore, embed_client=embed_client)

    chat_gpt_client = ChatGPTClient(config=ChatGPTConfig(api_key=OPENAI_API_KEY), memory_manager=memory_manager)

    batch_messages = [
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="Translate this sentence from English to French. I love programming."),
        ],
        [
            SystemMessage(content="You are a helpful assistant that translates English to French."),
            HumanMessage(content="Translate this sentence from English to French. I love artificial intelligence."),
        ],
    ]

    chat_gpt_client.converse(messages=batch_messages)
