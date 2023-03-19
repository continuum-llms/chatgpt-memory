# ChatGPT Memory
Allows to scale the ChatGPT API to multiple simultaneous sessions with infinite contextual and adaptive memory powered by GPT and Redis datastore.

## Getting Started

1. Create your free `Redis` datastore [here](https://redis.com/try-free/).
2. Get your `OpenAI` API key [here](https://platform.openai.com/overview).

```python 
## set the following ENVIRONMENT Variables before running this script
from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from chatgpt_memory.datastore.config import RedisDataStoreConfig
from chatgpt_memory.datastore.redis import RedisDataStore
from chatgpt_memory.llm_client.openai.conversation.chatgpt_client import ChatGPTClient
from chatgpt_memory.llm_client.openai.conversation.config import ChatGPTConfig
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
from chatgpt_memory.llm_client.openai.embedding.embedding_client import EmbeddingClient
from chatgpt_memory.memory.manager import MemoryManager


embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)
embed_client = EmbeddingClient(config=embedding_config)
redis_datastore_config = RedisDataStoreConfig(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)
redis_datastore = RedisDataStore(config=redis_datastore_config)
redis_datastore.connect()
redis_datastore.create_index()
memory_manager = MemoryManager(datastore=redis_datastore, embed_client=embed_client, topk=1)

chat_gpt_client = ChatGPTClient(
    config=ChatGPTConfig(api_key=OPENAI_API_KEY, verbose=True), memory_manager=memory_manager
)

conversation_id = None
while True:
    user_message = input("\n Please enter your message: ")
    response = chat_gpt_client.converse(message=user_message, conversation_id=conversation_id)
    conversation_id = response.conversation_id
    print(response.chat_gpt_answer)


```
