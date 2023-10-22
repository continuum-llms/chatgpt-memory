*Development on this repository has discontinued. Please check out OpenAI's retrieval plugin instead: https://github.com/openai/chatgpt-retrieval-plugin*

# ChatGPT Memory

Allows to scale the ChatGPT API to multiple simultaneous sessions with infinite contextual and adaptive memory powered by GPT and Redis datastore. This can be visualized as follows

<p  align="center">
<br>
<img src="https://user-images.githubusercontent.com/6007894/227480704-e7e66341-98fd-43df-809a-f43d60d7c76b.png">
<br>
</p>

## Getting Started

1. Create your free `Redis` datastore [here](https://redis.com/try-free/).
2. Get your `OpenAI` API key [here](https://platform.openai.com/overview).
3. Install dependencies using `poetry`.

```bash
poetry install
```

### Use with UI
<img width="1217" alt="Screenshot 2023-04-17 at 10 26 59 PM" src="https://user-images.githubusercontent.com/6007894/232608443-054e47e6-6057-4583-9d92-205843a260c8.png">



Start the FastAPI webserver.
```bash
poetry run uvicorn rest_api:app --host 0.0.0.0 --port 8000
```

Run the UI.
```bash
poetry run streamlit run ui.py
```

### Use with Terminal

The library is highly modular. In the following, we describe the usage of each component (visualized above).

First, start out by setting the required environment variables before running your script. This is optional but recommended.
You can use a `.env` file for this. See the `.env.example` file for an example.

```python
from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
```

Create an instance of the `RedisDataStore` class with the `RedisDataStoreConfig` configuration.

```python
from chatgpt_memory.datastore import RedisDataStoreConfig, RedisDataStore


redis_datastore_config = RedisDataStoreConfig(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)
redis_datastore = RedisDataStore(config=redis_datastore_config)
```

Create an instance of the `EmbeddingClient` class with the `EmbeddingConfig` configuration.

```python
from chatgpt_memory.llm_client import EmbeddingConfig, EmbeddingClient

embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)
embed_client = EmbeddingClient(config=embedding_config)
```

Create an instance of the `MemoryManager` class with the Redis datastore and Embedding client instances, and the `topk` value.

```python
from chatgpt_memory.memory.manager import MemoryManager

memory_manager = MemoryManager(datastore=redis_datastore, embed_client=embed_client, topk=1)
```

Create an instance of the `ChatGPTClient` class with the `ChatGPTConfig` configuration and the `MemoryManager` instance.

```python
from chatgpt_memory.llm_client import ChatGPTClient, ChatGPTConfig

chat_gpt_client = ChatGPTClient(
    config=ChatGPTConfig(api_key=OPENAI_API_KEY, verbose=True), memory_manager=memory_manager
)
```

Start the conversation by providing user messages to the converse method of the `ChatGPTClient` instance.

```python
conversation_id = None
while True:
    user_message = input("\n Please enter your message: ")
    response = chat_gpt_client.converse(message=user_message, conversation_id=conversation_id)
    conversation_id = response.conversation_id
    print(response.chat_gpt_answer)
```

This will allow you to talk to the AI assistant and extend its memory by using an external Redis datastore.

### Putting it together

Here's all of the above put together. You can also find it under [`examples/simple_usage.py`](examples/simple_usage.py)

```python
## set the following ENVIRONMENT Variables before running this script
# Import necessary modules
from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from chatgpt_memory.datastore import RedisDataStoreConfig, RedisDataStore
from chatgpt_memory.llm_client import ChatGPTClient, ChatGPTConfig, EmbeddingConfig, EmbeddingClient
from chatgpt_memory.memory import MemoryManager

# Instantiate an EmbeddingConfig object with the OpenAI API key
embedding_config = EmbeddingConfig(api_key=OPENAI_API_KEY)

# Instantiate an EmbeddingClient object with the EmbeddingConfig object
embed_client = EmbeddingClient(config=embedding_config)

# Instantiate a RedisDataStoreConfig object with the Redis connection details
redis_datastore_config = RedisDataStoreConfig(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
)

# Instantiate a RedisDataStore object with the RedisDataStoreConfig object
redis_datastore = RedisDataStore(config=redis_datastore_config)

# Instantiate a MemoryManager object with the RedisDataStore object and EmbeddingClient object
memory_manager = MemoryManager(datastore=redis_datastore, embed_client=embed_client, topk=1)

# Instantiate a ChatGPTConfig object with the OpenAI API key and verbose set to True
chat_gpt_config = ChatGPTConfig(api_key=OPENAI_API_KEY, verbose=True)

# Instantiate a ChatGPTClient object with the ChatGPTConfig object and MemoryManager object
chat_gpt_client = ChatGPTClient(
    config=chat_gpt_config,
    memory_manager=memory_manager
)

# Initialize conversation_id to None
conversation_id = None

# Start the chatbot loop
while True:
    # Prompt the user for input
    user_message = input("\n Please enter your message: ")


    # Use the ChatGPTClient object to generate a response
    response = chat_gpt_client.converse(message=user_message, conversation_id=conversation_id)

    # Update the conversation_id with the conversation_id from the response
    conversation_id = response.conversation_id


    # Print the response generated by the chatbot
    print(response.chat_gpt_answer)
```

# Acknowledgments

UI has been added thanks to the awesome work by [avrabyt/MemoryBot](https://github.com/avrabyt/MemoryBot).
