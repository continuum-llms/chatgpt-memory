#!/bin/bash
"""
This script describes a simple usage of the library.
You can see a breakdown of the individual steps in the README.md file.
"""
## set the following ENVIRONMENT Variables before running this script
# Import necessary modules
from chatgpt_memory.environment import OPENAI_API_KEY, REDIS_HOST, REDIS_PASSWORD, REDIS_PORT
from chatgpt_memory.datastore.config import RedisDataStoreConfig
from chatgpt_memory.datastore.redis import RedisDataStore
from chatgpt_memory.llm_client.openai.conversation.chatgpt_client import ChatGPTClient
from chatgpt_memory.llm_client.openai.conversation.config import ChatGPTConfig
from chatgpt_memory.llm_client.openai.embedding.config import EmbeddingConfig
from chatgpt_memory.llm_client.openai.embedding.embedding_client import EmbeddingClient
from chatgpt_memory.memory.manager import MemoryManager

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

# Connect to the Redis database
redis_datastore.connect()

# Create an index in the Redis database
redis_datastore.create_index()

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