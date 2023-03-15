from typing import Dict, List, Tuple

from chatgpt_memory.datastore.datastore import DataStore
from chatgpt_memory.llm_client.llm_client import LLMClient
from chatgpt_memory.llm_client.openai.embedding.embedding_client import EmbeddingClient

from .memory import Memory


class MemoryManager:
    """
    Manages the memory of conversations.

    Attributes:
        datastore (DataStore): Datastore to use for storing and retrieving memories.
        llm_client (LLMClient): LLM client to use for interacting with the LLM.
        embed_client (EmbeddingClient): Embedding client to call for embedding conversations.
        conversations (Dict[str, Memory]): Mapping of conversation IDs to memories to be managed.
    """

    def __init__(self, datastore: DataStore, llm_client: LLMClient, embed_client: EmbeddingClient) -> None:
        """
        Initializes the memory manager.

        Args:
            datastore (DataStore): Datastore to be used. Assumed to be connected.
            llm_client (LLMClient): LLM client to be used.
            embed_client (EmbeddingClient): Embedding client to be used.
        """
        self.datastore = datastore
        self.llm_client = llm_client
        self.embed_client = embed_client

        self.conversations: Dict[str, Memory] = {}

    def add_conversation(self, conversation: Memory) -> None:
        """
        Adds a conversation to the memory manager to be stored and manage.

        Args:
            conversation (Memory): Conversation to be added.
        """
        self.conversations[conversation.conversation_id] = conversation

    def remove_conversation(self, conversation_id: str) -> None:
        """
        Removes a conversation from the memory manager.

        Args:
            conversation_id (str): ID of the conversation to be removed.
        """
        del self.conversations[conversation_id]
        # TODO: remove from datastore?

    def clear(self) -> None:
        """
        Clears the memory manager.
        """
        for conversation_id in self.conversations:
            self.remove_conversation(conversation_id)

    def add_message(self, conversation_id: str, user: str, system: str) -> None:
        """
        Adds a message to a conversation.

        Args:
            conversation_id (str): ID of the conversation to add the message to.
            user (str): User message.
            system (str): System message.
        """
        self.conversations[conversation_id].conversation_history.append((user, system))

    def get_messages(self, conversation_id: str) -> List[Tuple[str, str]]:
        """
        Gets the messages of a conversation.

        Args:
            conversation_id (str): ID of the conversation to get the messages of.

        Returns:
            List[Tuple[str, str]]: List of messages of the conversation.
        """
        return self.conversations[conversation_id].conversation_history
