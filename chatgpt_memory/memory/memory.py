from abc import ABC, abstractmethod

from chatgpt_memory.datastore import DataStore


class Memory(ABC):
    """
    Abstract class for memory module.
    """

    def __init__(self, datastore: DataStore):
        self.datastore = datastore

    @abstractmethod
    def add_user_message(self, message: str) -> None:
        """
        Add a user's message to the memory.

        :param message: The message to add.
        :type message: str
        """
        pass

    @abstractmethod
    def add_assistant_message(self, message: str):
        """
        Add a assistant's message to the memory.

        :param message: The message to add.
        :type message: str
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        Clear the memory.

        :return: True if the memory was cleared, False otherwise.
        :rtype: bool
        """
        pass
