from abc import ABC

from chatgpt_memory.llm_client.config import LLMClientConfig


class LLMClient(ABC):
    """
    Wrapper for the HTTP APIs for LLMs acting as data container for API configurations.
    """

    def __init__(self, config: LLMClientConfig):
        self._api_key = config.api_key
        self._time_out = config.time_out

    @property
    def api_key(self):
        return self._api_key

    @property
    def time_out(self):
        return self._time_out
