class LLMClient:
    """
    Wrapper for the HTTP APIs for LLMs acting as data container for API configurations.

    :param api_key: Access key for the remote API.
    :param time_out: Time in seconds before the request times out.
    """

    def __init__(self, api_key: str, time_out: float = 30):
        self._api_key = api_key
        self._time_out = time_out

    @property
    def api_key(self):
        return self._api_key

    @property
    def time_out(self):
        return self._time_out
