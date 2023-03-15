from pydantic import BaseModel


class LLMClientConfig(BaseModel):
    api_key: str
    time_out: float = 30
