from pydantic import BaseModel

class DataStoreConfig(BaseModel):
    host: str
    port: int
    username: str
    password: str