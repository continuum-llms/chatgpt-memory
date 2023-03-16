"""
Contains a memory dataclass.
"""
from pydantic import BaseModel


class Memory(BaseModel):
    """
    A memory dataclass.
    """

    conversation_id: str
    """ID of the conversation."""
