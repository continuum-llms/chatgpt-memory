"""
Contains a memory dataclass.
"""
from typing import List, Optional, Tuple

from pydantic import BaseModel


class Memory(BaseModel):
    """
    A memory dataclass.
    """

    conversation_id: str
    """ID of the conversation."""
    conversation_history: List[Tuple[str, str]]
    """Conversation history. List of tuples of (user, system)."""
    embedding: Optional[bytes] = None
    """Optional embedding of the conversation history."""
