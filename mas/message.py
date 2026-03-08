"""Message class for inter-agent communication."""

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class Message:
    """Represents a message sent between agents.

    Attributes:
        sender: The agent ID of the sender.
        receiver: The agent ID of the intended recipient, or ``None`` for
            broadcast messages.
        content: The payload of the message (any serialisable value).
        timestamp: Unix timestamp of when the message was created.
    """

    sender: str
    receiver: str | None
    content: Any
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        receiver_label = self.receiver if self.receiver else "BROADCAST"
        return (
            f"[{self.sender} -> {receiver_label}]: {self.content}"
        )
