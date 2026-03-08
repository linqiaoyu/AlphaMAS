"""Base Agent class for the Multi-Agent System."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .message import Message

if TYPE_CHECKING:
    from .environment import Environment


class Agent(ABC):
    """Abstract base class for all agents in the MAS.

    Sub-classes must implement :meth:`perceive`, :meth:`reason`, and
    :meth:`act`.  The :meth:`step` method calls these three in order,
    providing the standard *sense–think–act* agent loop.

    Args:
        agent_id: A unique string identifier for this agent.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id
        self._environment: Environment | None = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def register(self, environment: "Environment") -> None:
        """Attach this agent to an :class:`~mas.environment.Environment`."""
        self._environment = environment

    def send(self, receiver: str | None, content: Any) -> None:
        """Send a message via the environment.

        Args:
            receiver: Target agent ID, or ``None`` to broadcast.
            content: Message payload.

        Raises:
            RuntimeError: If the agent has not been registered with an
                environment.
        """
        if self._environment is None:
            raise RuntimeError(
                f"Agent '{self.agent_id}' is not registered with an environment."
            )
        msg = Message(sender=self.agent_id, receiver=receiver, content=content)
        self._environment.deliver(msg)

    def receive_messages(self) -> list[Message]:
        """Retrieve all pending messages from the environment's inbox.

        Returns:
            A list of :class:`~mas.message.Message` objects addressed to
            this agent (or broadcast messages).

        Raises:
            RuntimeError: If the agent has not been registered with an
                environment.
        """
        if self._environment is None:
            raise RuntimeError(
                f"Agent '{self.agent_id}' is not registered with an environment."
            )
        return self._environment.get_messages(self.agent_id)

    # ------------------------------------------------------------------
    # Sense–Think–Act interface
    # ------------------------------------------------------------------

    @abstractmethod
    def perceive(self) -> Any:
        """Observe the environment and/or inbox.

        Returns:
            Any percept data that will be passed to :meth:`reason`.
        """

    @abstractmethod
    def reason(self, percepts: Any) -> Any:
        """Process percepts and decide what to do.

        Args:
            percepts: The output of :meth:`perceive`.

        Returns:
            Any decision/action descriptor that will be passed to
            :meth:`act`.
        """

    @abstractmethod
    def act(self, decision: Any) -> None:
        """Execute the chosen action.

        Args:
            decision: The output of :meth:`reason`.
        """

    def step(self) -> None:
        """Run one full sense–think–act cycle."""
        percepts = self.perceive()
        decision = self.reason(percepts)
        self.act(decision)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id={self.agent_id!r})"
