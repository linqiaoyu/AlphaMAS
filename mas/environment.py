"""Shared environment for the Multi-Agent System."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .message import Message

if TYPE_CHECKING:
    from .agent import Agent


class Environment:
    """The shared world that agents perceive and act within.

    The environment maintains:
    * A registry of agents by ID.
    * A message queue per agent (inbox).
    * A shared state dictionary that agents can read/write.

    Args:
        name: Optional human-readable name for this environment.
    """

    def __init__(self, name: str = "Environment") -> None:
        self.name: str = name
        self._agents: dict[str, "Agent"] = {}
        self._inboxes: dict[str, list[Message]] = defaultdict(list)
        self.state: dict = {}

    # ------------------------------------------------------------------
    # Agent management
    # ------------------------------------------------------------------

    def add_agent(self, agent: "Agent") -> None:
        """Register an agent with this environment.

        Args:
            agent: The agent to add.

        Raises:
            ValueError: If an agent with the same ID is already registered.
        """
        if agent.agent_id in self._agents:
            raise ValueError(
                f"An agent with id '{agent.agent_id}' is already registered."
            )
        self._agents[agent.agent_id] = agent
        agent.register(self)

    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent from this environment.

        Args:
            agent_id: ID of the agent to remove.

        Raises:
            KeyError: If no agent with ``agent_id`` is registered.
        """
        if agent_id not in self._agents:
            raise KeyError(f"No agent with id '{agent_id}' is registered.")
        del self._agents[agent_id]
        self._inboxes.pop(agent_id, None)

    @property
    def agents(self) -> list["Agent"]:
        """Return all registered agents."""
        return list(self._agents.values())

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def deliver(self, message: Message) -> None:
        """Place a message in the appropriate inbox(es).

        If ``message.receiver`` is ``None`` the message is broadcast to
        every agent *except* the sender.

        Args:
            message: The message to deliver.
        """
        if message.receiver is None:
            for agent_id in self._agents:
                if agent_id != message.sender:
                    self._inboxes[agent_id].append(message)
        else:
            self._inboxes[message.receiver].append(message)

    def get_messages(self, agent_id: str) -> list[Message]:
        """Return and clear the inbox for *agent_id*.

        Args:
            agent_id: The agent whose messages to retrieve.

        Returns:
            A list of pending messages (may be empty).
        """
        messages = self._inboxes.pop(agent_id, [])
        return messages

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Environment(name={self.name!r}, agents={list(self._agents.keys())})"
        )
