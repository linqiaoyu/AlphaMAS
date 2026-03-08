"""Coordinator that orchestrates the multi-agent system run loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .environment import Environment


class Coordinator:
    """Orchestrates agents in an :class:`~mas.environment.Environment`.

    The coordinator drives the simulation loop, stepping every registered
    agent in registration order for the requested number of rounds.

    Args:
        environment: The environment whose agents will be stepped.
    """

    def __init__(self, environment: "Environment") -> None:
        self.environment: "Environment" = environment
        self._round: int = 0

    @property
    def round(self) -> int:
        """The current simulation round (starts at 0 before the first run)."""
        return self._round

    def run(self, rounds: int = 1) -> None:
        """Step all agents for *rounds* rounds.

        Each round calls :meth:`~mas.agent.Agent.step` on every registered
        agent in registration order.

        Args:
            rounds: Number of rounds to simulate.

        Raises:
            ValueError: If *rounds* is not a positive integer.
        """
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds!r}.")

        for _ in range(rounds):
            self._round += 1
            for agent in self.environment.agents:
                agent.step()

    def reset(self) -> None:
        """Reset the round counter to 0."""
        self._round = 0

    def __repr__(self) -> str:
        return (
            f"Coordinator(environment={self.environment.name!r},"
            f" round={self._round})"
        )
