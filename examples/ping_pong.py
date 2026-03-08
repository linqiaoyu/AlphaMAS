"""Simple ping-pong example using AlphaMAS.

Two agents (``PingAgent`` and ``PongAgent``) exchange messages across a
shared environment.  A third agent (``LoggerAgent``) listens for
broadcasts and prints every message it receives.

Run::

    python examples/ping_pong.py
"""

import sys
import os

# Allow running the example directly from the repo root without installing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mas import Agent, Coordinator, Environment  # noqa: E402


class PingAgent(Agent):
    """Sends a 'ping' message to PongAgent every step, then waits for pong."""

    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        for msg in percepts:
            print(f"  PingAgent received: {msg.content!r}")
        return "ping"

    def act(self, decision):
        print(f"  PingAgent sends: {decision!r}")
        self.send("pong_agent", decision)


class PongAgent(Agent):
    """Replies with 'pong' whenever it receives a 'ping'."""

    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        replies = []
        for msg in percepts:
            if msg.content == "ping":
                replies.append("pong")
        return replies

    def act(self, decision):
        for reply in decision:
            print(f"  PongAgent sends: {reply!r}")
            self.send("ping_agent", reply)
            # Also broadcast to let the logger hear it.
            self.send(None, f"[broadcast] {reply}")


class LoggerAgent(Agent):
    """Logs every broadcast message it receives."""

    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        return percepts

    def act(self, decision):
        for msg in decision:
            print(f"  LoggerAgent heard broadcast: {msg.content!r}")


def main() -> None:
    env = Environment(name="PingPongWorld")

    ping = PingAgent("ping_agent")
    pong = PongAgent("pong_agent")
    logger = LoggerAgent("logger_agent")

    env.add_agent(ping)
    env.add_agent(pong)
    env.add_agent(logger)

    coordinator = Coordinator(env)

    rounds = 3
    print(f"Running {rounds} rounds of ping-pong...\n")
    for r in range(1, rounds + 1):
        print(f"--- Round {r} ---")
        coordinator.run(rounds=1)

    print(f"\nSimulation complete after {coordinator.round} round(s).")


if __name__ == "__main__":
    main()
