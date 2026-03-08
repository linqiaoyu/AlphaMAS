"""Tests for the AlphaMAS Multi-Agent System."""

import pytest
from mas import Agent, Coordinator, Environment, Message


# ---------------------------------------------------------------------------
# Concrete agent fixtures
# ---------------------------------------------------------------------------

class EchoAgent(Agent):
    """Agent that echoes every message it receives back to the sender."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.acted: list = []

    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        return [
            (msg.sender, f"echo:{msg.content}")
            for msg in percepts
            if msg.sender != self.agent_id
        ]

    def act(self, decision):
        for receiver, content in decision:
            self.acted.append(content)
            self.send(receiver, content)


class CounterAgent(Agent):
    """Agent that increments a shared counter in the environment state."""

    def perceive(self):
        return self._environment.state.get("counter", 0)

    def reason(self, percepts):
        return percepts + 1

    def act(self, decision):
        self._environment.state["counter"] = decision


class BroadcastAgent(Agent):
    """Sends a single broadcast every step."""

    def perceive(self):
        return None

    def reason(self, percepts):
        return "hello everyone"

    def act(self, decision):
        self.send(None, decision)


class ListenerAgent(Agent):
    """Collects all messages into a list."""

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.inbox_log: list[Message] = []

    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        return percepts

    def act(self, decision):
        self.inbox_log.extend(decision)


# ---------------------------------------------------------------------------
# Message tests
# ---------------------------------------------------------------------------

class TestMessage:
    def test_direct_message(self):
        msg = Message(sender="a", receiver="b", content="hi")
        assert msg.sender == "a"
        assert msg.receiver == "b"
        assert msg.content == "hi"
        assert msg.timestamp > 0

    def test_broadcast_message(self):
        msg = Message(sender="a", receiver=None, content="broadcast")
        assert msg.receiver is None
        assert "BROADCAST" in str(msg)

    def test_str_format(self):
        msg = Message(sender="alice", receiver="bob", content="hello")
        text = str(msg)
        assert "alice" in text
        assert "bob" in text
        assert "hello" in text


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_add_and_list_agents(self):
        env = Environment()
        agent = CounterAgent("c1")
        env.add_agent(agent)
        assert agent in env.agents
        assert len(env.agents) == 1

    def test_duplicate_agent_raises(self):
        env = Environment()
        env.add_agent(CounterAgent("dup"))
        with pytest.raises(ValueError, match="dup"):
            env.add_agent(CounterAgent("dup"))

    def test_remove_agent(self):
        env = Environment()
        env.add_agent(CounterAgent("rem"))
        env.remove_agent("rem")
        assert len(env.agents) == 0

    def test_remove_nonexistent_raises(self):
        env = Environment()
        with pytest.raises(KeyError):
            env.remove_agent("ghost")

    def test_direct_delivery(self):
        env = Environment()
        sender = CounterAgent("s")
        receiver = ListenerAgent("r")
        env.add_agent(sender)
        env.add_agent(receiver)

        msg = Message(sender="s", receiver="r", content="direct")
        env.deliver(msg)

        messages = env.get_messages("r")
        assert len(messages) == 1
        assert messages[0].content == "direct"

    def test_broadcast_delivery(self):
        env = Environment()
        env.add_agent(BroadcastAgent("broadcaster"))
        listener1 = ListenerAgent("l1")
        listener2 = ListenerAgent("l2")
        env.add_agent(listener1)
        env.add_agent(listener2)

        msg = Message(sender="broadcaster", receiver=None, content="bcast")
        env.deliver(msg)

        assert len(env.get_messages("l1")) == 1
        assert len(env.get_messages("l2")) == 1
        # Sender should NOT receive its own broadcast.
        assert len(env.get_messages("broadcaster")) == 0

    def test_get_messages_clears_inbox(self):
        env = Environment()
        env.add_agent(CounterAgent("a"))
        env.add_agent(ListenerAgent("b"))
        msg = Message(sender="a", receiver="b", content="once")
        env.deliver(msg)
        first = env.get_messages("b")
        second = env.get_messages("b")
        assert len(first) == 1
        assert len(second) == 0

    def test_shared_state(self):
        env = Environment()
        env.state["key"] = "value"
        assert env.state["key"] == "value"


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------

class TestAgent:
    def test_register_sets_environment(self):
        env = Environment()
        agent = CounterAgent("a")
        env.add_agent(agent)
        assert agent._environment is env

    def test_send_without_environment_raises(self):
        agent = CounterAgent("unregistered")
        with pytest.raises(RuntimeError, match="not registered"):
            agent.send(None, "oops")

    def test_receive_without_environment_raises(self):
        agent = CounterAgent("unregistered")
        with pytest.raises(RuntimeError, match="not registered"):
            agent.receive_messages()

    def test_echo_agent_step(self):
        env = Environment()
        sender = BroadcastAgent("broadcaster")
        echo = EchoAgent("echo")
        env.add_agent(sender)
        env.add_agent(echo)

        # Deliver a direct message to the echo agent.
        env.deliver(Message(sender="broadcaster", receiver="echo", content="test"))
        echo.step()
        assert echo.acted == ["echo:test"]

    def test_counter_agent_increments(self):
        env = Environment()
        counter = CounterAgent("c")
        env.add_agent(counter)
        env.state["counter"] = 0

        counter.step()
        assert env.state["counter"] == 1

        counter.step()
        assert env.state["counter"] == 2

    def test_repr(self):
        agent = CounterAgent("my_agent")
        assert "my_agent" in repr(agent)


# ---------------------------------------------------------------------------
# Coordinator tests
# ---------------------------------------------------------------------------

class TestCoordinator:
    def test_initial_round_is_zero(self):
        env = Environment()
        coord = Coordinator(env)
        assert coord.round == 0

    def test_run_increments_round(self):
        env = Environment()
        env.add_agent(CounterAgent("c"))
        coord = Coordinator(env)
        coord.run(rounds=3)
        assert coord.round == 3

    def test_run_steps_all_agents(self):
        env = Environment()
        c1 = CounterAgent("c1")
        c2 = CounterAgent("c2")
        env.add_agent(c1)
        env.add_agent(c2)
        env.state["counter"] = 0
        coord = Coordinator(env)
        # Each step increments the counter; both agents share state,
        # so 2 agents × 1 round = counter becomes 2.
        coord.run(rounds=1)
        assert env.state["counter"] == 2

    def test_invalid_rounds_raises(self):
        coord = Coordinator(Environment())
        with pytest.raises(ValueError):
            coord.run(rounds=0)

    def test_reset(self):
        env = Environment()
        env.add_agent(CounterAgent("c"))
        coord = Coordinator(env)
        coord.run(rounds=5)
        coord.reset()
        assert coord.round == 0

    def test_repr(self):
        env = Environment(name="TestEnv")
        coord = Coordinator(env)
        assert "TestEnv" in repr(coord)

    def test_broadcast_and_listen(self):
        """BroadcastAgent sends; ListenerAgents receive in the same round."""
        env = Environment()
        broadcaster = BroadcastAgent("b")
        l1 = ListenerAgent("l1")
        l2 = ListenerAgent("l2")
        env.add_agent(broadcaster)
        env.add_agent(l1)
        env.add_agent(l2)

        coord = Coordinator(env)
        coord.run(rounds=1)

        # Messages sent by broadcaster in round 1 are delivered to l1/l2
        # when they take their step in the same round.
        assert len(l1.inbox_log) == 1
        assert l1.inbox_log[0].content == "hello everyone"
        assert len(l2.inbox_log) == 1
