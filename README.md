# AlphaMAS — Alpha Multi-Agent System

A lightweight, pure-Python Multi-Agent System (MAS) framework built around
the classic **sense–think–act** agent loop.

## Features

* **`Agent`** — abstract base class with a `perceive → reason → act` lifecycle
* **`Environment`** — shared world with a message queue per agent and a key/value state dictionary
* **`Message`** — typed, timestamped messages for direct or broadcast communication
* **`Coordinator`** — drives the simulation loop, stepping every agent each round

## Quick start

No external dependencies are required — AlphaMAS uses only the Python
standard library.

```python
from mas import Agent, Coordinator, Environment


class HelloAgent(Agent):
    def perceive(self):
        return self.receive_messages()

    def reason(self, percepts):
        return "hello, world!"

    def act(self, decision):
        print(f"[{self.agent_id}] {decision}")
        self.send(None, decision)   # broadcast to all other agents


env = Environment(name="Demo")
env.add_agent(HelloAgent("agent_1"))
env.add_agent(HelloAgent("agent_2"))

coordinator = Coordinator(env)
coordinator.run(rounds=2)
```

## Project layout

```
mas/
  __init__.py      # public API
  agent.py         # Agent base class
  coordinator.py   # Coordinator
  environment.py   # Environment
  message.py       # Message dataclass
examples/
  ping_pong.py     # Ping-pong demo (PingAgent ↔ PongAgent + LoggerAgent)
tests/
  test_mas.py      # pytest test suite (24 tests)
```

## Running the example

```bash
python examples/ping_pong.py
```

## Running the tests

```bash
pip install pytest
pytest tests/
```
