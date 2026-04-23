from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "TradingAgentsGraph": (".trading_graph", "TradingAgentsGraph"),
    "ConditionalLogic": (".conditional_logic", "ConditionalLogic"),
    "GraphSetup": (".setup", "GraphSetup"),
    "Propagator": (".propagation", "Propagator"),
    "Reflector": (".reflection", "Reflector"),
    "SignalProcessor": (".signal_processing", "SignalProcessor"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)


if TYPE_CHECKING:
    from .trading_graph import TradingAgentsGraph
    from .conditional_logic import ConditionalLogic
    from .setup import GraphSetup
    from .propagation import Propagator
    from .reflection import Reflector
    from .signal_processing import SignalProcessor
