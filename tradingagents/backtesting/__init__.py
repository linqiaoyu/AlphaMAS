from importlib import import_module
from typing import TYPE_CHECKING

_EXPORTS = {
    "BacktestResult": (".backtester", "BacktestResult"),
    "SingleAssetBacktester": (".backtester", "SingleAssetBacktester"),
    "run_strategy_suite": (".backtester", "run_strategy_suite"),
    "SimulatedExchange": (".execution", "SimulatedExchange"),
    "format_execution_summary": (".execution", "format_execution_summary"),
    "BacktestMetrics": (".metrics", "BacktestMetrics"),
    "compute_backtest_metrics": (".metrics", "compute_backtest_metrics"),
    "BaseSignalStrategy": (".strategies", "BaseSignalStrategy"),
    "BuyAndHoldStrategy": (".strategies", "BuyAndHoldStrategy"),
    "KdjRsiStrategy": (".strategies", "KdjRsiStrategy"),
    "MacdStrategy": (".strategies", "MacdStrategy"),
    "SmaCrossStrategy": (".strategies", "SmaCrossStrategy"),
    "StrategyDecision": (".strategies", "StrategyDecision"),
    "TradingAgentsStrategy": (".strategies", "TradingAgentsStrategy"),
    "ZmrStrategy": (".strategies", "ZmrStrategy"),
    "build_rule_based_baselines": (".strategies", "build_rule_based_baselines"),
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
    from .backtester import BacktestResult, SingleAssetBacktester, run_strategy_suite
    from .execution import SimulatedExchange, format_execution_summary
    from .metrics import BacktestMetrics, compute_backtest_metrics
    from .strategies import (
        BaseSignalStrategy,
        BuyAndHoldStrategy,
        KdjRsiStrategy,
        MacdStrategy,
        SmaCrossStrategy,
        StrategyDecision,
        TradingAgentsStrategy,
        ZmrStrategy,
        build_rule_based_baselines,
    )
