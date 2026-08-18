"""Fixed, PIT-safe ARMA(1,1) benchmark strategy.

The model input is derived only from raw close prices and corporate actions in
the strategy-visible snapshot.  No vendor back-adjustment factor is used.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from tradingagents.backtesting.models import Action, DecisionStatus, StrategyDecision
from tradingagents.backtesting.portfolio import Portfolio

ARMA11_BENCHMARK_ID = "ARMA11_FIXED_V1"
ARMA11_ORDER = (1, 0, 1)
ARMA11_TREND = "c"
ARMA11_RETURN_COUNT = 252
ARMA11_PRICE_COUNT = ARMA11_RETURN_COUNT + 1
ARMA11_FORECAST_STEPS = 5
ARMA11_FIT_METHOD = "statespace"
ARMA11_MAXITER = 200
ARMA11_RETURN_CONVENTION = "daily_simple_pit_total_return"
ARMA11_FAILURE_STATUS = "MODEL_FAILURE_HOLD_FALLBACK"


@dataclass(frozen=True)
class ReturnWindow:
    returns: pd.Series
    price_start_session: str
    input_start_session: str
    input_end_session: str
    sha256: str


@dataclass(frozen=True)
class FitForecast:
    params: dict[str, float]
    forecasts: tuple[float, ...]
    cumulative_forecast: float
    converged: bool | None
    warnings: tuple[str, ...]


def _canonical_return_sha256(values: pd.Series) -> str:
    lines = ["session,return"]
    lines.extend(
        f"{pd.Timestamp(session).date().isoformat()},{float(value):.17g}"
        for session, value in values.items()
    )
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def pit_safe_total_returns(
    market_history: pd.DataFrame,
    *,
    decision_session: str,
    return_count: int = ARMA11_RETURN_COUNT,
) -> ReturnWindow:
    """Build an exact trailing window from session-local raw prices/actions.

    For session ``t``, the gross total-return factor is
    ``(Close_t * split_factor_t + Dividend_t) / Close_(t-1)`` where a recorded
    zero split means factor one.  Every term is observable by ``t`` close, so a
    later corporate action can never alter an earlier return.
    """
    if isinstance(return_count, bool) or not isinstance(return_count, int) or return_count <= 0:
        raise ValueError("return_count must be a positive integer")
    required = {"Close", "Dividends", "Stock Splits"}
    missing = required - set(market_history.columns)
    if missing:
        raise ValueError(f"market history lacks PIT return fields: {sorted(missing)}")
    history = market_history.copy().sort_index()
    history.index = pd.to_datetime(history.index, errors="raise").tz_localize(None).normalize()
    if not history.index.is_unique:
        raise ValueError("market history has duplicate sessions")
    cutoff = pd.Timestamp(decision_session).normalize()
    if not history.empty and history.index.max() > cutoff:
        raise ValueError("market history contains a session after the decision close")
    price_count = return_count + 1
    if len(history) < price_count:
        raise ValueError(
            f"fewer than {return_count} valid PIT-safe daily returns: "
            f"need {price_count} prices, observed {len(history)}"
        )
    window = history.iloc[-price_count:].copy()
    numeric = window.loc[:, ["Close", "Dividends", "Stock Splits"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("PIT return window contains non-finite price/action data")
    if (numeric["Close"] <= 0).any() or (numeric[["Dividends", "Stock Splits"]] < 0).any().any():
        raise ValueError("PIT return window contains invalid price/action data")
    split = numeric["Stock Splits"].where(numeric["Stock Splits"] != 0.0, 1.0)
    gross = (numeric["Close"] * split + numeric["Dividends"]) / numeric["Close"].shift(1)
    returns = (gross - 1.0).iloc[1:].astype(float)
    if len(returns) != return_count or not np.isfinite(returns.to_numpy()).all():
        raise ValueError(f"failed to construct exactly {return_count} finite PIT returns")
    if (returns <= -1.0).any():
        raise ValueError("PIT total return must be greater than -1")
    return ReturnWindow(
        returns=returns,
        price_start_session=window.index[0].date().isoformat(),
        input_start_session=returns.index[0].date().isoformat(),
        input_end_session=returns.index[-1].date().isoformat(),
        sha256=_canonical_return_sha256(returns),
    )


def fit_forecast_arma11(returns: pd.Series) -> FitForecast:
    """Fit the frozen statespace specification and forecast exactly five steps."""
    values = np.asarray(returns, dtype=float)
    if values.shape != (ARMA11_RETURN_COUNT,) or not np.isfinite(values).all():
        raise ValueError(f"ARMA input must contain exactly {ARMA11_RETURN_COUNT} finite returns")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model = ARIMA(
            values,
            order=ARMA11_ORDER,
            trend=ARMA11_TREND,
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        fitted = model.fit(
            method=ARMA11_FIT_METHOD,
            method_kwargs={"maxiter": ARMA11_MAXITER, "disp": 0},
        )
        forecast = np.asarray(fitted.forecast(steps=ARMA11_FORECAST_STEPS), dtype=float)
    params_array = np.asarray(fitted.params, dtype=float)
    if not np.isfinite(params_array).all():
        raise ValueError("ARMA fit returned non-finite parameters")
    if forecast.shape != (ARMA11_FORECAST_STEPS,) or not np.isfinite(forecast).all():
        raise ValueError("ARMA fit returned a malformed or non-finite five-step forecast")
    cumulative = float(np.prod(1.0 + forecast) - 1.0)
    if not np.isfinite(cumulative):
        raise ValueError("ARMA compounded forecast is non-finite")
    mle_retvals = getattr(fitted, "mle_retvals", None)
    converged_value = mle_retvals.get("converged") if isinstance(mle_retvals, dict) else None
    converged = bool(converged_value) if converged_value is not None else None
    warning_messages = tuple(
        f"{item.category.__name__}: {item.message}" for item in captured
    )
    return FitForecast(
        params={
            str(name): float(value)
            for name, value in zip(fitted.param_names, params_array, strict=True)
        },
        forecasts=tuple(float(value) for value in forecast),
        cumulative_forecast=cumulative,
        converged=converged,
        warnings=warning_messages,
    )


def action_for_target(current_quantity: float, target_weight: float) -> Action:
    current_target = 1.0 if current_quantity > Portfolio.tolerance else 0.0
    if current_target == target_weight:
        return Action.HOLD
    return Action.BUY if target_weight == 1.0 else Action.SELL


class ARMA11Strategy:
    """The isolated fixed ARMA(1,1) weekly target-weight benchmark."""

    strategy_id = "arma11_fixed_v1"
    fail_fast_on_decision_failure = False

    def decide(
        self,
        *,
        symbol: str,
        decision_session: str,
        decision_time: Any,
        market_history: pd.DataFrame,
        portfolio_snapshot: Any,
        context: dict[str, Any],
    ) -> StrategyDecision:
        current_target = 1.0 if portfolio_snapshot.quantity > Portfolio.tolerance else 0.0
        base_metadata: dict[str, Any] = {
            "benchmark_identity": ARMA11_BENCHMARK_ID,
            "return_convention": ARMA11_RETURN_CONVENTION,
            "model_order": list(ARMA11_ORDER),
            "trend": ARMA11_TREND,
            "fit_method": ARMA11_FIT_METHOD,
            "maxiter": ARMA11_MAXITER,
            "forecast_steps": ARMA11_FORECAST_STEPS,
            "statsmodels_version": importlib.metadata.version("statsmodels"),
            "current_portfolio_state": portfolio_snapshot.to_dict(),
            "next_execution_session": context.get("execution_session"),
            "model_failure": False,
        }
        window: ReturnWindow | None = None
        try:
            window = pit_safe_total_returns(
                market_history, decision_session=decision_session
            )
            fit = fit_forecast_arma11(window.returns)
            target = 1.0 if fit.cumulative_forecast > 0.0 else 0.0
            action = action_for_target(portfolio_snapshot.quantity, target)
            metadata = {
                **base_metadata,
                "fit_status": "success",
                "fit_success": True,
                "converged": fit.converged,
                "fit_warnings": list(fit.warnings),
                "price_start_session": window.price_start_session,
                "input_start_session": window.input_start_session,
                "input_end_session": window.input_end_session,
                "input_observation_count": len(window.returns),
                "input_price_observation_count": ARMA11_PRICE_COUNT,
                "return_series_sha256": window.sha256,
                "fitted_parameters": fit.params,
                "forecast_daily_returns": list(fit.forecasts),
                "forecast_5d_cumulative": fit.cumulative_forecast,
                "signal": "LONG" if target == 1.0 else "CASH",
                "target_weight": target,
                "action_label": action.value,
            }
            return StrategyDecision(
                symbol=symbol,
                decision_session=decision_session,
                decision_time_utc=decision_time,
                action=action,
                target_weight=target,
                status=DecisionStatus.SUCCESS,
                reason="fixed ARMA(1,1) five-session cumulative forecast",
                strategy_id=self.strategy_id,
                experiment_id=context["experiment_id"],
                raw_signal=f"target={target:.1f}",
                metadata=metadata,
            )
        except Exception as exc:
            window_metadata = (
                {
                    "price_start_session": window.price_start_session,
                    "input_start_session": window.input_start_session,
                    "input_end_session": window.input_end_session,
                    "input_observation_count": len(window.returns),
                    "input_price_observation_count": ARMA11_PRICE_COUNT,
                    "return_series_sha256": window.sha256,
                }
                if window is not None else {}
            )
            metadata = {
                **base_metadata,
                **window_metadata,
                "fit_status": ARMA11_FAILURE_STATUS,
                "fit_success": False,
                "converged": None,
                "fit_warnings": [],
                "model_failure": True,
                "failure_type": type(exc).__name__,
                "failure_message": str(exc),
                "signal": "PRESERVE",
                "target_weight": current_target,
                "action_label": Action.HOLD.value,
            }
            return StrategyDecision(
                symbol=symbol,
                decision_session=decision_session,
                decision_time_utc=decision_time,
                action=Action.HOLD,
                target_weight=current_target,
                status=DecisionStatus.SUCCESS,
                reason=ARMA11_FAILURE_STATUS,
                strategy_id=self.strategy_id,
                experiment_id=context["experiment_id"],
                raw_signal=ARMA11_FAILURE_STATUS,
                metadata=metadata,
            )
