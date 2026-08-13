# TradingAgents/graph/trading_graph.py

import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evidence.finmultitime import FrozenFinMultiTimeEvidenceStore
from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.openai_client import validate_deepseek_thinking
from tradingagents.reporting import write_report_tree
from tradingagents.runtime.run_context import (
    AuditTrail,
    RunContext,
    activate_run_context,
    audit_source,
    create_run_context,
    current_run_context,
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

_MEMORY_OUTCOME_PRICE_MODE = "adjusted_close"
_MEMORY_OUTCOME_HISTORY_OPTIONS = {
    # Freeze the yfinance daily adjusted-close contract instead of inheriting
    # provider defaults, which have changed across yfinance releases.
    "period": None,
    "interval": "1d",
    "prepost": False,
    "actions": False,
    "auto_adjust": True,
    "back_adjust": False,
    "repair": False,
    "keepna": False,
    "rounding": False,
}


def _completed_outcome_cutoff(context: RunContext, benchmark: str) -> pd.Timestamp:
    """Return the latest daily session label whose close is safely available.

    Date-only historical cutoffs mean the completed UTC day by RunContext
    contract. Timestamped US/``SPY`` cutoffs are checked against the actual
    XNYS session close, including DST and early closes. Other timestamped
    historical benchmarks conservatively exclude the current date. Live mode
    also excludes the current UTC date so an in-progress daily bar cannot count
    toward maturity.
    """
    cutoff = pd.Timestamp(context.as_of.date())
    if context.mode == "live":
        return cutoff - pd.Timedelta(days=1)
    if context.as_of.time() == datetime.max.time():
        return cutoff
    if benchmark == "SPY":
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XNYS")
        if calendar.is_session(cutoff):
            if pd.Timestamp(context.as_of) >= calendar.session_close(cutoff):
                return cutoff
            return calendar.previous_session(cutoff).tz_localize(None)
        return cutoff
    return cutoff - pd.Timedelta(days=1)


def _xnys_outcome_sessions(
    start_session: pd.Timestamp, holding_days: int,
) -> pd.DatetimeIndex:
    """Return the authoritative XNYS start plus subsequent holding sessions."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(start_session):
        raise ValueError(f"outcome start is not an XNYS session: {start_session.date()}")
    return calendar.sessions_window(start_session, holding_days + 1).tz_localize(None)


def _outcome_close_by_session(
    history: pd.DataFrame,
    *,
    start_session: pd.Timestamp,
    cutoff_session: pd.Timestamp | None,
) -> pd.Series:
    """Return valid adjusted closes indexed by timezone-naive session date.

    yfinance daily indexes represent exchange-local sessions and are commonly
    timezone-aware. Removing the timezone without converting it preserves that
    local session date and makes date-only historical cutoffs safe to compare.
    Rows outside the permitted window are removed before price validation so
    future provider output cannot affect the outcome.
    """
    if not isinstance(history, pd.DataFrame) or "Close" not in history.columns:
        raise ValueError("outcome history must contain a Close column")

    parsed_index = pd.to_datetime(history.index, errors="coerce")
    if not isinstance(parsed_index, pd.DatetimeIndex):
        parsed_index = pd.DatetimeIndex(parsed_index)
    valid_index = ~parsed_index.isna()
    normalized = history.iloc[valid_index][["Close"]].copy()
    normalized.index = parsed_index[valid_index].tz_localize(None).normalize()
    normalized = normalized.loc[normalized.index >= start_session]
    if cutoff_session is not None:
        normalized = normalized.loc[normalized.index <= cutoff_session]
    normalized = normalized.sort_index()

    if not normalized.index.is_unique:
        raise ValueError("outcome history has duplicate daily session dates")
    normalized["Close"] = pd.to_numeric(normalized["Close"], errors="coerce")
    close = normalized["Close"]
    if close.isna().any() or (close <= 0).any() or not close.map(math.isfinite).all():
        raise ValueError("outcome history contains an invalid adjusted Close")
    return close


def _coerce_max_retries(value):
    """Validate an ``llm_max_retries`` value to a non-negative int.

    Accepts an int or a numeric string (env vars arrive as strings). Rejects
    booleans and negatives loudly so a misconfiguration fails at startup rather
    than silently disabling retries.
    """
    if isinstance(value, bool):
        raise ValueError(f"llm_max_retries must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_max_retries must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"llm_max_retries must be >= 0, got {n}")
    return n


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # The Graph receives a complete resolved configuration. Replace the
        # dataflow process-global state so an explicit empty mapping (notably
        # tool_vendors={}) clears overrides left by an earlier Graph instance.
        set_config(self.config, replace=True)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # M1 is an explicit opt-in.  Validation happens before any graph is
        # constructed so a configured M1 run can never silently degrade to M0.
        self.finmultitime_evidence_store = (
            FrozenFinMultiTimeEvidenceStore.from_config(self.config)
            if self.config.get("finmultitime_evidence_enabled", False)
            else None
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)
        self._memory_namespace_mode = "live"
        self._run_context: RunContext | None = None

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            self.finmultitime_evidence_store,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Graph-shape-affecting run choices, kept for the checkpoint signature.
        self.selected_analysts = tuple(selected_analysts)

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        elif provider == "deepseek":
            thinking = self.config.get("deepseek_thinking")
            if thinking is not None and thinking != "":
                kwargs["deepseek_thinking"] = validate_deepseek_thinking(thinking)

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a TRADINGAGENTS_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        # SDK retry budget is cross-provider. Forward it only when explicitly set
        # so each provider keeps its own default (usually 2) otherwise (#1091).
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None and max_retries != "":
            kwargs["max_retries"] = _coerce_max_retries(max_retries)

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Deterministic verification snapshot (bound to the analyst
                    # LLM and required by its prompt; must be executable here or
                    # the call fails and the model reports it "unavailable").
                    get_verified_market_snapshot,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_macro_indicators,
                    get_prediction_markets,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _memory_holding_horizon_sessions(self) -> int:
        config = getattr(self, "config", {})
        value = (
            config.get("memory_holding_horizon_sessions", 5)
            if isinstance(config, dict) else 5
        )
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("memory_holding_horizon_sessions must be a positive integer")
        if value <= 0:
            raise ValueError("memory_holding_horizon_sessions must be a positive integer")
        return value

    def _memory_outcome_price_mode(self) -> str:
        config = getattr(self, "config", {})
        value = (
            config.get("memory_outcome_price_mode", _MEMORY_OUTCOME_PRICE_MODE)
            if isinstance(config, dict) else _MEMORY_OUTCOME_PRICE_MODE
        )
        if value != _MEMORY_OUTCOME_PRICE_MODE:
            raise ValueError(
                "memory_outcome_price_mode must be 'adjusted_close'; "
                f"got {value!r}"
            )
        return value

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int | None = None,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch adjusted-close raw and alpha returns over a complete horizon.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). XNYS defines the session labels for
        SPY-benchmarked outcomes; otherwise the asset's exchange-local daily
        labels define them. Both price series must contain every required label.
        Observation zero must be exactly ``trade_date`` and observation
        ``holding_days`` is the end price, so a five-session return requires six
        aligned observations. Raw return is the asset adjusted-close return;
        alpha is raw return minus the benchmark's adjusted-close return over
        those identical endpoints.

        Returns ``(raw_return, alpha_return, holding_days)`` or
        ``(None, None, None)`` if the complete horizon is unavailable.
        """
        from tradingagents.dataflows.symbol_utils import normalize_symbol

        if holding_days is None:
            # Call the implementation on the class so ``MagicMock(spec=Graph)``
            # compatibility tests cannot replace this helper with a mock value.
            holding_days = TradingAgentsGraph._memory_holding_horizon_sessions(self)
        TradingAgentsGraph._memory_outcome_price_mode(self)

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            start_session = pd.Timestamp(start.date())
            context = current_run_context()
            cutoff_session = _completed_outcome_cutoff(context, benchmark)
            if start_session > cutoff_session:
                return None, None, None
            # yfinance ``end`` is exclusive. Asking for the day after the last
            # completed session includes that label without crossing the cutoff.
            end_str = (cutoff_session + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            stock_history = yf.Ticker(normalize_symbol(ticker)).history(
                start=trade_date, end=end_str, **_MEMORY_OUTCOME_HISTORY_OPTIONS
            )
            benchmark_history = yf.Ticker(benchmark).history(
                start=trade_date, end=end_str, **_MEMORY_OUTCOME_HISTORY_OPTIONS
            )
            stock = _outcome_close_by_session(
                stock_history,
                start_session=start_session,
                cutoff_session=cutoff_session,
            ).rename("asset")
            bench = _outcome_close_by_session(
                benchmark_history,
                start_session=start_session,
                cutoff_session=cutoff_session,
            ).rename("benchmark")
            if benchmark == "SPY":
                required_sessions = _xnys_outcome_sessions(start_session, holding_days)
                asset_window = stock.reindex(required_sessions)
                benchmark_window = bench.reindex(required_sessions)
                complete_horizon = (
                    required_sessions[-1] <= cutoff_session
                    and not asset_window.isna().any()
                    and not benchmark_window.isna().any()
                )
            else:
                complete_horizon = (
                    start_session in stock.index and len(stock) >= holding_days + 1
                )
                asset_window = stock.iloc[:holding_days + 1]
                endpoint = asset_window.index[-1] if complete_horizon else None
                benchmark_window = (
                    bench.loc[:endpoint] if endpoint is not None else bench.iloc[:0]
                )
                complete_horizon = (
                    complete_horizon
                    and asset_window.index.equals(benchmark_window.index)
                )

            if not complete_horizon:
                audit_source(
                    source_name="decision_memory.outcome_prices",
                    capability="POINT_IN_TIME",
                    status="unavailable",
                    requested_start=trade_date,
                    requested_end=context.as_of,
                    reason=(
                        f"complete aligned horizon of {holding_days} trading sessions "
                        "is unavailable at the current cutoff or provider output"
                    ),
                )
                return None, None, None

            window = pd.concat([asset_window, benchmark_window], axis=1)
            raw = float(window["asset"].iloc[-1] / window["asset"].iloc[0] - 1)
            bench_ret = float(
                window["benchmark"].iloc[-1] / window["benchmark"].iloc[0] - 1
            )
            alpha = raw - bench_ret
            audit_source(
                source_name="decision_memory.outcome_prices",
                capability="POINT_IN_TIME",
                status="used",
                requested_start=trade_date,
                requested_end=context.as_of,
                latest_event_time=window.index[-1].date(),
                latest_available_time=window.index[-1].date(),
                reason=(
                    f"adjusted-close return used {holding_days} asset trading "
                    "sessions with matching benchmark observations"
                ),
            )
            return raw, alpha, holding_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        audit_source(
            source_name="decision_memory",
            capability="POINT_IN_TIME",
            status="used" if pending else "unavailable",
            requested_end=current_run_context().as_of,
            reason=(
                f"{len(pending)} pending entries inspected; historical entries are isolated "
                "and outcomes require a complete trading-day holding horizon."
            ),
        )
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        holding_days = self._memory_holding_horizon_sessions()
        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(
                ticker, entry["date"], holding_days=holding_days, benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
                # Reflections depend on post-decision prices. Persist the
                # first cutoff at which this evidence existed so replaying an
                # earlier decision in the same lineage cannot read it.
                "outcome_visible_from": current_run_context().as_of.isoformat(),
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.
        """
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity)

    def _run_signature(
        self,
        asset_type: str,
        ticker: str | None = None,
        trade_date: str | None = None,
    ) -> str:
        """Graph-shape inputs that must invalidate a checkpoint if changed.

        Keyed into the checkpoint thread ID so a resume under a different analyst
        selection, debate/risk depth, or asset mode starts fresh instead of
        silently continuing the previous graph (#1089).
        """
        context = current_run_context()
        return "|".join([
            "analysts=" + ",".join(self.selected_analysts),
            f"debate={self.config['max_debate_rounds']}",
            f"risk={self.config['max_risk_discuss_rounds']}",
            f"asset={asset_type}",
            f"mode={context.mode}",
            f"as_of={context.historical_as_of or ''}",
            "memory_lineage=" + str(
                context.memory_lineage_id
                or self.config.get("historical_memory_lineage_id")
                or ""
            ),
            "graph_config=" + str(self.config.get("graph_config_sha256") or ""),
            "finmultitime_enabled=" + str(
                bool(self.config.get("finmultitime_evidence_enabled", False))
            ),
            "finmultitime_bundle=" + str(
                getattr(
                    getattr(self, "finmultitime_evidence_store", None),
                    "bundle_identity",
                    "",
                )
            ),
            "finmultitime_packet=" + (
                self.finmultitime_evidence_store.case_identity(
                    ticker, str(trade_date)
                )["packet_json_sha256"]
                if ticker is not None
                and trade_date is not None
                and getattr(self, "finmultitime_evidence_store", None) is not None
                else ""
            ),
        ])

    def _historical_memory_config(self, ticker: str, context: RunContext) -> dict[str, Any]:
        """Return a safe disabled/date/experiment historical memory namespace."""
        config = dict(self.config)
        memory_mode = config.get("memory_mode", "isolated_date")
        if memory_mode not in {"disabled", "isolated_date", "experiment"}:
            raise ValueError(f"unsupported memory_mode: {memory_mode!r}")
        if memory_mode == "disabled":
            config["memory_log_path"] = None
            config["historical_as_of"] = context.as_of.isoformat()
            return config
        explicit = config.get("historical_memory_log_path")
        if memory_mode == "experiment" and explicit:
            raise ValueError(
                "experiment memory cannot use historical_memory_log_path; "
                "use historical_memory_dir so lineage, graph, and symbol isolation apply"
            )
        if explicit:
            path = Path(explicit).expanduser()
        else:
            root = Path(config.get("historical_memory_dir") or config["data_cache_dir"]) / "historical_memory"
            safe_symbol = safe_ticker_component(ticker)
            if memory_mode == "experiment":
                if not context.experiment_id:
                    raise ValueError("experiment memory requires RunContext.experiment_id")
                experiment = safe_ticker_component(context.experiment_id, max_len=128)
                graph_hash = config.get("graph_config_sha256")
                if not isinstance(graph_hash, str) or len(graph_hash) != 64:
                    raise ValueError(
                        "experiment memory requires a resolved graph_config_sha256"
                    )
                try:
                    int(graph_hash, 16)
                except ValueError as exc:
                    raise ValueError(
                        "experiment memory requires a resolved graph_config_sha256"
                    ) from exc
                lineage_value = (
                    context.memory_lineage_id
                    or config.get("historical_memory_lineage_id")
                )
                if not lineage_value:
                    raise ValueError(
                        "experiment memory requires a historical memory lineage"
                    )
                lineage = safe_ticker_component(str(lineage_value), max_len=128)
                path = (
                    root / experiment / graph_hash / lineage / f"{safe_symbol}.md"
                )
                config["historical_memory_lineage_id"] = lineage
            else:
                path = root / f"{safe_symbol}_{context.as_of.date().isoformat()}.md"
        config["memory_log_path"] = str(path)
        config["historical_as_of"] = context.as_of.isoformat()
        return config

    def _audit_path(self, ticker: str, context: RunContext) -> Path:
        configured = self.config.get("historical_audit_path")
        if configured:
            return Path(configured).expanduser()
        return (
            Path(self.config["results_dir"])
            / safe_ticker_component(ticker)
            / f"historical_data_audit_{context.as_of.date().isoformat()}.json"
        )

    def propagate(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        *,
        mode: str = "live",
        historical_as_of=None,
        run_context: RunContext | None = None,
    ):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name
        if run_context is None:
            # Supplying an as-of is itself an explicit historical request; the
            # mode keyword remains available for callers that prefer it.
            if historical_as_of is not None and mode == "live":
                mode = "historical"
            run_context = create_run_context(mode, as_of=historical_as_of)
        elif historical_as_of is not None:
            raise ValueError("pass either run_context or historical_as_of, not both")

        audit = AuditTrail(run_context)
        with activate_run_context(run_context, audit):
            self._run_context = run_context
            if run_context.mode == "historical":
                # Make the fail-closed policy visible even when an LLM does
                # not happen to request an optional source in this run.
                for source_name, capability, reason in (
                    ("stocktwits", "LIVE_ONLY", "StockTwits live stream disabled in historical mode."),
                    ("reddit", "LIVE_ONLY", "Reddit live search disabled in historical mode."),
                    ("polymarket", "LIVE_ONLY", "Polymarket current market snapshot disabled in historical mode."),
                    ("fred", "LIVE_ONLY", "FRED current revisions lack an as-of vintage."),
                    ("yfinance.ticker_info", "LIVE_ONLY", "Ticker.info is a current snapshot."),
                    ("yfinance.financial_statements", "LIVE_ONLY", "No filing/publication timestamp is available."),
                    ("yfinance.insider_transactions", "LIVE_ONLY", "No verifiable public filing timestamp is available."),
                ):
                    audit_source(
                        source_name=source_name,
                        capability=capability,
                        status="blocked",
                        requested_end=run_context.as_of,
                        reason=reason,
                    )
                self.memory_log = TradingMemoryLog(
                    self._historical_memory_config(company_name, run_context)
                )
                self._memory_namespace_mode = "historical"
            else:
                # A historical run must not leave its isolated log attached to
                # the graph object when the same instance is reused live.
                if getattr(self, "_memory_namespace_mode", "live") == "historical":
                    self.memory_log = TradingMemoryLog(self.config)
                self._memory_namespace_mode = "live"
            try:
                # Resolve pending entries before the pipeline runs. Historical
                # mode only sees its isolated namespace and mature outcomes.
                self._resolve_pending_entries(company_name)

                if self.config.get("checkpoint_enabled"):
                    self._checkpointer_ctx = get_checkpointer(
                        self.config["data_cache_dir"], company_name
                    )
                    saver = self._checkpointer_ctx.__enter__()
                    self.graph = self.workflow.compile(checkpointer=saver)

                    step = checkpoint_step(
                        self.config["data_cache_dir"], company_name, str(trade_date),
                        self._run_signature(asset_type, company_name, str(trade_date)),
                    )
                    if step is not None:
                        logger.info(
                            "Resuming from step %d for %s on %s", step, company_name, trade_date
                        )
                    else:
                        logger.info("Starting fresh for %s on %s", company_name, trade_date)

                try:
                    return self._run_graph(company_name, trade_date, asset_type=asset_type)
                finally:
                    if self._checkpointer_ctx is not None:
                        self._checkpointer_ctx.__exit__(None, None, None)
                        self._checkpointer_ctx = None
                        self.graph = self.workflow.compile()
            finally:
                self._run_context = None
                if run_context.mode == "historical":
                    audit.write(self._audit_path(company_name, run_context))

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def _run_graph(self, company_name, trade_date, asset_type: str = "stock"):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM and the
        # deterministically resolved instrument identity for all agents.
        past_context = self.memory_log.get_past_context(company_name)
        instrument_context = self.resolve_instrument_context(company_name, asset_type)
        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            instrument_context=instrument_context,
            run_context=current_run_context(),
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date+graph-shape resumes; a different
        # date or graph shape starts fresh (#1089).
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(
                company_name,
                str(trade_date),
                self._run_signature(asset_type, company_name, str(trade_date)),
            )
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            last_printed = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk["messages"]:
                    msg = chunk["messages"][-1]
                    # Nodes after the trader don't append to messages, so the
                    # same trailing message repeats across chunks. Print it only
                    # when it changes (#1027); the trace/state merge is unchanged.
                    signature = (type(msg).__name__, getattr(msg, "content", None))
                    if signature != last_printed:
                        msg.pretty_print()
                        last_printed = signature
                    trace.append(chunk)
            # Streamed chunks are per-node deltas. Merge them so the returned
            # state matches what graph.invoke() yields in the non-debug path.
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(asset_type, company_name, str(trade_date)),
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
