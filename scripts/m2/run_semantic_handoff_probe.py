#!/usr/bin/env python3
"""Freeze and run the M2-07 semantic hand-off cost calibration.

The harness is deliberately outside ``tradingagents/``.  It reuses the live M1
analyst/research/Trader graph, compiles that unchanged workflow with an
``interrupt_after=["Trader"]`` observation boundary, and never invokes Risk or
Portfolio Manager nodes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import tempfile
import threading
import time
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.backtesting.llm_usage import extract_provider_usage
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.evidence.finmultitime import RoutedEvidence
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.runtime.run_context import AuditTrail, RunContext, activate_run_context

TASK_ID = "M2-07"
SCHEMA_VERSION = "M2-SEMANTIC-HANDOFF-v1"
CORPUS_IDENTITY = "3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420"
CORPUS_ARCHIVE_SHA = "6b2406f1e12e1988c27b44880a1e153a9b750c2e"
M1_CONTRACT_VERSION = "M1-FINMULTITIME-v1.0.2"
M1_CONTRACT_SHA256 = "46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08"
MODEL = "deepseek-v4-flash"
PROVIDER = "deepseek"
PROBE_POSITIONS = (0, 6, 12, 18, 24, 31)
TIER_COUNTS = {"COMPACT": 72, "STANDARD": 80, "MAXIMUM": 96}
TIER_IDENTITIES = {
    "COMPACT": "f35488ed0f910f73b11713bb7eadf00191f13f5e5b711b309dc879aca8075d62",
    "STANDARD": "f1253ab0ed8e23d9ae5656fc4250d7dca63bd6f9342e2fd9a2d84eac8e377452",
    "MAXIMUM": "68fbc780cbb454f78d0ae719f91665737ed6e04928eba91f4c9c0530745d601f",
}
RESERVE_CNY = Decimal("4.00")
TARGET_BUDGET_CNY = Decimal("40.00")
HARD_BUDGET_CNY = Decimal("50.00")
M2_07_HARD_CEILING_CNY = Decimal("5.00")
FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_debate_state",
        "risk_debate_state",
        "final_trade_decision",
        "reward",
        "future_return",
        "future_price",
        "outcome",
        "executed_action",
        "portfolio_outcome",
        "cash",
        "quantity",
        "weight",
        "equity",
        "prior_position",
    }
)
PERFORMANCE_KEYS = frozenset(
    {
        "reward",
        "return",
        "returns",
        "future_price",
        "future_return",
        "performance",
        "accuracy",
        "sharpe",
        "drawdown",
        "benchmark",
        "best_action",
        "label",
        "ground_truth",
    }
)
_ACTION_RE = re.compile(
    r"FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*\s*$", re.I
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def frozen_corpus_json(value: Any) -> bytes:
    """Match the byte canonicalisation frozen by the M2-06 corpus builder."""
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git(repository: Path, *args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=repository, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def assert_clean_at(repository: Path, expected_sha: str) -> None:
    if git(repository, "rev-parse", "HEAD") != expected_sha:
        raise RuntimeError(f"repository HEAD does not match frozen SHA: {repository}")
    if git(repository, "status", "--short"):
        raise RuntimeError(f"repository must be clean before paid probes: {repository}")


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_walk_keys(nested))
    return keys


def normalize_action(rendered_proposal: str) -> str:
    matches = _ACTION_RE.findall(rendered_proposal)
    actions = {match.upper() for match in matches}
    if len(actions) != 1:
        raise ValueError(f"ambiguous Prompt Trader action: {sorted(actions)}")
    return actions.pop()


def validate_actor_visible_state(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("semantic hand-off schema version mismatch")
    plan = value.get("research_manager", {}).get("investment_plan")
    proposal = value.get("prompt_trader", {}).get("rendered_proposal")
    action = value.get("prompt_trader", {}).get("normalized_action")
    if not isinstance(plan, str) or not plan:
        raise ValueError("Research Manager investment_plan is absent")
    if not isinstance(proposal, str) or not proposal:
        raise ValueError("Prompt Trader rendered_proposal is absent")
    if action not in {"BUY", "HOLD", "SELL"} or normalize_action(proposal) != action:
        raise ValueError("Prompt Trader action is missing or inconsistent")
    if value["research_manager"].get("investment_plan_sha256") != sha256_text(plan):
        raise ValueError("Research Manager plan SHA mismatch")
    if value["prompt_trader"].get("proposal_sha256") != sha256_text(proposal):
        raise ValueError("Prompt Trader proposal SHA mismatch")
    forbidden = _walk_keys(value) & FORBIDDEN_ACTOR_KEYS
    if forbidden:
        raise ValueError(f"forbidden Actor-visible fields: {sorted(forbidden)}")
    if value.get("reusable_for_M2_08") is not True:
        raise ValueError("probe state is not marked reusable for M2-08")


class M2FrozenEvidenceStore:
    """Read-only, fail-closed adapter exposing only frozen analyst projections."""

    bundle_scope = "M2_PREFORMAL"
    expected_archive_commit = CORPUS_ARCHIVE_SHA
    expected_contract_version = M1_CONTRACT_VERSION
    expected_contract_sha256 = M1_CONTRACT_SHA256

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        manifest = read_json(self.root / "manifests/corpus_manifest.json")
        if manifest.get("packet_count") != 96:
            raise ValueError("frozen M2-06 corpus does not contain 96 cases")
        if manifest.get("preformal_evidence_corpus_identity_sha256") != CORPUS_IDENTITY:
            raise ValueError("frozen M2-06 corpus identity differs")
        self.bundle_identity = CORPUS_IDENTITY
        self.tiers = read_json(self.root / "manifests/tier_membership.json")
        actual_tiers = self.tiers.get("tiers", {})
        for tier, count in TIER_COUNTS.items():
            if actual_tiers.get(tier, {}).get("case_count") != count:
                raise ValueError(f"{tier} count differs")
            if (
                actual_tiers[tier].get("canonical_case_list_sha256")
                != TIER_IDENTITIES[tier]
            ):
                raise ValueError(f"{tier} identity differs")
        self.membership = {
            row["case_id"]: row for row in self.tiers.get("cases", [])
        }
        if len(self.membership) != 96:
            raise ValueError("tier membership does not contain 96 unique cases")

    def _paths(self, symbol: str, decision_session: str) -> tuple[Path, Path]:
        case_id = f"{symbol.strip().upper()}:{decision_session}"
        membership = self.membership.get(case_id)
        if membership is None:
            raise ValueError(f"case is absent from frozen tier membership: {case_id}")
        filename = case_id.replace(":", "_")
        for role_dir in ("train", "validation", "final_holdout", "e2e_pilot"):
            structured = self.root / f"packets/structured/{role_dir}/{filename}.json"
            rendered = self.root / f"packets/rendered/{role_dir}/{filename}.txt"
            if structured.is_file() and rendered.is_file():
                return structured, rendered
        raise ValueError(f"frozen packet files missing: {case_id}")

    def get_packet(self, symbol: str, decision_session: str) -> dict[str, Any]:
        structured, rendered = self._paths(symbol, decision_session)
        packet = read_json(structured)
        rendered_bytes = rendered.read_bytes()
        if packet.get("packet_status") != "FINAL_FROZEN":
            raise ValueError("packet is not FINAL_FROZEN")
        if packet.get("case_id") != f"{symbol.strip().upper()}:{decision_session}":
            raise ValueError("packet identity mismatch")
        if packet.get("contract_version") != M1_CONTRACT_VERSION:
            raise ValueError("packet M1 contract differs")
        expected_packet_sha = packet.get("packet_sha256")
        unhashed_packet = {key: value for key, value in packet.items() if key != "packet_sha256"}
        if expected_packet_sha != sha256_bytes(frozen_corpus_json(unhashed_packet)):
            raise ValueError("structured Evidence Packet SHA mismatch")
        if packet.get("agent_text") != rendered_bytes.decode("utf-8"):
            raise ValueError("structured/rendered Evidence Packet mismatch")
        return copy.deepcopy(packet)

    def get_routed_evidence(
        self, symbol: str, decision_session: str, analyst_key: str
    ) -> RoutedEvidence | None:
        if analyst_key == "social":
            return None
        packet = self.get_packet(symbol, decision_session)
        packet_key = {
            "market": "market_analyst",
            "news": "news_analyst",
            "fundamentals": "fundamentals_analyst",
        }.get(analyst_key)
        if packet_key is None:
            raise ValueError(f"unsupported analyst route: {analyst_key}")
        route = packet["routed_projections"][packet_key]
        text = route["text"]
        return RoutedEvidence(
            text=text,
            analyst_key=analyst_key,
            case_id=packet["case_id"],
            symbol=packet["symbol"],
            decision_session=packet["decision_session"],
            packet_json_sha256=sha256_file(self._paths(symbol, decision_session)[0]),
            route_sha256=sha256_text(text),
            bundle_identity=self.bundle_identity,
            contract_version=M1_CONTRACT_VERSION,
            contract_sha256=M1_CONTRACT_SHA256,
            bundle_scope=self.bundle_scope,
            archive_commit=CORPUS_ARCHIVE_SHA,
        )


def select_probe_cases(corpus_root: Path) -> list[dict[str, Any]]:
    store = M2FrozenEvidenceStore(corpus_root)
    compact_train: list[dict[str, Any]] = []
    for case_id, membership in store.membership.items():
        symbol, session = case_id.split(":", 1)
        packet = store.get_packet(symbol, session)
        if packet.get("role") == "TRAIN" and membership.get("in_compact") is True:
            rendered_path = store._paths(symbol, session)[1]
            rendered = rendered_path.read_bytes()
            compact_train.append(
                {
                    "case_id": case_id,
                    "symbol": symbol,
                    "decision_session": session,
                    "role": "TRAIN",
                    "in_compact": True,
                    "in_standard": membership.get("in_standard") is True,
                    "in_maximum": membership.get("in_maximum") is True,
                    "packet_bytes": len(rendered),
                    "packet_sha256": packet["packet_sha256"],
                }
            )
    compact_train.sort(key=lambda row: (row["packet_bytes"], row["case_id"]))
    if len(compact_train) != 32:
        raise ValueError(f"COMPACT TRAIN count must be 32, got {len(compact_train)}")
    return [
        {**compact_train[position], "quantile_position": position}
        for position in PROBE_POSITIONS
    ]


def pricing_snapshot(retrieved_at: str) -> dict[str, Any]:
    return {
        "schema_version": "M2-DEEPSEEK-PRICING-v1",
        "model": MODEL,
        "pricing_source": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
        "source_authority": "official DeepSeek API documentation",
        "retrieval_timestamp": retrieved_at,
        "billing_currency": "CNY",
        "unit": "per_1m_tokens",
        "input_cache_hit_price": "0.02",
        "input_cache_miss_price": "1.00",
        "output_price": "2.00",
        "formula": "cache_hit_input/1e6*0.02 + cache_miss_input/1e6*1.00 + output/1e6*2.00",
        "conservative_fallback": "when cache split is unavailable, all input tokens use cache-miss price",
        "fx_conversion_required": False,
    }


def prepare_probe_archive(
    corpus_root: Path, output_root: Path, retrieved_at: str
) -> list[dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"probe archive already exists and is non-empty: {output_root}")
    probes = select_probe_cases(corpus_root)
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "probe_manifest.json",
        {
            "schema_version": "M2-SEMANTIC-COST-PROBE-MANIFEST-v1",
            "task_id": TASK_ID,
            "corpus_identity_sha256": CORPUS_IDENTITY,
            "compact_train_count": 32,
            "selection_sort": ["packet_bytes", "case_id"],
            "selection_positions_zero_based": list(PROBE_POSITIONS),
            "outcome_fields_used": False,
            "cases": probes,
        },
    )
    write_json(output_root / "pricing_snapshot.json", pricing_snapshot(retrieved_at))
    (output_root / "README.md").write_text(
        "# M2-07 semantic cost calibration\n\n"
        "This pre-Formal archive freezes six deterministic COMPACT/TRAIN cost probes. "
        "It contains no reward, future outcome, validation, FINAL_HOLDOUT, or E2E_PILOT "
        "probe output. The six successful semantic states are reusable by M2-08 and must "
        "not be regenerated.\n",
        encoding="utf-8",
    )
    return probes


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else {}
    return {}


def _raw_usage(response: Any) -> dict[str, Any]:
    output = _mapping(getattr(response, "llm_output", None))
    for key in ("token_usage", "usage"):
        usage = _mapping(output.get(key))
        if usage:
            return dict(usage)
    for generations in getattr(response, "generations", ()) or ():
        for generation in generations or ():
            message = getattr(generation, "message", None)
            metadata = _mapping(getattr(message, "response_metadata", None))
            for key in ("token_usage", "usage"):
                usage = _mapping(metadata.get(key))
                if usage:
                    return dict(usage)
    return {}


class SemanticUsageRecorder(BaseCallbackHandler):
    """Observational request recorder; it never mutates invocation inputs."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._ordinal = 0
        self._starts: dict[UUID | str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        self._case: dict[str, str] = {}

    @property
    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._rows)

    def start_case(self, case: Mapping[str, Any]) -> int:
        with self._lock:
            self._case = {
                key: str(case[key])
                for key in ("case_id", "symbol", "decision_session")
            }
            return len(self._rows)

    def finish_case(self, start: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = copy.deepcopy(self._rows[start:])
            self._case = {}
            return rows

    def _start(
        self,
        serialized: Mapping[str, Any],
        inputs: Any,
        run_id: UUID | str,
        metadata: Mapping[str, Any] | None,
        kwargs: Mapping[str, Any],
    ) -> None:
        metadata = _mapping(metadata)
        invocation = _mapping(kwargs.get("invocation_params"))
        serialized_kwargs = _mapping(serialized.get("kwargs"))
        with self._lock:
            self._ordinal += 1
            self._starts.setdefault(
                run_id,
                {
                    **self._case,
                    "request_ordinal": self._ordinal,
                    "agent_node": metadata.get("langgraph_node")
                    or metadata.get("agent_node"),
                    "model": metadata.get("ls_model_name")
                    or invocation.get("model")
                    or invocation.get("model_name")
                    or serialized_kwargs.get("model")
                    or MODEL,
                    "prompt_sha256": sha256_bytes(canonical_json(_jsonable(inputs))),
                    "started": time.monotonic(),
                },
            )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, messages, run_id, metadata, kwargs)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(serialized, prompts, run_id, metadata, kwargs)

    def _finish(
        self, run_id: UUID | str, *, response: Any = None, error: BaseException | None = None
    ) -> None:
        with self._lock:
            started = self._starts.pop(run_id, None)
            if started is None:
                return
            elapsed = time.monotonic() - started.pop("started")
            usage = (
                extract_provider_usage(response)
                if response is not None
                else {
                    "prompt_tokens": None,
                    "prompt_cache_hit_tokens": None,
                    "prompt_cache_miss_tokens": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                    "total_tokens": None,
                }
            )
            self._rows.append(
                {
                    **started,
                    "provider": PROVIDER,
                    "thinking": "disabled",
                    "temperature": 0,
                    **usage,
                    "provider_raw_usage": _raw_usage(response) if response else {},
                    "request_success": error is None,
                    "error_type": type(error).__name__ if error else None,
                    "latency_seconds": elapsed,
                }
            )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        del kwargs
        self._finish(run_id, response=response)

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        del kwargs
        self._finish(run_id, error=error)


def request_cost_cny(row: Mapping[str, Any], prices: Mapping[str, Any]) -> Decimal:
    prompt = int(row.get("prompt_tokens") or 0)
    output = int(row.get("completion_tokens") or 0)
    hit = row.get("prompt_cache_hit_tokens")
    miss = row.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        hit_count, miss_count, method = 0, prompt, "ALL_INPUT_AS_CACHE_MISS"
    else:
        hit_count, miss_count, method = int(hit), int(miss), "PROVIDER_CACHE_SPLIT"
        if hit_count + miss_count != prompt:
            raise ValueError("provider cache token split does not equal prompt tokens")
    cost = (
        Decimal(hit_count) * Decimal(str(prices["input_cache_hit_price"]))
        + Decimal(miss_count) * Decimal(str(prices["input_cache_miss_price"]))
        + Decimal(output) * Decimal(str(prices["output_price"]))
    ) / Decimal(1_000_000)
    if isinstance(row, dict):
        row["cost_method"] = method
        row["cost_cny"] = str(cost)
    return cost


def case_usage(rows: list[dict[str, Any]], prices: Mapping[str, Any]) -> dict[str, Any]:
    total_cost = sum((request_cost_cny(row, prices) for row in rows), Decimal(0))
    return {
        "schema_version": "M2-DEEPSEEK-USAGE-v1",
        "requests": rows,
        "total_calls": len(rows),
        "successful_calls": sum(row["request_success"] is True for row in rows),
        "failed_calls": sum(row["request_success"] is False for row in rows),
        "input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("completion_tokens") or 0) for row in rows),
        "cache_hit_tokens": sum(
            int(row.get("prompt_cache_hit_tokens") or 0) for row in rows
        ),
        "cache_miss_tokens": sum(
            int(row.get("prompt_cache_miss_tokens") or 0) for row in rows
        ),
        "case_cost_cny": str(total_cost),
    }


def build_probe_graph(store: M2FrozenEvidenceStore, recorder: SemanticUsageRecorder):
    runtime_root = Path(tempfile.mkdtemp(prefix="alphamas-m2-07-"))
    config = {
        **DEFAULT_CONFIG,
        "data_cache_dir": str(runtime_root / "cache"),
        "results_dir": str(runtime_root / "results"),
        "llm_provider": PROVIDER,
        "quick_think_llm": MODEL,
        "deep_think_llm": MODEL,
        "deepseek_thinking": "disabled",
        "temperature": 0.0,
        "research_depth": "medium",
        "max_debate_rounds": 3,
        "max_risk_discuss_rounds": 3,
        "selected_analysts": ["market", "social", "news", "fundamentals"],
        "output_language": "English",
        "checkpoint_enabled": False,
        "finmultitime_evidence_enabled": False,
    }
    graph = TradingAgentsGraph(
        selected_analysts=config["selected_analysts"],
        config=config,
        callbacks=[recorder],
    )
    graph.finmultitime_evidence_store = store
    graph.graph_setup = GraphSetup(
        graph.quick_thinking_llm,
        graph.deep_thinking_llm,
        graph.tool_nodes,
        graph.conditional_logic,
        store,
    )
    graph.workflow = graph.graph_setup.setup_graph(config["selected_analysts"])
    graph.graph = graph.workflow.compile(interrupt_after=["Trader"])
    return graph, runtime_root


def _actor_state(
    case: Mapping[str, Any],
    packet: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    runner_sha: str,
    probe_inputs_sha: str,
) -> dict[str, Any]:
    plan = state["investment_plan"]
    proposal = state["trader_investment_plan"]
    model_config = {
        "provider": PROVIDER,
        "model": MODEL,
        "quick_model": MODEL,
        "deep_model": MODEL,
        "thinking": "disabled",
        "temperature": 0,
        "research_depth": "Medium",
        "research_debate_rounds": 3,
        "analysts": ["market", "social", "news", "fundamentals"],
        "output_language": "English",
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case["case_id"],
        "symbol": case["symbol"],
        "decision_session": case["decision_session"],
        "role": "TRAIN",
        "evidence_packet_sha256": case["packet_sha256"],
        "research_manager": {
            "investment_plan": plan,
            "investment_plan_sha256": sha256_text(plan),
        },
        "prompt_trader": {
            "rendered_proposal": proposal,
            "normalized_action": normalize_action(proposal),
            "proposal_sha256": sha256_text(proposal),
        },
        "context": {
            "company_of_interest": state["company_of_interest"],
            "ticker": case["symbol"],
            "instrument_context": state["instrument_context"],
            "decision_session": case["decision_session"],
            "deterministic_temporal_context": {
                "run_mode": state["run_mode"],
                "historical_as_of": state["historical_as_of"],
                "frozen_decision_time": packet["decision_time"],
            },
        },
        "provenance": {
            "source_sha": runner_sha,
            "experiments_sha": probe_inputs_sha,
            "model_id": MODEL,
            "model_configuration_identity": sha256_bytes(canonical_json(model_config)),
            "run_identity": sha256_bytes(
                canonical_json(
                    {
                        "case_id": case["case_id"],
                        "runner_sha": runner_sha,
                        "probe_inputs_sha": probe_inputs_sha,
                    }
                )
            ),
        },
        "reusable_for_M2_08": True,
    }
    validate_actor_visible_state(value)
    return value


def _upstream_trace(
    case: Mapping[str, Any], state: Mapping[str, Any], audit: AuditTrail
) -> dict[str, Any]:
    return {
        "schema_version": "M2-SEMANTIC-UPSTREAM-TRACE-v1",
        "actor_visible": False,
        "case_id": case["case_id"],
        "analyst_outputs": {
            "market": state.get("market_report", ""),
            "social": state.get("sentiment_report", ""),
            "news": state.get("news_report", ""),
            "fundamentals": state.get("fundamentals_report", ""),
        },
        "research_debate": state.get("investment_debate_state", {}),
        "research_manager_provenance": {
            "investment_plan_sha256": sha256_text(state["investment_plan"])
        },
        "capture_boundary": {
            "stop_after": "Prompt Trader",
            "risk_debate_executed": False,
            "portfolio_manager_executed": False,
            "trade_executed": False,
        },
        "source_audit": audit.as_dict(),
    }


def write_case_artifacts(
    output_root: Path,
    case: Mapping[str, Any],
    actor: Mapping[str, Any],
    trace: Mapping[str, Any],
    usage: Mapping[str, Any],
) -> None:
    case_dir = output_root / "cases" / str(case["case_id"]).replace(":", "_")
    if case_dir.exists():
        raise RuntimeError(f"authoritative probe artifacts already exist: {case_dir}")
    case_dir.mkdir(parents=True)
    write_json(case_dir / "actor_visible_state.json", actor)
    write_json(case_dir / "upstream_trace.json", trace)
    write_json(case_dir / "usage.json", usage)
    files = {
        name: sha256_file(case_dir / name)
        for name in (
            "actor_visible_state.json",
            "upstream_trace.json",
            "usage.json",
        )
    }
    write_json(case_dir / "sha256.json", {"algorithm": "SHA-256", "files": files})


def run_probes(
    source_repo: Path,
    experiments_repo: Path,
    corpus_root: Path,
    output_root: Path,
    runner_sha: str,
    probe_inputs_sha: str,
) -> None:
    assert_clean_at(source_repo, runner_sha)
    assert_clean_at(experiments_repo, probe_inputs_sha)
    manifest = read_json(output_root / "probe_manifest.json")
    prices = read_json(output_root / "pricing_snapshot.json")
    if sha256_file(output_root / "probe_manifest.json") == sha256_file(
        output_root / "pricing_snapshot.json"
    ):
        raise RuntimeError("impossible manifest/pricing identity collision")
    expected = select_probe_cases(corpus_root)
    if manifest.get("cases") != expected:
        raise RuntimeError("pre-committed probe manifest differs from frozen selection")
    recorder = SemanticUsageRecorder()
    store = M2FrozenEvidenceStore(corpus_root)
    graph, runtime_root = build_probe_graph(store, recorder)
    actual_spend = Decimal(0)
    successful_costs: list[Decimal] = []
    try:
        for case in expected:
            case_dir = output_root / "cases" / case["case_id"].replace(":", "_")
            if case_dir.exists():
                actor = read_json(case_dir / "actor_visible_state.json")
                validate_actor_visible_state(actor)
                prior_usage = read_json(case_dir / "usage.json")
                cost = Decimal(str(prior_usage["case_cost_cny"]))
                actual_spend += cost
                successful_costs.append(cost)
                continue
            if successful_costs:
                guard = max(successful_costs)
                conservative_finish = actual_spend + guard * Decimal(
                    6 - len(successful_costs)
                )
                if conservative_finish > M2_07_HARD_CEILING_CNY:
                    raise RuntimeError("projected M2-07 probe spend exceeds CNY 5.00")
            packet = store.get_packet(case["symbol"], case["decision_session"])
            context = RunContext.historical(
                packet["decision_time"],
                experiment_id="M2_semantic_cost_calibration_v1",
                memory_lineage_id=f"M2-07:{case['case_id']}",
            )
            audit = AuditTrail(context)
            start = recorder.start_case(case)
            try:
                with activate_run_context(context, audit):
                    instrument_context = build_instrument_context(case["symbol"])
                    initial_state = graph.propagator.create_initial_state(
                        case["symbol"],
                        case["decision_session"],
                        asset_type="stock",
                        past_context="",
                        instrument_context=instrument_context,
                        run_context=context,
                    )
                    state = graph.graph.invoke(
                        initial_state, **graph.propagator.get_graph_args()
                    )
            except Exception:
                failed_rows = recorder.finish_case(start)
                failed_dir = output_root / "failed_attempts"
                failed_dir.mkdir(parents=True, exist_ok=True)
                write_json(
                    failed_dir
                    / f"{case['case_id'].replace(':', '_')}_{int(time.time())}.json",
                    case_usage(failed_rows, prices),
                )
                raise
            rows = recorder.finish_case(start)
            usage = case_usage(rows, prices)
            actor = _actor_state(
                case,
                packet,
                state,
                runner_sha=runner_sha,
                probe_inputs_sha=probe_inputs_sha,
            )
            trace = _upstream_trace(case, state, audit)
            write_case_artifacts(output_root, case, actor, trace, usage)
            cost = Decimal(str(usage["case_cost_cny"]))
            actual_spend += cost
            successful_costs.append(cost)
            if actual_spend > M2_07_HARD_CEILING_CNY:
                raise RuntimeError("actual M2-07 probe spend exceeded CNY 5.00")
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)
    if len(successful_costs) != 6:
        raise RuntimeError("six correctness-valid reusable probes are required")


def project_tiers(
    actual_probe_spend: Decimal,
    guard_cost: Decimal,
    sunk_failed_cost: Decimal = Decimal(0),
) -> list[dict[str, Any]]:
    if actual_probe_spend < 0 or guard_cost < 0 or sunk_failed_cost < 0:
        raise ValueError("cost inputs must be non-negative")
    projections = []
    for tier in ("COMPACT", "STANDARD", "MAXIMUM"):
        count = TIER_COUNTS[tier]
        # actual_probe_spend is defined to include any billed failed attempts;
        # sunk_failed_cost is reported separately and is not added twice.
        semantic_cost = actual_probe_spend + Decimal(count - 6) * guard_cost
        with_reserve = semantic_cost + RESERVE_CNY
        projections.append(
            {
                "tier": tier,
                "case_count": count,
                "probe_spend_cny": str(actual_probe_spend),
                "sunk_failed_cost_cny_included_in_probe_spend": str(sunk_failed_cost),
                "remaining_case_count": count - 6,
                "guard_cost_per_case_cny": str(guard_cost),
                "projected_semantic_generation_cost_cny": str(semantic_cost),
                "reserve_cny": str(RESERVE_CNY),
                "projected_with_reserve_cny": str(with_reserve),
                "within_target_40": with_reserve <= TARGET_BUDGET_CNY,
                "within_hard_50": with_reserve <= HARD_BUDGET_CNY,
            }
        )
    return projections


def select_tier(projections: list[Mapping[str, Any]]) -> tuple[str, str]:
    if any(_walk_keys(row) & PERFORMANCE_KEYS for row in projections):
        raise ValueError("performance fields are forbidden from tier selection")
    by_tier = {str(row["tier"]): row for row in projections}
    for tier in ("MAXIMUM", "STANDARD", "COMPACT"):
        if by_tier[tier]["within_target_40"] is True:
            return tier, "TARGET_BUDGET"
    if by_tier["COMPACT"]["within_hard_50"] is True:
        return "COMPACT", "HARD_BUDGET_FALLBACK"
    raise RuntimeError("COMPACT projection plus reserve exceeds CNY 50.00")


def finalize_archive(output_root: Path) -> dict[str, Any]:
    manifest_path = output_root / "probe_manifest.json"
    pricing_path = output_root / "pricing_snapshot.json"
    manifest = read_json(manifest_path)
    per_case = []
    all_rows: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        case_dir = output_root / "cases" / case["case_id"].replace(":", "_")
        actor = read_json(case_dir / "actor_visible_state.json")
        validate_actor_visible_state(actor)
        usage = read_json(case_dir / "usage.json")
        rows = usage["requests"]
        all_rows.extend(rows)
        per_case.append(
            {
                "case_id": case["case_id"],
                "total_calls": usage["total_calls"],
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "case_cost_cny": usage["case_cost_cny"],
                "actor_visible_state_sha256": sha256_file(
                    case_dir / "actor_visible_state.json"
                ),
                "correctness_valid": True,
                "reusable_for_M2_08": True,
            }
        )
    costs = [Decimal(row["case_cost_cny"]) for row in per_case]
    failed_attempt_rows: list[dict[str, Any]] = []
    for failed_path in sorted((output_root / "failed_attempts").glob("*.json")):
        failed_attempt_rows.extend(read_json(failed_path).get("requests", []))
    failed = [
        row
        for row in [*all_rows, *failed_attempt_rows]
        if row["request_success"] is False
    ]
    sunk_failed = sum(
        (Decimal(str(row.get("cost_cny", "0"))) for row in failed_attempt_rows),
        Decimal(0),
    )
    actual = sum(costs, Decimal(0)) + sunk_failed
    sorted_costs = sorted(costs)
    median = (sorted_costs[2] + sorted_costs[3]) / Decimal(2)
    summary = {
        "schema_version": "M2-SEMANTIC-COST-SUMMARY-v1",
        "task_id": TASK_ID,
        "model": MODEL,
        "pricing_snapshot_sha256": sha256_file(pricing_path),
        "valid_probe_count": 6,
        "failed_billed_attempt_count": len(failed),
        "actual_probe_spend_cny": str(actual),
        "per_case_costs": per_case,
        "minimum_case_cost_cny": str(min(costs)),
        "median_case_cost_cny": str(median),
        "mean_case_cost_cny": str(sum(costs, Decimal(0)) / Decimal(6)),
        "maximum_case_cost_cny": str(max(costs)),
        "guard_cost_per_case_cny": str(max(costs)),
        "sunk_failed_cost_cny": str(sunk_failed),
        "total_llm_calls": len(all_rows),
        "total_input_tokens": sum(int(row.get("prompt_tokens") or 0) for row in all_rows),
        "total_output_tokens": sum(
            int(row.get("completion_tokens") or 0) for row in all_rows
        ),
    }
    write_json(output_root / "cost_summary.json", summary)
    projections = project_tiers(actual, max(costs), sunk_failed)
    projection_doc = {
        "schema_version": "M2-SEMANTIC-TIER-PROJECTION-v1",
        "task_id": TASK_ID,
        "formula": "ACTUAL_PROBE_SPEND_CNY + (N - 6) * GUARD_COST_PER_CASE_CNY",
        "failed_cost_accounting": "actual_probe_spend includes billed failed cost; sunk_failed_cost is disclosed but not double-counted",
        "performance_used": False,
        "tiers": projections,
    }
    write_json(output_root / "tier_projection.json", projection_doc)
    tier, status = select_tier(projections)
    selection = {
        "schema_version": "M2-SEMANTIC-TIER-SELECTION-v1",
        "selected_tier": tier,
        "selection_reason": "COST_ONLY",
        "budget_status": status,
        "selected_case_count": TIER_COUNTS[tier],
        "tier_identity_sha256": TIER_IDENTITIES[tier],
        "pricing_snapshot_sha256": sha256_file(pricing_path),
        "probe_manifest_sha256": sha256_file(manifest_path),
        "cost_summary_sha256": sha256_file(output_root / "cost_summary.json"),
        "performance_used": False,
    }
    write_json(output_root / "tier_selection.json", selection)
    write_archive_sha256(output_root)
    return selection


def write_archive_sha256(output_root: Path) -> None:
    files = {
        path.relative_to(output_root).as_posix(): sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "sha256.json"
    }
    write_json(output_root / "sha256.json", {"algorithm": "SHA-256", "files": files})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--corpus-root", type=Path, required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--pricing-retrieved-at", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-repo", type=Path, required=True)
    run.add_argument("--experiments-repo", type=Path, required=True)
    run.add_argument("--corpus-root", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--runner-sha", required=True)
    run.add_argument("--probe-inputs-sha", required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        probes = prepare_probe_archive(
            args.corpus_root, args.output_root, args.pricing_retrieved_at
        )
        print(json.dumps({"status": "PASS", "cases": probes}, indent=2))
    elif args.command == "run":
        run_probes(
            args.source_repo,
            args.experiments_repo,
            args.corpus_root,
            args.output_root,
            args.runner_sha,
            args.probe_inputs_sha,
        )
        selection = finalize_archive(args.output_root)
        print(json.dumps({"status": "PASS", **selection}, indent=2))
    else:
        selection = finalize_archive(args.output_root)
        print(json.dumps({"status": "PASS", **selection}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
