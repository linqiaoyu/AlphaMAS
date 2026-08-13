#!/usr/bin/env python3
"""Run the isolated M1 pilot routing probe without an Agent or an LLM.

The probe loads the archived pilot bundle through the same fail-closed store
used by runtime configuration, formats the three analyst-local projections,
and verifies that social has no FinMultiTime route.  It does not construct a
model client, call a provider, update Memory, or execute a trading run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.finmultitime.build_m1_pilot_inputs import (  # noqa: E402
    PILOT_DATASET_ID,
    PILOT_SESSIONS,
)
from tradingagents.evidence.finmultitime import (  # noqa: E402
    FrozenFinMultiTimeEvidenceStore,
)
from tradingagents.evidence.prompt import build_analyst_local_messages  # noqa: E402

ANALYST_KEYS = ("news", "fundamentals", "market", "social")
EXPECTED_MODALITY_STATUS = {
    "TEXT": "UNAVAILABLE",
    "TABLE": "AVAILABLE",
    "TIME_SERIES": "AVAILABLE",
    "IMAGE": "UNAVAILABLE",
}


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def probe(root: Path, config_path: Path) -> dict:
    config = _load_config(config_path)
    config["finmultitime_input_root"] = str(root)
    store = FrozenFinMultiTimeEvidenceStore.from_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "STATIC ANALYST SYSTEM MESSAGE"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    cases = []
    for session in PILOT_SESSIONS:
        symbol = "AAPL"
        packet = store.get_packet(symbol, session)
        assert {
            modality: packet[modality]["status"]
            for modality in EXPECTED_MODALITY_STATUS
        } == EXPECTED_MODALITY_STATUS
        state = {
            "company_of_interest": symbol,
            "trade_date": session,
            "messages": [("human", f"Analyze {symbol}")],
        }
        identity = store.case_identity(symbol, session)
        routes = {}
        for analyst_key in ANALYST_KEYS:
            routed = store.get_routed_evidence(symbol, session, analyst_key)
            local_messages = build_analyst_local_messages(state, store, analyst_key)
            formatted = prompt.format_messages(messages=local_messages)
            if routed is None:
                assert local_messages == state["messages"]
                routes[analyst_key] = {"route": "NO_FINMULTITIME_ROUTE"}
                continue
            assert formatted[2].content == f"Analyze {symbol}"
            routes[analyst_key] = {
                "packet_json_sha256": routed.packet_json_sha256,
                "route_sha256": routed.route_sha256,
                "prompt_message_count": len(formatted),
            }
        cases.append(
            {
                "case_id": identity["case_id"],
                "packet_json_sha256": identity["packet_json_sha256"],
                "bundle_scope": identity["bundle_scope"],
                "routes": routes,
            }
        )
    return {
        "probe": "M1 isolated out-of-window pilot runtime harness",
        "dataset_id": PILOT_DATASET_ID,
        "bundle_scope": store.bundle_scope,
        "input_bundle_identity": store.bundle_identity,
        "packet_manifest_sha256": store.expected_packet_manifest_sha256,
        "contract_version": store.expected_contract_version,
        "contract_sha256": store.expected_contract_sha256,
        "pilot_sessions": list(PILOT_SESSIONS),
        "packet_count": len(cases),
        "formal_m1_executed": False,
        "agent_runs": 0,
        "llm_calls": 0,
        "paid_api_calls": 0,
        "memory_updates": 0,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/m1_pilot_aapl_2023q4.json",
    )
    args = parser.parse_args()
    print(json.dumps(probe(args.input_root.resolve(), args.config.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
