"""Read-only adapter for the eight frozen M2 E2E_PILOT evidence packets."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from tradingagents.evidence.finmultitime import (
    DEFAULT_FINMULTITIME_CONTRACT_SHA256,
    DEFAULT_FINMULTITIME_CONTRACT_VERSION,
    EXPECTED_ROUTE_SECTIONS,
    ROUTE_KEYS,
    ROUTE_PACKET_KEYS,
    FrozenEvidenceError,
    RoutedEvidence,
)
from tradingagents.runtime.run_context import audit_source

CORPUS_IDENTITY = "3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420"
CORPUS_MANIFEST_SHA256 = "85f5c6421dcdfcc92436cda6ce7d4f7b3d36d27d3c2e346fcef9066a11e660ea"
ARCHIVE_COMMIT = "6b2406f1e12e1988c27b44880a1e153a9b750c2e"
DECISION_SESSION = "2023-10-06"
PACKETS = {
    "AAPL": ("5cbafc5b3a59c672c084d5debdd15a72f3693a7cbf5e4d10032153dd9fcadb17", "4c4b94ed9324a887513971316feb7a7d9db12e4b1974780cf0583861853ea10f", "a2f12dc55b8770dd7fec0f971bb08d9fa898ac48637c96d9d1a8449898621545"),
    "AEMD": ("cf1a3fb958f0a38114050621c76b264da6327608f32fece1d0ae859ff6c1b6ac", "76004eb8be037bcfd19d8842e55ac6c56f8bc08b95d28b6b624fd8df5573f9da", "7f41ee5f6bc6f28683060342fef59a7d215c9682cd253212b74bd31adc212627"),
    "AGI": ("6829b5a7c479def7fd5c488f310c1efd36e4fdbd79c24cf1f8caa87e0fd4ad18", "867b1a85a7724cab5e45ac924c7278134e7d9d89af1bae731df2fad3d9fb74f5", "5d45e8d24175d5f3026e697fa446eea057387ca51a456dcb814074fcc7407904"),
    "AMZN": ("44eae5d63ef538255e4c70df2ffd4b509ccea14e49e4289eba51f2fe03f34abc", "54e6fd7342aebae90858c683f3d07abb8b807e191cd795fdf72931035555738e", "57edd43284e93d1c07010ad9d5e9b2dcfc88bdb4848df18fbe78078441393441"),
    "ARR": ("9be361fbd3f98dbd332f1a6b33f9d4406a33aaf103d046a1facfa924ee55f2bd", "96e9f5c6f07a445173b9f9f87b571fb22f32fc6a3d310ca8dcb9234818e1fafb", "99f8b19f213fbce80f79dd135534b8a2d699cdde865a4eff377c2cfd9b4cde7d"),
    "EML": ("3e53229aa08b6b63de0f3e40532672ae7f247945f7514d093e57273d523cd056", "3c33297a79ef72bbc68137f119132517576bfd12d964952a5955d014d717aa68", "c2bc226255f03aeaa24b4f7bdb4fad34c1d125936299854b94154307c2d0669c"),
    "JBSS": ("35a1ff6be9c64cd89493574cd8601824977bddd6955f10699483e7b9dfeca20f", "6ca4da49acdd160c9006a6136f835bf986b3a3ebced0d07fd2fa0fa79933411e", "bbaccd0a3dfdc6dd6eb1bae5a35a05f05a7574271abecf1f20eda7ac722b85b8"),
    "JPM": ("008cd8434ff604032e3cd94f67879880a86d37157de9ab14147e1e9d05559d34", "9a680275053640659a25fad2520357df9cf788dfb1704fd7aac31c4bb45c5245", "f744417095e7c05eaf471e812296f4c15bdb781aa9b8717ff5152f1eda871fe6"),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenM2E2EEvidenceStore:
    """Fail closed on exactly eight non-protected engineering-only packets."""

    bundle_scope = "M2_E2E_PILOT"
    bundle_identity = CORPUS_IDENTITY
    expected_contract_version = DEFAULT_FINMULTITIME_CONTRACT_VERSION
    expected_contract_sha256 = DEFAULT_FINMULTITIME_CONTRACT_SHA256
    expected_archive_commit = ARCHIVE_COMMIT

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifests/corpus_manifest.json"
        if not self.root.is_dir() or _sha(manifest_path) != CORPUS_MANIFEST_SHA256:
            raise FrozenEvidenceError("M2 E2E evidence corpus manifest identity mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("preformal_evidence_corpus_identity_sha256") != CORPUS_IDENTITY
            or manifest.get("packet_count") != 96
        ):
            raise FrozenEvidenceError("M2 E2E evidence corpus identity/count mismatch")
        binding = manifest.get("binding", {})
        packet_identities = binding.get("packet_sha256", {})
        self._packets: dict[str, dict[str, Any]] = {}
        for symbol, (json_sha, text_sha, packet_identity) in PACKETS.items():
            case_id = f"{symbol}:{DECISION_SESSION}"
            structured = self.root / f"packets/structured/e2e_pilot/{symbol}_{DECISION_SESSION}.json"
            rendered = self.root / f"packets/rendered/e2e_pilot/{symbol}_{DECISION_SESSION}.txt"
            if _sha(structured) != json_sha or _sha(rendered) != text_sha:
                raise FrozenEvidenceError(f"M2 E2E packet bytes changed: {case_id}")
            packet = json.loads(structured.read_text(encoding="utf-8"))
            if (
                packet_identities.get(case_id) != packet_identity
                or packet.get("packet_sha256") != packet_identity
                or packet.get("case_id") != case_id
                or packet.get("role") != "E2E_PILOT"
                or packet.get("engineering_only") is not True
                or packet.get("protected") is not False
                or packet.get("performance_for_selection") is not False
                or packet.get("contract_version") != self.expected_contract_version
                or packet.get("routing", {}).get("raw_packet_direct_injection") is not False
            ):
                raise FrozenEvidenceError(f"M2 E2E packet contract changed: {case_id}")
            self._packets[symbol] = packet

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FrozenM2E2EEvidenceStore:
        return cls(config.get("m2_preformal_evidence_root"))

    def _packet(self, symbol: str, session: str) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        if session != DECISION_SESSION or normalized not in self._packets:
            raise FrozenEvidenceError(f"no exact M2 E2E packet for {normalized}:{session}")
        return self._packets[normalized]

    def get_packet(self, symbol: str, decision_session: str) -> dict[str, Any]:
        return copy.deepcopy(self._packet(symbol, decision_session))

    def case_identity(self, symbol: str, decision_session: str) -> dict[str, Any]:
        packet = self._packet(symbol, decision_session)
        routes = {
            key: hashlib.sha256(
                packet["routed_projections"][ROUTE_PACKET_KEYS[key]]["text"].encode()
            ).hexdigest()
            for key in ROUTE_KEYS
        }
        file_sha, _, packet_identity = PACKETS[packet["symbol"]]
        return {
            "finmultitime_enabled": True,
            "bundle_scope": self.bundle_scope,
            "archive_commit": self.expected_archive_commit,
            "contract_version": self.expected_contract_version,
            "contract_sha256": self.expected_contract_sha256,
            "input_bundle_identity": self.bundle_identity,
            "case_id": packet["case_id"],
            "packet_json_sha256": file_sha,
            "packet_identity_sha256": packet_identity,
            "route_sha256": routes,
            "social": "NO_FINMULTITIME_ROUTE",
        }

    def get_routed_evidence(
        self, symbol: str, decision_session: str, analyst_key: str
    ) -> RoutedEvidence | None:
        if analyst_key == "social":
            return None
        if analyst_key not in ROUTE_KEYS:
            raise FrozenEvidenceError("unsupported M2 E2E analyst route")
        packet = self._packet(symbol, decision_session)
        projection = packet["routed_projections"][ROUTE_PACKET_KEYS[analyst_key]]
        if projection.get("sections") != EXPECTED_ROUTE_SECTIONS[analyst_key]:
            raise FrozenEvidenceError("M2 E2E route sections changed")
        text = projection.get("text")
        if not isinstance(text, str):
            raise FrozenEvidenceError("M2 E2E route text is malformed")
        route_sha = hashlib.sha256(text.encode()).hexdigest()
        file_sha, _, _ = PACKETS[packet["symbol"]]
        audit_source(
            source_name="finmultitime.frozen_m2_e2e_packet",
            capability="POINT_IN_TIME",
            status="used",
            requested_start=decision_session,
            requested_end=decision_session,
            reason="exact frozen M2 E2E routed projection delivered to analyst-local prompt",
            metadata=self.case_identity(symbol, decision_session) | {
                "analyst_key": analyst_key,
                "route_sha256": route_sha,
            },
        )
        return RoutedEvidence(
            text=text,
            analyst_key=analyst_key,
            case_id=packet["case_id"],
            symbol=packet["symbol"],
            decision_session=packet["decision_session"],
            packet_json_sha256=file_sha,
            route_sha256=route_sha,
            bundle_identity=self.bundle_identity,
            contract_version=self.expected_contract_version,
            contract_sha256=self.expected_contract_sha256,
            bundle_scope=self.bundle_scope,
            archive_commit=self.expected_archive_commit,
        )
