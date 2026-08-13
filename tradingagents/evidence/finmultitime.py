"""Fail-closed access to the frozen M1 FinMultiTime Evidence Packets.

This module deliberately contains no preprocessing, selection, rendering,
network, or model code.  It verifies the byte-addressed frozen input bundle,
looks up an exact ``(symbol, decision_session)`` packet, and exposes only the
packet's already-frozen analyst projection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tradingagents.runtime.run_context import audit_source

DEFAULT_FINMULTITIME_CONTRACT_VERSION = "M1-FINMULTITIME-v1.0.2"
DEFAULT_FINMULTITIME_CONTRACT_SHA256 = (
    "46f6a05f12a7c402936178748c55dab099c8754d99fa1a0c41faf525cd37ae08"
)
DEFAULT_FINMULTITIME_PACKET_MANIFEST_SHA256 = (
    "05f10129b430475bad3d5dee9dfceffba463f8a99cdd3f087ff4b755a869d63d"
)
DEFAULT_FINMULTITIME_INPUT_BUNDLE_IDENTITY = (
    "30596a54788101873f1c88bdf653df7f12ac3b4861a7058b6a36df0861274121"
)
DEFAULT_FINMULTITIME_ARCHIVE_COMMIT = (
    "3750fa50224ba46ab1d4bf5511cb5e8fa514445b"
)
DEFAULT_FINMULTITIME_BUNDLE_SCOPE = "FORMAL"
PILOT_FINMULTITIME_DATASET_ID = "finmultitime_m1_pilot_aapl_2023q4_4w_v1"
PILOT_FINMULTITIME_SESSIONS = (
    "2023-10-06",
    "2023-10-13",
    "2023-10-20",
    "2023-10-27",
)
# These are populated once the separately archived pilot bundle is committed.
# PILOT mode fails closed while they are unset; it never derives an identity
# from the directory it was pointed at.
DEFAULT_PILOT_PACKET_MANIFEST_SHA256: str | None = None
DEFAULT_PILOT_INPUT_BUNDLE_IDENTITY: str | None = None
EVIDENCE_BUNDLE_SCOPES = frozenset({"FORMAL", "PILOT"})

EXPECTED_PACKET_COUNT = 78
EXPECTED_SYMBOLS = frozenset({"AAPL", "AMZN", "JPM"})
ROUTE_KEYS = ("news", "fundamentals", "market")
ROUTE_PACKET_KEYS = {
    "news": "news_analyst",
    "fundamentals": "fundamentals_analyst",
    "market": "market_analyst",
}
EXPECTED_ROUTE_SECTIONS = {
    "news": ["TEXT"],
    "fundamentals": ["TABLE"],
    "market": ["TIME_SERIES", "IMAGE"],
}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class FrozenEvidenceError(ValueError):
    """Raised whenever frozen M1 evidence cannot be proven intact."""


@dataclass(frozen=True)
class RoutedEvidence:
    """One exact frozen analyst projection and its provenance identities."""

    text: str
    analyst_key: str
    case_id: str
    symbol: str
    decision_session: str
    packet_json_sha256: str
    route_sha256: str
    bundle_identity: str
    contract_version: str
    contract_sha256: str
    bundle_scope: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrozenEvidenceError(f"cannot read frozen JSON: {path}") from exc
    if not isinstance(value, dict):
        raise FrozenEvidenceError(f"frozen JSON must contain an object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrozenEvidenceError(message)


def _safe_relative_path(root: Path, relative: str, *, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} must be a path string")
    candidate = Path(relative)
    _require(not candidate.is_absolute(), f"{label} must be relative: {relative!r}")
    resolved = (root / candidate).resolve()
    _require(
        resolved == root or root in resolved.parents,
        f"{label} escapes the frozen input root: {relative!r}",
    )
    return resolved


def _bundle_identity(manifest_hashes: dict[str, str]) -> str:
    payload = json.dumps(
        manifest_hashes,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrozenFinMultiTimeEvidenceStore:
    """Read-only, fail-closed adapter for the frozen M1 input archive."""

    def __init__(
        self,
        input_root: str | Path,
        *,
        bundle_scope: str = DEFAULT_FINMULTITIME_BUNDLE_SCOPE,
        expected_contract_version: str = DEFAULT_FINMULTITIME_CONTRACT_VERSION,
        expected_contract_sha256: str = DEFAULT_FINMULTITIME_CONTRACT_SHA256,
        expected_packet_manifest_sha256: str | None = None,
        expected_input_bundle_identity: str | None = None,
        expected_archive_commit: str = DEFAULT_FINMULTITIME_ARCHIVE_COMMIT,
        verify_full_bundle_on_start: bool = True,
    ) -> None:
        if input_root is None or str(input_root).strip() == "":
            raise FrozenEvidenceError(
                "FinMultiTime evidence is enabled but finmultitime_input_root is missing"
            )
        normalized_scope = str(bundle_scope).strip().upper()
        _require(
            normalized_scope in EVIDENCE_BUNDLE_SCOPES,
            f"unsupported FinMultiTime bundle scope: {bundle_scope!r}",
        )
        if normalized_scope == "FORMAL":
            expected_packet_manifest_sha256 = (
                expected_packet_manifest_sha256 or DEFAULT_FINMULTITIME_PACKET_MANIFEST_SHA256
            )
            expected_input_bundle_identity = (
                expected_input_bundle_identity or DEFAULT_FINMULTITIME_INPUT_BUNDLE_IDENTITY
            )
        else:
            if not expected_packet_manifest_sha256 or not expected_input_bundle_identity:
                raise FrozenEvidenceError(
                    "PILOT scope requires explicitly configured packet-manifest and "
                    "input-bundle identities"
                )
        self.bundle_scope = normalized_scope
        self.root = Path(input_root).expanduser().resolve()
        _require(self.root.is_dir(), f"frozen FinMultiTime input root is missing: {self.root}")
        self.expected_contract_version = expected_contract_version
        self.expected_contract_sha256 = expected_contract_sha256
        self.expected_packet_manifest_sha256 = expected_packet_manifest_sha256
        self.expected_input_bundle_identity = expected_input_bundle_identity
        self.expected_archive_commit = expected_archive_commit
        self.verify_full_bundle_on_start = bool(verify_full_bundle_on_start)

        self.manifest = _read_json(self.root / "manifest.json")
        self.bundle_manifest = _read_json(
            _safe_relative_path(
                self.root,
                self.manifest.get("input_bundle_manifest", {}).get("path", ""),
                label="input bundle manifest path",
            )
        )
        self.checksum_manifest = _read_json(
            self.root / "manifests/input_bundle_checksums.json"
        )
        self.packet_manifest = _read_json(
            self.root / "manifests/evidence_packet_manifest.json"
        )
        self.bundle_identity = self._validate_bundle()
        self._packet_entries = self._validate_packet_manifest()
        self._verify_packet_files()
        if self.verify_full_bundle_on_start:
            self._verify_checksum_inventory()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FrozenFinMultiTimeEvidenceStore:
        """Construct the store from explicit runtime configuration."""
        scope = str(
            config.get("finmultitime_bundle_scope", DEFAULT_FINMULTITIME_BUNDLE_SCOPE)
        ).strip().upper()
        packet_manifest_default = (
            DEFAULT_FINMULTITIME_PACKET_MANIFEST_SHA256
            if scope == "FORMAL" else DEFAULT_PILOT_PACKET_MANIFEST_SHA256
        )
        bundle_identity_default = (
            DEFAULT_FINMULTITIME_INPUT_BUNDLE_IDENTITY
            if scope == "FORMAL" else DEFAULT_PILOT_INPUT_BUNDLE_IDENTITY
        )
        return cls(
            config.get("finmultitime_input_root"),
            bundle_scope=scope,
            expected_contract_version=config.get(
                "finmultitime_expected_contract_version",
                DEFAULT_FINMULTITIME_CONTRACT_VERSION,
            ),
            expected_contract_sha256=config.get(
                "finmultitime_expected_contract_sha256",
                DEFAULT_FINMULTITIME_CONTRACT_SHA256,
            ),
            expected_packet_manifest_sha256=config.get(
                "finmultitime_expected_packet_manifest_sha256",
                packet_manifest_default,
            ),
            expected_input_bundle_identity=config.get(
                "finmultitime_expected_input_bundle_identity",
                bundle_identity_default,
            ),
            expected_archive_commit=config.get(
                "finmultitime_archive_commit",
                DEFAULT_FINMULTITIME_ARCHIVE_COMMIT,
            ),
            verify_full_bundle_on_start=config.get(
                "finmultitime_verify_full_bundle_on_start", True
            ),
        )

    def _validate_bundle(self) -> str:
        manifest = self.manifest
        bundle_manifest = self.bundle_manifest
        packet_manifest_path = self.root / "manifests/evidence_packet_manifest.json"
        packet_manifest_sha = _sha256_file(packet_manifest_path)
        bundle_manifest_path = _safe_relative_path(
            self.root,
            manifest.get("input_bundle_manifest", {}).get("path", ""),
            label="input bundle manifest path",
        )
        bundle_manifest_sha = _sha256_file(bundle_manifest_path)
        checksum_manifest_path = self.root / "manifests/input_bundle_checksums.json"
        checksum_manifest_sha = _sha256_file(checksum_manifest_path)
        processed_manifest_path = self.root / "manifests/processed_sha256.json"
        processed_manifest_sha = _sha256_file(processed_manifest_path)

        if self.bundle_scope == "FORMAL":
            _require(
                manifest.get("bundle_scope") in (None, "FORMAL"),
                "bundle scope mismatch: FORMAL runtime cannot load a PILOT bundle",
            )
            _require(
                bundle_manifest.get("bundle_scope") in (None, "FORMAL"),
                "input bundle scope mismatch: FORMAL runtime cannot load a PILOT bundle",
            )
        else:
            _require(
                manifest.get("bundle_scope") == "PILOT"
                and manifest.get("pilot_only") is True
                and manifest.get("formal_eligible") is False,
                "bundle scope mismatch: PILOT runtime requires an explicit pilot bundle",
            )
            _require(
                bundle_manifest.get("bundle_scope") == "PILOT"
                and bundle_manifest.get("pilot_only") is True
                and bundle_manifest.get("formal_eligible") is False,
                "input bundle scope mismatch: PILOT metadata is incomplete",
            )

        _require(
            manifest.get("input_bundle_frozen") is True,
            "frozen FinMultiTime bundle must declare input_bundle_frozen=true",
        )
        top_formal_status = manifest.get("status", {}).get("formal_m1_run")
        _require(
            top_formal_status in (None, False),
            "frozen FinMultiTime bundle must not declare formal_m1_run=true",
        )
        _require(
            manifest.get("frozen_evidence_contract_version")
            == self.expected_contract_version,
            "frozen FinMultiTime contract version mismatch",
        )
        _require(
            manifest.get("evidence_contract_sha256") == self.expected_contract_sha256,
            "frozen FinMultiTime contract SHA mismatch",
        )
        _require(
            bundle_manifest.get("contract", {}).get("version")
            == self.expected_contract_version,
            "input bundle contract version mismatch",
        )
        _require(
            bundle_manifest.get("contract", {}).get("sha256")
            == self.expected_contract_sha256,
            "input bundle contract SHA mismatch",
        )
        _require(
            manifest.get("input_bundle_manifest", {}).get("sha256") == bundle_manifest_sha,
            "input bundle manifest checksum mismatch",
        )
        _require(
            bundle_manifest.get("packets", {}).get("manifest_sha256")
            == packet_manifest_sha,
            "packet manifest checksum does not match the input bundle manifest",
        )
        _require(
            manifest.get("final_evidence_packet_manifest", {}).get("sha256")
            == packet_manifest_sha,
            "top-level packet manifest checksum mismatch",
        )
        _require(
            packet_manifest_sha == self.expected_packet_manifest_sha256,
            "packet manifest SHA does not match the configured frozen identity",
        )
        _require(
            bundle_manifest.get("status", {}).get("input_bundle_frozen") is True,
            "input bundle manifest is not frozen",
        )
        _require(
            bundle_manifest.get("status", {}).get("formal_m1_run") is False,
            "input bundle manifest represents a formal M1 run",
        )
        if self.bundle_scope == "FORMAL":
            _require(
                manifest.get("final_evidence_packet_count") == EXPECTED_PACKET_COUNT
                and manifest.get("formal_case_count") == EXPECTED_PACKET_COUNT,
                "frozen FinMultiTime bundle must contain exactly 78 formal cases",
            )
            _require(
                bundle_manifest.get("packets", {}).get("count") == EXPECTED_PACKET_COUNT,
                "input bundle manifest packet count mismatch",
            )
        else:
            _require(
                manifest.get("dataset_id") == PILOT_FINMULTITIME_DATASET_ID
                and manifest.get("pilot_case_count") == 4
                and manifest.get("final_evidence_packet_count") == 4
                and manifest.get("formal_case_count") == 0,
                "PILOT bundle identity/count metadata mismatch",
            )
            _require(
                bundle_manifest.get("dataset", {}).get("dataset_id")
                == PILOT_FINMULTITIME_DATASET_ID
                and bundle_manifest.get("dataset", {}).get("symbols") == ["AAPL"]
                and bundle_manifest.get("dataset", {}).get("pilot_sessions")
                == list(PILOT_FINMULTITIME_SESSIONS)
                and bundle_manifest.get("dataset", {}).get("case_count") == 4
                and bundle_manifest.get("packets", {}).get("count") == 4,
                "PILOT bundle schedule/packet metadata mismatch",
            )
        _require(
            self.checksum_manifest.get("bundle_manifest_sha256") == bundle_manifest_sha,
            "checksum inventory does not bind the input bundle manifest",
        )
        _require(
            self.checksum_manifest.get("hash_algorithm") == "SHA-256",
            "frozen checksum inventory must use SHA256",
        )

        hashes = {
            "manifest.json": _sha256_file(self.root / "manifest.json"),
            "manifests/input_bundle_manifest.json": bundle_manifest_sha,
            "manifests/input_bundle_checksums.json": checksum_manifest_sha,
            "manifests/evidence_packet_manifest.json": packet_manifest_sha,
            "manifests/processed_sha256.json": processed_manifest_sha,
        }
        identity = _bundle_identity(hashes)
        _require(
            identity == self.expected_input_bundle_identity,
            "frozen FinMultiTime input bundle identity mismatch",
        )
        return identity

    def _validate_packet_manifest(self) -> dict[tuple[str, str], dict[str, Any]]:
        packets = self.packet_manifest.get("packets")
        _require(
            self.packet_manifest.get("contract_version") == self.expected_contract_version,
            "packet manifest contract version mismatch",
        )
        expected_count = EXPECTED_PACKET_COUNT if self.bundle_scope == "FORMAL" else 4
        _require(
            self.packet_manifest.get("packet_count") == expected_count
            and isinstance(packets, list)
            and len(packets) == expected_count,
            f"packet manifest must contain exactly {expected_count} entries",
        )
        if self.bundle_scope == "PILOT":
            _require(
                self.packet_manifest.get("bundle_scope") == "PILOT"
                and self.packet_manifest.get("pilot_only") is True
                and self.packet_manifest.get("formal_eligible") is False
                and self.packet_manifest.get("symbols") == ["AAPL"]
                and self.packet_manifest.get("pilot_sessions") == list(PILOT_FINMULTITIME_SESSIONS),
                "pilot packet manifest scope/schedule metadata mismatch",
            )
        entries: dict[tuple[str, str], dict[str, Any]] = {}
        counts: dict[str, int] = {}
        for entry in packets:
            _require(isinstance(entry, dict), "packet manifest entry must be an object")
            symbol = entry.get("symbol")
            session = entry.get("decision_session")
            if self.bundle_scope == "FORMAL":
                _require(
                    isinstance(symbol, str) and symbol in EXPECTED_SYMBOLS,
                    f"unexpected frozen packet symbol: {symbol!r}",
                )
            else:
                _require(symbol == "AAPL", f"unexpected pilot packet symbol: {symbol!r}")
            _require(
                isinstance(session, str) and _DATE_RE.fullmatch(session),
                f"invalid frozen packet decision session: {session!r}",
            )
            date.fromisoformat(session)
            key = (symbol, session)
            _require(key not in entries, f"duplicate frozen packet case: {symbol}:{session}")
            json_path = _safe_relative_path(
                self.root, entry.get("json_path", ""), label="packet JSON path"
            )
            text_path = _safe_relative_path(
                self.root, entry.get("text_path", ""), label="packet text path"
            )
            _require(json_path.parts[-3] == "evidence_packets", "unexpected packet JSON path")
            _require(text_path.parts[-3] == "evidence_packets", "unexpected packet text path")
            _require(
                isinstance(entry.get("json_sha256"), str)
                and len(entry["json_sha256"]) == 64,
                "packet manifest JSON SHA is invalid",
            )
            if self.bundle_scope == "PILOT":
                _require(
                    entry.get("packet_status") == "PILOT_FROZEN"
                    and entry.get("pilot_only") is True
                    and entry.get("formal_eligible") is False
                    and isinstance(entry.get("text_sha256"), str)
                    and len(entry["text_sha256"]) == 64,
                    "pilot packet manifest entry metadata is invalid",
                )
            entries[key] = entry
            counts[symbol] = counts.get(symbol, 0) + 1
        if self.bundle_scope == "FORMAL":
            _require(counts == dict.fromkeys(sorted(EXPECTED_SYMBOLS), 26),
                     f"frozen packet symbol/date structure mismatch: {counts}")
        else:
            _require(counts == {"AAPL": 4}, f"pilot packet symbol/date structure mismatch: {counts}")
        return entries

    def _verify_packet_files(self) -> None:
        for (symbol, session), entry in self._packet_entries.items():
            json_path = _safe_relative_path(self.root, entry["json_path"], label="packet JSON path")
            text_path = _safe_relative_path(self.root, entry["text_path"], label="packet text path")
            _require(json_path.is_file(), f"missing frozen packet: {entry['json_path']}")
            _require(text_path.is_file(), f"missing frozen packet text: {entry['text_path']}")
            actual_sha = _sha256_file(json_path)
            _require(
                actual_sha == entry["json_sha256"],
                f"frozen packet SHA mismatch for {symbol}:{session}",
            )
            if self.bundle_scope == "PILOT":
                _require(
                    _sha256_file(text_path) == entry["text_sha256"],
                    f"pilot packet text SHA mismatch for {symbol}:{session}",
                )

    def _verify_checksum_inventory(self) -> None:
        files = self.checksum_manifest.get("files")
        if self.bundle_scope == "FORMAL":
            _require(isinstance(files, list) and len(files) == 251,
                     "frozen checksum inventory must contain 251 files")
        else:
            _require(isinstance(files, list) and files,
                     "pilot checksum inventory must contain files")
        listed: set[str] = set()
        for item in files:
            _require(isinstance(item, dict), "checksum inventory entry must be an object")
            relative = item.get("path")
            path = _safe_relative_path(self.root, relative, label="checksum inventory path")
            _require(relative not in listed, f"duplicate checksum inventory path: {relative}")
            listed.add(relative)
            _require(path.is_file(), f"missing frozen input file: {relative}")
            _require(path.stat().st_size == item.get("bytes"), f"byte count mismatch: {relative}")
            _require(_sha256_file(path) == item.get("sha256"), f"SHA mismatch: {relative}")

        actual = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        expected_unlisted = {"manifests/input_bundle_checksums.json"}
        if self.bundle_scope == "FORMAL":
            expected_unlisted.add("manifest.json")
        _require(
            actual == listed | expected_unlisted,
            "frozen input bundle contains missing or unexpected files",
        )

    def _load_packet(self, symbol: str, decision_session: str) -> tuple[dict[str, Any], dict[str, Any]]:
        key = (str(symbol).strip().upper(), str(decision_session))
        entry = self._packet_entries.get(key)
        _require(entry is not None, f"no exact frozen packet for {key[0]}:{key[1]}")
        path = _safe_relative_path(self.root, entry["json_path"], label="packet JSON path")
        actual_sha = _sha256_file(path)
        _require(actual_sha == entry["json_sha256"], f"frozen packet SHA mismatch for {key[0]}:{key[1]}")
        packet = _read_json(path)
        expected_status = "FINAL_FROZEN" if self.bundle_scope == "FORMAL" else "PILOT_FROZEN"
        _require(packet.get("packet_status") == expected_status, f"packet status mismatch: {key}")
        if self.bundle_scope == "PILOT":
            _require(
                packet.get("pilot_only") is True and packet.get("formal_eligible") is False,
                f"pilot packet scope metadata mismatch: {key}",
            )
        _require(packet.get("symbol") == key[0], f"packet symbol mismatch: {key}")
        _require(packet.get("decision_session") == key[1], f"packet decision session mismatch: {key}")
        _require(packet.get("contract_version") == self.expected_contract_version, f"packet contract mismatch: {key}")
        _require(packet.get("case_id") == f"{key[0]}:{key[1]}", f"packet case identity mismatch: {key}")
        return packet, entry

    def get_packet(self, symbol: str, decision_session: str) -> dict[str, Any]:
        """Return a defensive copy of the exact frozen packet for a case."""
        packet, _ = self._load_packet(symbol, decision_session)
        return copy.deepcopy(packet)

    def case_identity(self, symbol: str, decision_session: str) -> dict[str, Any]:
        """Return the lightweight identity used by cache/checkpoint plumbing."""
        packet, entry = self._load_packet(symbol, decision_session)
        routes = {}
        projections = packet.get("routed_projections")
        _require(isinstance(projections, dict), f"packet has no routed projections: {packet.get('case_id')}")
        for analyst_key in ROUTE_KEYS:
            route = projections.get(ROUTE_PACKET_KEYS[analyst_key])
            _require(isinstance(route, dict) and isinstance(route.get("text"), str),
                     f"packet route is malformed: {analyst_key}")
            routes[analyst_key] = hashlib.sha256(route["text"].encode("utf-8")).hexdigest()
        return {
            "finmultitime_enabled": True,
            "bundle_scope": self.bundle_scope,
            "contract_version": self.expected_contract_version,
            "contract_sha256": self.expected_contract_sha256,
            "input_bundle_identity": self.bundle_identity,
            "case_id": packet["case_id"],
            "packet_json_sha256": entry["json_sha256"],
            "route_sha256": routes,
            "social": "NO_FINMULTITIME_ROUTE",
        }

    def get_routed_evidence(
        self,
        symbol: str,
        decision_session: str,
        analyst_key: str,
    ) -> RoutedEvidence | None:
        """Return only the exact frozen projection allowed for ``analyst_key``."""
        if analyst_key == "social":
            return None
        _require(analyst_key in ROUTE_KEYS, f"unsupported FinMultiTime analyst key: {analyst_key!r}")
        packet, entry = self._load_packet(symbol, decision_session)
        projections = packet.get("routed_projections")
        _require(isinstance(projections, dict), f"packet has no routed projections: {packet.get('case_id')}")
        projection = projections.get(ROUTE_PACKET_KEYS[analyst_key])
        _require(isinstance(projection, dict), f"missing frozen route for {analyst_key}")
        text = projection.get("text")
        sections = projection.get("sections")
        _require(isinstance(text, str) and isinstance(sections, list), f"malformed frozen route: {analyst_key}")
        _require(sections == EXPECTED_ROUTE_SECTIONS[analyst_key], f"frozen route sections mismatch: {analyst_key}")
        _require(packet.get("routing", {}).get("raw_packet_direct_injection") is False,
                 f"raw packet direct injection is enabled: {packet.get('case_id')}")
        route_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        audit_source(
            source_name="finmultitime.frozen_evidence_packet",
            capability="POINT_IN_TIME",
            status="used",
            requested_start=decision_session,
            requested_end=decision_session,
            reason="exact frozen routed projection delivered to analyst-local prompt",
            metadata={
                "finmultitime_enabled": True,
                "bundle_scope": self.bundle_scope,
                "contract_version": self.expected_contract_version,
                "contract_sha256": self.expected_contract_sha256,
                "input_bundle_identity": self.bundle_identity,
                "case_id": packet["case_id"],
                "packet_json_sha256": entry["json_sha256"],
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
            packet_json_sha256=entry["json_sha256"],
            route_sha256=route_sha,
            bundle_identity=self.bundle_identity,
            contract_version=self.expected_contract_version,
            contract_sha256=self.expected_contract_sha256,
            bundle_scope=self.bundle_scope,
        )
