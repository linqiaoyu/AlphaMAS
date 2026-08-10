"""Content-addressed, atomic cache for expensive agent decisions."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


class DecisionCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def case_dir(self, key: str) -> Path:
        return self.root / key[:2] / key

    def load(self, key: str) -> dict[str, Any] | None:
        bundle = self.load_bundle(key)
        return bundle["decision"] if bundle else None

    def load_bundle(
        self, key: str, *, require_artifacts: bool = False,
        require_usage: bool = False,
    ) -> dict[str, Any] | None:
        """Load a successful entry, optionally requiring formal-case artifacts."""
        path = self.case_dir(key)
        status_path, decision_path = path / "run_status.json", path / "decision.json"
        if not status_path.is_file() or not decision_path.is_file():
            return None
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if status.get("status") != "success":
            return None
        stats: dict[str, Any] = {}
        source_audit: dict[str, Any] = {}
        usage: list[dict[str, Any]] = []
        try:
            if (path / "stats.json").is_file():
                stats = json.loads((path / "stats.json").read_text(encoding="utf-8"))
            if (path / "source_audit.json").is_file():
                source_audit = json.loads(
                    (path / "source_audit.json").read_text(encoding="utf-8")
                )
            if (path / "llm_usage.json").is_file():
                usage = json.loads((path / "llm_usage.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(stats, dict) or not isinstance(source_audit, dict):
            return None
        if not isinstance(usage, list):
            return None
        if require_artifacts:
            sources = source_audit.get("sources")
            reports = path / "reports"
            if (
                not isinstance(sources, list)
                or not sources
                or not (reports / "complete_report.md").is_file()
            ):
                return None
        if require_usage and not usage:
            return None
        return {
            "decision": decision,
            "stats": stats,
            "source_audit": source_audit,
            "llm_usage": usage,
            "path": path,
        }

    def save_success(self, key: str, decision: dict[str, Any], extras: dict[str, Any] | None = None) -> Path:
        target = self.case_dir(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=target.parent))
        try:
            (tmp / "decision.json").write_text(json.dumps(decision, indent=2, default=str) + "\n", encoding="utf-8")
            (tmp / "stats.json").write_text(json.dumps(extras or {}, indent=2, default=str) + "\n", encoding="utf-8")
            (tmp / "source_audit.json").write_text(
                json.dumps((extras or {}).get("source_audit", {}), indent=2) + "\n",
                encoding="utf-8",
            )
            (tmp / "llm_usage.json").write_text(
                json.dumps((extras or {}).get("llm_usage", []), indent=2, default=str) + "\n",
                encoding="utf-8",
            )
            (tmp / "reports").mkdir()
            report_path = (extras or {}).get("report_path")
            if report_path and Path(report_path).is_dir():
                shutil.copytree(report_path, tmp / "reports", dirs_exist_ok=True)
            (tmp / "run_status.json").write_text('{"status": "success"}\n', encoding="utf-8")
            if target.exists():
                backup = target.with_name(target.name + ".old")
                if backup.exists():
                    raise FileExistsError(backup)
                target.replace(backup)
                tmp.replace(target)
                shutil.rmtree(backup)
            else:
                os.replace(tmp, target)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        return target
