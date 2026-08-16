"""Production M2 policy boundary between Prompt Trader and Risk Debate.

The Prompt Trader remains the sole generative Trader.  This module encodes its
complete proposal and the Research Manager plan, executes the frozen local
PA-CTPPO Actor, persists isolated per-symbol fast state, and renders one
deterministic downstream handoff.  It never invokes an LLM.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from scripts.m2.online_delayed_adaptation import (
    C09_FILE_SHA,
    C09_PARAMETER_SHA,
    FAST_PARAMETER_NAMES,
    MAX_GRAD_NORM,
    REPRESENTATION_IDENTITY,
    REWARD_IDENTITY,
    TREE_IDENTITY,
    WEIGHT_DECAY,
    _decode_optimizer_state,
    _encode_optimizer_state,
    _tensor_from_payload,
    _tensor_payload,
    canonical_json_bytes,
    payload_sha,
)
from scripts.m2.pa_ctppo import (
    EXPECTED_FAST_PARAMETERS,
    METHOD_ID,
    OBSERVATION_DIMENSION,
    PPO_CLIP,
    SEED,
    VALUE_LOSS_COEFFICIENT,
    PromptAnchoredActorCritic,
    configure_determinism,
    deterministic_action,
    fast_checkpoint_sha,
)
from scripts.m2.reward_simulator import (
    FORMAL_COMMISSION_BPS,
    FORMAL_SLIPPAGE_BPS,
    MarketBar,
    PortfolioState,
    RewardStatus,
    RewardWindow,
    candidate_rewards,
    simulate_counterfactuals,
)
from scripts.m2.semantic_state_representation import (
    ACTION_ORDER,
    MODEL_REPO,
    MODEL_REVISION,
    REFERENCE_DIMENSION,
    build_actor_observation,
    build_semantic_base,
    mrl_prefix_normalize,
    text_sha256,
    validate_token_count,
)
from scripts.m2.train_global_pa_ctppo import canonical_parameter_sha, sha256_file
from tradingagents.backtesting.calendar import ExchangeSchedule
from tradingagents.backtesting.models import PortfolioSnapshot

SCHEMA_VERSION = "M2-14-PRODUCTION-TRADER-STATE-v1"
ARCHITECTURE_ID = "M2-FINAL-ARCHITECTURE-v1"
ONLINE_CANDIDATE_ID = "O08"
ONLINE_LEARNING_RATE = 1e-3
ONLINE_UPDATE_EPOCHS = 2
MODEL_SNAPSHOT_IDENTITY = "1d7b1bddebe83694815066f5254c5b0c7a1d05febd4e2b9e2120f2ec3fe3c018"
ENCODER_ENVIRONMENT_IDENTITY = "89ef97b30e703f8d67d4510d269083d9aea568e4230040d92a454c6cf40ca892"
A2_PARAMETER_SHA = "60a0fec7b69ef2d0576a9c0894be09c377d573585db827c162279fb27483303e"
A2_FILE_SHA = "4d65dd2c1563b144aee8e79878171feb1ecd990a8bd8e2c584547eaa3a546c9f"

_PROMPT_FINAL = re.compile(
    r"(?im)^FINAL TRANSACTION PROPOSAL:\s*\*\*(BUY|HOLD|SELL)\*\*\s*$"
)
_AUTHORITATIVE_FINAL = re.compile(
    r"(?im)^M2 AUTHORITATIVE TRADER ACTION:\s*\*\*(BUY|HOLD|SELL)\*\*\s*$"
)
_SAFE_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.-]{0,15}$")


def parse_prompt_trader_action(proposal: str) -> str:
    """Read exactly one canonical Prompt Trader terminal action."""

    matches = _PROMPT_FINAL.findall(proposal or "")
    if len(matches) != 1:
        raise ValueError("Prompt Trader proposal must contain exactly one terminal action")
    return matches[0].upper()


def parse_authoritative_m2_action(handoff: str) -> str:
    """Read only M2's explicit authoritative field, never provenance text."""

    matches = _AUTHORITATIVE_FINAL.findall(handoff or "")
    if len(matches) != 1:
        raise ValueError("M2 handoff must contain exactly one authoritative action")
    return matches[0].upper()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _mapping_sha(value: Mapping[str, Any]) -> str:
    return payload_sha(dict(value))


def _portfolio_snapshot(value: Mapping[str, Any]) -> PortfolioSnapshot:
    required = {
        "timestamp", "session", "symbol", "cash", "quantity", "close_price",
        "market_value", "equity", "current_weight", "average_entry_price",
        "unrealized_pnl", "realized_pnl", "cumulative_cost", "current_drawdown",
        "peak_equity",
    }
    if not required <= set(value):
        raise ValueError("M2 portfolio snapshot is incomplete")
    payload = dict(value)
    timestamp = pd.Timestamp(payload["timestamp"])
    if timestamp.tzinfo is None:
        raise ValueError("M2 portfolio timestamp must be timezone-aware")
    payload["timestamp"] = timestamp.to_pydatetime()
    return PortfolioSnapshot(**payload)


def _normalise_embedding(matrix: Any) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.shape != (2, REFERENCE_DIMENSION) or not np.isfinite(values).all():
        raise ValueError("frozen encoder must return two finite 1024-dimensional rows")
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("frozen encoder returned a zero row")
    return (values.astype(np.float64) / norms).astype(np.float32)


class FrozenQwenEncoder:
    """Lazy exact-revision Qwen3 encoder using the M2-09 inference path."""

    def __init__(self, snapshot: Path, manifest_path: Path) -> None:
        self.snapshot = snapshot.resolve()
        if not self.snapshot.is_dir():
            raise ValueError("M2 encoder snapshot directory is missing")
        self.manifest_path = manifest_path.resolve()
        self._verify_snapshot()
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def _verify_snapshot(self) -> None:
        if not self.manifest_path.is_file():
            raise ValueError("M2 encoder snapshot manifest is missing")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected_files = manifest.get("files")
        if (
            manifest.get("identity_sha256") != MODEL_SNAPSHOT_IDENTITY
            or not isinstance(expected_files, list)
        ):
            raise RuntimeError("M2 encoder snapshot manifest identity mismatch")
        actual_files = []
        for path in sorted(item for item in self.snapshot.rglob("*") if item.is_file()):
            resolved = path.resolve()
            digest = hashlib.sha256()
            with resolved.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_files.append(
                {
                    "path": path.relative_to(self.snapshot).as_posix(),
                    "size": resolved.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        identity = hashlib.sha256(
            json.dumps(actual_files, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if actual_files != expected_files or identity != MODEL_SNAPSHOT_IDENTITY:
            raise RuntimeError("M2 encoder snapshot bytes differ from the M2-09 freeze")

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.snapshot, local_files_only=True, padding_side="left"
        )
        self._tokenizer.padding_side = "left"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = AutoModel.from_pretrained(
            self.snapshot,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).to(device)
        self._model.eval()

    @staticmethod
    def _last_token(hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        if bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item()):
            return hidden_states[:, -1]
        lengths = attention_mask.sum(dim=1) - 1
        return hidden_states[range(hidden_states.shape[0]), lengths]

    def __call__(self, texts: list[str]) -> np.ndarray:
        if len(texts) != 2 or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("M2 encoder requires manager and Prompt Trader text")
        self._load()
        assert self._tokenizer is not None and self._model is not None
        vectors = []
        with torch.inference_mode():
            for text in texts:
                encoded = self._tokenizer(
                    text, padding=False, truncation=False, return_tensors="pt"
                )
                validate_token_count(int(encoded["input_ids"].shape[1]))
                encoded = {name: value.to(self._model.device) for name, value in encoded.items()}
                output = self._model(**encoded)
                pooled = self._last_token(output.last_hidden_state, encoded["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                vectors.append(pooled[0].detach().cpu().numpy().astype(np.float32))
        return np.stack(vectors)


@dataclass
class ProductionCredit:
    event_id: str
    symbol: str
    origin_decision_session: str
    origin_observation_identity: str
    semantic_state_identity: str
    portfolio_state_identity: str
    prompt_action: str
    pi_old: tuple[float, float, float]
    selected_action: str
    expected_maturity_session: str
    origin_portfolio_state: dict[str, float]
    decision_close_price: float
    reward_identity: str = REWARD_IDENTITY
    status: str = "PENDING"
    counterfactual_r3: tuple[float, float, float] | None = None
    application_id: str | None = None
    pre_fast_sha: str | None = None
    post_fast_sha: str | None = None
    maturity_input_identity: str | None = None


@dataclass
class ProductionSymbolState:
    symbol: str
    model: PromptAnchoredActorCritic
    optimiser: torch.optim.AdamW
    checkpoint_parameter_identity: str
    checkpoint_file_identity: str
    online_adaptation_enabled: bool
    next_expected_decision_session: str | None = None
    credits: list[ProductionCredit] = field(default_factory=list)
    applied_event_ids: set[str] = field(default_factory=set)
    issued_actions: list[dict[str, Any]] = field(default_factory=list)
    update_records: list[dict[str, Any]] = field(default_factory=list)


class M2ProductionTraderRuntime:
    """One frozen M2 runtime with strictly isolated symbol checkpoints."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        state_root: Path,
        encoder: Callable[[list[str]], np.ndarray],
        checkpoint_parameter_identity: str = C09_PARAMETER_SHA,
        checkpoint_file_identity: str = C09_FILE_SHA,
        online_adaptation_enabled: bool = True,
        learning_rate: float = ONLINE_LEARNING_RATE,
        update_epochs: int = ONLINE_UPDATE_EPOCHS,
    ) -> None:
        if learning_rate != ONLINE_LEARNING_RATE or update_epochs != ONLINE_UPDATE_EPOCHS:
            raise ValueError("production M2 requires frozen O08 hyperparameters")
        if checkpoint_parameter_identity not in {C09_PARAMETER_SHA, A2_PARAMETER_SHA}:
            raise ValueError("unregistered M2 initial checkpoint")
        if checkpoint_file_identity not in {C09_FILE_SHA, A2_FILE_SHA}:
            raise ValueError("unregistered M2 checkpoint file identity")
        self.checkpoint_path = checkpoint_path.resolve()
        self.state_root = state_root.resolve()
        self.encoder = encoder
        self.checkpoint_parameter_identity = checkpoint_parameter_identity
        self.checkpoint_file_identity = checkpoint_file_identity
        self.online_adaptation_enabled = bool(online_adaptation_enabled)
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        self.schedule = ExchangeSchedule()
        configure_determinism(SEED)
        if sha256_file(self.checkpoint_path) != checkpoint_file_identity:
            raise RuntimeError("M2 initial model.pt SHA mismatch")
        raw = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self.base_model = PromptAnchoredActorCritic()
        self.base_model.load_state_dict(raw["model_state"], strict=True)
        if canonical_parameter_sha(self.base_model) != checkpoint_parameter_identity:
            raise RuntimeError("M2 initial canonical parameter SHA mismatch")
        self._states: dict[str, ProductionSymbolState] = {}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> M2ProductionTraderRuntime:
        required = (
            "m2_checkpoint_path",
            "m2_encoder_snapshot_path",
            "m2_encoder_snapshot_manifest_path",
            "m2_state_root",
        )
        missing = [name for name in required if not config.get(name)]
        if missing:
            raise ValueError(f"M2 runtime configuration is missing: {missing}")
        variant = str(config.get("m2_variant", "FULL_M2"))
        if variant == "FULL_M2":
            parameter_sha, file_sha, online = C09_PARAMETER_SHA, C09_FILE_SHA, True
        elif variant == "A1_NO_ONLINE_ADAPTATION":
            parameter_sha, file_sha, online = C09_PARAMETER_SHA, C09_FILE_SHA, False
        elif variant == "A2_NO_GLOBAL_PRETRAINING":
            parameter_sha, file_sha, online = A2_PARAMETER_SHA, A2_FILE_SHA, True
        else:
            raise ValueError("unknown M2 variant")
        expected_research = {
            "m2_method_id": METHOD_ID,
            "m2_representation_identity": REPRESENTATION_IDENTITY,
            "m2_checkpoint_parameter_identity": parameter_sha,
            "m2_checkpoint_file_identity": file_sha,
            "m2_online_candidate_id": ONLINE_CANDIDATE_ID,
            "m2_online_learning_rate": ONLINE_LEARNING_RATE,
            "m2_online_update_epochs": ONLINE_UPDATE_EPOCHS,
            "m2_online_weight_decay": WEIGHT_DECAY,
            "m2_online_gradient_clip": MAX_GRAD_NORM,
        }
        for name, expected in expected_research.items():
            if config.get(name) != expected:
                raise ValueError(f"M2 frozen runtime contract mismatch: {name}")
        if config.get("m2_preformal_evidence_enabled", False) and (
            config.get("m2_preformal_evidence_identity")
            != "3e9bb6e66fcd998c0b4deff30f7d5728c563126d3a1b976e21bbf034174e4420"
            or config.get("m2_preformal_evidence_role") != "E2E_PILOT"
        ):
            raise ValueError("M2 frozen E2E evidence contract mismatch")
        return cls(
            checkpoint_path=Path(str(config["m2_checkpoint_path"])),
            state_root=Path(str(config["m2_state_root"])),
            encoder=FrozenQwenEncoder(
                Path(str(config["m2_encoder_snapshot_path"])),
                Path(str(config["m2_encoder_snapshot_manifest_path"])),
            ),
            checkpoint_parameter_identity=parameter_sha,
            checkpoint_file_identity=file_sha,
            online_adaptation_enabled=online,
        )

    def _state_path(self, symbol: str) -> Path:
        if not _SAFE_SYMBOL.fullmatch(symbol):
            raise ValueError("unsafe M2 symbol")
        return self.state_root / symbol / "runtime_state.json"

    def _fresh(self, symbol: str) -> ProductionSymbolState:
        model = copy.deepcopy(self.base_model)
        model.freeze_global_for_formal()
        names = tuple(name for name, _ in model.fast_named_parameters())
        if names != FAST_PARAMETER_NAMES or model.fast_parameter_count() != EXPECTED_FAST_PARAMETERS:
            raise RuntimeError("M2 fast-parameter boundary changed")
        optimiser = torch.optim.AdamW(
            [parameter for _, parameter in model.fast_named_parameters()],
            lr=self.learning_rate,
            weight_decay=WEIGHT_DECAY,
        )
        return ProductionSymbolState(
            symbol=symbol,
            model=model,
            optimiser=optimiser,
            checkpoint_parameter_identity=self.checkpoint_parameter_identity,
            checkpoint_file_identity=self.checkpoint_file_identity,
            online_adaptation_enabled=self.online_adaptation_enabled,
        )

    def _payload(self, state: ProductionSymbolState) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "architecture_id": ARCHITECTURE_ID,
            "method_id": METHOD_ID,
            "representation_identity": REPRESENTATION_IDENTITY,
            "train_tree_identity": TREE_IDENTITY,
            "reward_identity": REWARD_IDENTITY,
            "online_candidate_id": ONLINE_CANDIDATE_ID,
            "online_learning_rate": self.learning_rate,
            "online_update_epochs": self.update_epochs,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clipping": MAX_GRAD_NORM,
            "symbol": state.symbol,
            "checkpoint_parameter_identity": state.checkpoint_parameter_identity,
            "checkpoint_file_identity": state.checkpoint_file_identity,
            "online_adaptation_enabled": state.online_adaptation_enabled,
            "next_expected_decision_session": state.next_expected_decision_session,
            "fast_parameters": {
                name: _tensor_payload(parameter)
                for name, parameter in state.model.fast_named_parameters()
            },
            "optimiser_state": _encode_optimizer_state(state.optimiser),
            "credits": [asdict(credit) for credit in state.credits],
            "applied_event_ids": sorted(state.applied_event_ids),
            "issued_actions": state.issued_actions,
            "update_records": state.update_records,
        }
        payload["state_identity"] = payload_sha(payload)
        return payload

    def _save(self, state: ProductionSymbolState) -> str:
        payload = self._payload(state)
        _atomic_write(self._state_path(state.symbol), canonical_json_bytes(payload))
        return str(payload["state_identity"])

    def _load(self, symbol: str) -> ProductionSymbolState:
        if symbol in self._states:
            return self._states[symbol]
        path = self._state_path(symbol)
        if not path.exists():
            state = self._fresh(symbol)
            self._states[symbol] = state
            return state
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.pop("state_identity", None)
        if not isinstance(identity, str) or payload_sha(payload) != identity:
            raise RuntimeError("M2 runtime-state identity mismatch")
        expected = {
            "schema_version": SCHEMA_VERSION,
            "architecture_id": ARCHITECTURE_ID,
            "method_id": METHOD_ID,
            "representation_identity": REPRESENTATION_IDENTITY,
            "train_tree_identity": TREE_IDENTITY,
            "reward_identity": REWARD_IDENTITY,
            "online_candidate_id": ONLINE_CANDIDATE_ID,
            "online_learning_rate": self.learning_rate,
            "online_update_epochs": self.update_epochs,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clipping": MAX_GRAD_NORM,
            "symbol": symbol,
            "checkpoint_parameter_identity": self.checkpoint_parameter_identity,
            "checkpoint_file_identity": self.checkpoint_file_identity,
            "online_adaptation_enabled": self.online_adaptation_enabled,
        }
        for name, value in expected.items():
            if payload.get(name) != value:
                raise RuntimeError(f"M2 runtime lineage mismatch: {name}")
        state = self._fresh(symbol)
        state.next_expected_decision_session = payload["next_expected_decision_session"]
        named = dict(state.model.fast_named_parameters())
        if set(payload["fast_parameters"]) != set(FAST_PARAMETER_NAMES):
            raise RuntimeError("persisted M2 fast-parameter set changed")
        for name, value in payload["fast_parameters"].items():
            named[name].data.copy_(_tensor_from_payload(value))
        state.optimiser.load_state_dict(_decode_optimizer_state(payload["optimiser_state"]))
        state.credits = [
            ProductionCredit(
                **{
                    **item,
                    "pi_old": tuple(item["pi_old"]),
                    "counterfactual_r3": (
                        None if item["counterfactual_r3"] is None
                        else tuple(item["counterfactual_r3"])
                    ),
                }
            )
            for item in payload["credits"]
        ]
        state.applied_event_ids = set(payload["applied_event_ids"])
        state.issued_actions = payload["issued_actions"]
        state.update_records = payload["update_records"]
        self._states[symbol] = state
        return state

    def state_identity(self, symbol: str) -> str:
        return str(self._payload(self._load(symbol))["state_identity"])

    def cache_identity_for_decision(self, symbol: str, session: str) -> str:
        """Bind retries to the state that existed before this action was issued."""

        state = self._load(symbol)
        existing = next(
            (item for item in state.issued_actions if item["decision_session"] == session),
            None,
        )
        if existing is not None:
            return str(existing["pre_runtime_state_identity"])
        return str(self._payload(state)["state_identity"])

    def decision_is_persisted(self, symbol: str, session: str) -> bool:
        """Return whether a cache hit has an exact durable M2 action record."""

        state = self._load(symbol)
        return any(item["decision_session"] == session for item in state.issued_actions)

    @staticmethod
    def _global_state(model: PromptAnchoredActorCritic) -> dict[str, Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if name not in FAST_PARAMETER_NAMES
        }

    def _apply(self, state: ProductionSymbolState, credit: ProductionCredit) -> None:
        if credit.status != "MATURED" or credit.counterfactual_r3 is None:
            raise RuntimeError("only MATURED M2 credit may be applied")
        if credit.event_id in state.applied_event_ids:
            raise RuntimeError("duplicate M2 credit application")
        issued = next(item for item in state.issued_actions if item["event_id"] == credit.event_id)
        observation = torch.tensor(issued["actor_observation"], dtype=torch.float32)
        old = torch.tensor(credit.pi_old, dtype=torch.float32)
        rewards = torch.tensor(credit.counterfactual_r3, dtype=torch.float32)
        target = torch.sum(old * rewards).detach()
        advantages = (rewards - target).detach()
        global_before = self._global_state(state.model)
        pre_fast = fast_checkpoint_sha(state.model)
        losses = []
        gradient_norms = []
        for _ in range(self.update_epochs):
            output = state.model(observation.unsqueeze(0))
            ratio = output["probabilities"].squeeze(0) / old
            surrogate = torch.sum(
                old * torch.minimum(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages,
                )
            )
            loss = -surrogate + VALUE_LOSS_COEFFICIENT * (
                output["value"].squeeze(0) - target
            ).square()
            state.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            parameters = [parameter for _, parameter in state.model.fast_named_parameters()]
            gradient_norms.append(float(nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM)))
            state.optimiser.step()
            losses.append(float(loss.detach()))
        for name, before in global_before.items():
            if not torch.equal(before, dict(state.model.named_parameters())[name].detach().cpu()):
                raise RuntimeError(f"global M2 parameter changed online: {name}")
        post_fast = fast_checkpoint_sha(state.model)
        application_id = payload_sha(
            {"event_id": credit.event_id, "pre_fast_sha": pre_fast, "post_fast_sha": post_fast}
        )
        credit.status = "APPLIED"
        credit.pre_fast_sha = pre_fast
        credit.post_fast_sha = post_fast
        credit.application_id = application_id
        state.applied_event_ids.add(credit.event_id)
        state.update_records.append(
            {
                "event_id": credit.event_id,
                "application_id": application_id,
                "pre_fast_sha": pre_fast,
                "post_fast_sha": post_fast,
                "losses": losses,
                "gradient_norms_before_clipping": gradient_norms,
            }
        )

    def issue(self, graph_state: Mapping[str, Any]) -> dict[str, Any]:
        """Issue and durably record the current RL Trader action."""

        symbol = str(graph_state["company_of_interest"]).upper()
        session = str(graph_state["trade_date"])
        manager_text = str(graph_state["investment_plan"])
        prompt_text = str(graph_state["trader_investment_plan"])
        prompt_action = parse_prompt_trader_action(prompt_text)
        snapshot_mapping = graph_state.get("m2_portfolio_snapshot")
        reward_state_mapping = graph_state.get("m2_portfolio_reward_state")
        if not isinstance(snapshot_mapping, Mapping) or not isinstance(reward_state_mapping, Mapping):
            raise ValueError("M2 requires the actual PIT portfolio state")
        snapshot = _portfolio_snapshot(snapshot_mapping)
        if snapshot.symbol != symbol or snapshot.session != session:
            raise ValueError("M2 portfolio state does not match the current case")
        state = self._load(symbol)
        pre_runtime_state_identity = str(self._payload(state)["state_identity"])
        existing = next(
            (item for item in state.issued_actions if item["decision_session"] == session), None
        )
        input_identity = payload_sha(
            {
                "symbol": symbol,
                "decision_session": session,
                "manager_text_sha256": text_sha256(manager_text),
                "prompt_text_sha256": text_sha256(prompt_text),
                "prompt_action": prompt_action,
                "portfolio_snapshot_identity": _mapping_sha(snapshot_mapping),
            }
        )
        if existing is not None:
            if existing["decision_input_identity"] != input_identity:
                raise RuntimeError("safe resume encountered changed M2 decision input")
            return dict(existing["node_output"])
        if state.next_expected_decision_session not in {None, session}:
            raise RuntimeError("M2 next expected decision/session mismatch")
        embeddings = _normalise_embedding(self.encoder([manager_text, prompt_text]))
        selected = mrl_prefix_normalize(embeddings, REFERENCE_DIMENSION)
        semantic = build_semantic_base(selected[0:1], selected[1:2], [prompt_action])[0]
        observation_np = build_actor_observation(semantic, snapshot)
        if observation_np.shape != (OBSERVATION_DIMENSION,):
            raise RuntimeError("M2 Actor observation dimension changed")
        observation = torch.from_numpy(observation_np.copy())
        pre_fast_sha = fast_checkpoint_sha(state.model)
        with torch.no_grad():
            policy = state.model(observation)
        probabilities = tuple(float(item) for item in policy["probabilities"].tolist())
        logits = tuple(float(item) for item in policy["policy_logits"].tolist())
        prompt_index = ACTION_ORDER.index(prompt_action)
        rl_action = deterministic_action(policy["probabilities"], prompt_index)
        override = rl_action != prompt_action
        handoff = (
            "M2 AUTHORITATIVE TRADER HANDOFF\n"
            f"M2 AUTHORITATIVE TRADER ACTION: **{rl_action}**\n"
            "Action source: frozen M2 Actor (no additional LLM call)\n"
            f"Prompt action: {prompt_action}\n"
            f"Override: {str(override).lower()}\n\n"
            "--- BEGIN NON-AUTHORITATIVE PROMPT TRADER PROVENANCE ---\n"
            f"{prompt_text}\n"
            "--- END NON-AUTHORITATIVE PROMPT TRADER PROVENANCE ---"
        )
        if parse_authoritative_m2_action(handoff) != rl_action:
            raise AssertionError("authoritative M2 handoff parser mismatch")
        semantic_identity = hashlib.sha256(semantic.tobytes()).hexdigest()
        observation_identity = hashlib.sha256(observation_np.tobytes()).hexdigest()
        event_id = payload_sha(
            {"symbol": symbol, "decision_session": session, "observation": observation_identity}
        )
        maturity = self.schedule.calendar.sessions_window(session, 6)[-1].date().isoformat()
        reward_state = {
            name: float(reward_state_mapping[name])
            for name in ("cash", "quantity", "average_entry_price", "open_position_commission")
        }
        credit = ProductionCredit(
            event_id=event_id,
            symbol=symbol,
            origin_decision_session=session,
            origin_observation_identity=observation_identity,
            semantic_state_identity=semantic_identity,
            portfolio_state_identity=_mapping_sha(reward_state),
            prompt_action=prompt_action,
            pi_old=probabilities,
            selected_action=rl_action,
            expected_maturity_session=maturity,
            origin_portfolio_state=reward_state,
            decision_close_price=float(snapshot.close_price),
        )
        node_output = {
            "trader_investment_plan": handoff,
            "prompt_trader_proposal_original": prompt_text,
            "prompt_trader_action": prompt_action,
            "m2_rl_action": rl_action,
            "m2_override": override,
            "m2_trader_handoff_metadata": {
                "schema_version": "M2-TRADER-HANDOFF-v1",
                "authoritative_action": rl_action,
                "action_source": "FROZEN_M2_ACTOR",
                "prompt_action": prompt_action,
                "override": override,
                "prompt_trader_proposal_sha256": text_sha256(prompt_text),
                "research_manager_plan_sha256": text_sha256(manager_text),
                "semantic_state_identity": semantic_identity,
                "actor_observation_identity": observation_identity,
                "actor_observation_dimension": int(observation_np.shape[0]),
                "portfolio_state": observation_np[-4:].tolist(),
                "pre_decision_fast_parameter_identity": pre_fast_sha,
                "actor_logits": list(logits),
                "actor_probabilities": list(probabilities),
                "event_id": event_id,
                "expected_maturity_session": maturity,
                "checkpoint_parameter_identity": self.checkpoint_parameter_identity,
                "checkpoint_file_identity": self.checkpoint_file_identity,
                "representation_identity": REPRESENTATION_IDENTITY,
                "encoder_repo": MODEL_REPO,
                "encoder_revision": MODEL_REVISION,
                "no_second_llm_call": True,
            },
        }
        next_session = graph_state.get("m2_next_decision_session")
        state.next_expected_decision_session = (
            str(next_session) if isinstance(next_session, str) and next_session else None
        )
        state.credits.append(credit)
        record = {
            "event_id": event_id,
            "decision_session": session,
            "decision_input_identity": input_identity,
            "pre_runtime_state_identity": pre_runtime_state_identity,
            "actor_observation": observation_np.tolist(),
            "node_output": node_output,
        }
        state.issued_actions.append(record)
        self._save(state)
        return node_output

    @staticmethod
    def _market_bars(history: pd.DataFrame, sessions: list[str]) -> tuple[MarketBar, ...]:
        rows = []
        for session in sessions:
            timestamp = pd.Timestamp(session)
            if timestamp not in history.index:
                return ()
            bar = history.loc[timestamp]
            rows.append(
                MarketBar(
                    session=session,
                    open_price=float(bar["Open"]),
                    close_price=float(bar["Close"]),
                    dividend_per_share=float(bar.get("Dividends", 0.0)),
                    split_ratio=float(bar.get("Stock Splits", 0.0)),
                )
            )
        return tuple(rows)

    def mature_visible(
        self,
        symbol: str,
        *,
        visible_market_history: pd.DataFrame,
        cutoff_session: str,
        allow_update_for_later_decision: bool,
    ) -> list[dict[str, Any]]:
        """Mature credits after the current action; apply only if useful later."""

        state = self._load(symbol)
        events = []
        normalized = visible_market_history.copy()
        normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None).normalize()
        for credit in state.credits:
            if credit.status != "PENDING" or credit.expected_maturity_session > cutoff_session:
                continue
            sessions = self.schedule.calendar.sessions_window(
                credit.origin_decision_session, 6
            )[1:].tz_localize(None)
            labels = [item.date().isoformat() for item in sessions]
            bars = self._market_bars(normalized, labels)
            if len(bars) != 5:
                continue
            window = RewardWindow(
                symbol=symbol,
                decision_session=credit.origin_decision_session,
                decision_close_price=credit.decision_close_price,
                bars=bars,
            )
            simulation = simulate_counterfactuals(
                window,
                PortfolioState(**credit.origin_portfolio_state),
                commission_bps=FORMAL_COMMISSION_BPS,
                slippage_bps=FORMAL_SLIPPAGE_BPS,
                schedule=self.schedule,
            )
            if simulation.status is not RewardStatus.MATURED or simulation.outcomes is None:
                continue
            rewards = tuple(
                candidate_rewards(simulation.outcomes[action])[REWARD_IDENTITY]
                for action in ACTION_ORDER
            )
            credit.counterfactual_r3 = rewards
            credit.maturity_input_identity = payload_sha(
                {
                    "event_id": credit.event_id,
                    "sessions": labels,
                    "bars": [asdict(bar) for bar in bars],
                    "rewards": rewards,
                }
            )
            credit.status = "MATURED"
            if self.online_adaptation_enabled and allow_update_for_later_decision:
                self._apply(state, credit)
            else:
                credit.status = "ARCHIVED"
            events.append(
                {
                    "event_id": credit.event_id,
                    "status": credit.status,
                    "maturity_input_identity": credit.maturity_input_identity,
                    "counterfactual_r3": list(rewards),
                }
            )
        if events:
            self._save(state)
        return events


def create_m2_trader_node(runtime: M2ProductionTraderRuntime):
    """Return the local graph node inserted after the unchanged Prompt Trader."""

    def m2_trader_node(state: Mapping[str, Any]) -> dict[str, Any]:
        return runtime.issue(state)

    return m2_trader_node
