from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.finmultitime.generate_qwen_synthetic_chart import build_png
from scripts.finmultitime.run_qwen_caption import (
    CAPTION_FIELDS,
    MODEL_REPO,
    MODEL_REVISION,
    CaptionValidationError,
    FreezeValidationError,
    build_model_messages,
    load_and_validate_identities,
    parse_canonical_caption,
    validate_mode_and_image,
)


@pytest.fixture(autouse=True)
def _isolate_config():
    """The offline caption utilities do not import the formal Agent runtime."""
    yield


def _caption(**overrides: str) -> str:
    values = dict.fromkeys(CAPTION_FIELDS, "none visible")
    values.update(overrides)
    return json.dumps(values)


def _freeze(contract: dict, contract_path: Path) -> dict:
    prompt = contract["qwen_image_adapter"]["prompt"]
    schema_json = json.dumps(list(CAPTION_FIELDS), separators=(",", ":"))
    return {
        "status": "FROZEN_FOR_SMOKE",
        "qwen": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "model_class": "Qwen3VLForConditionalGeneration",
            "processor_class": "Qwen3VLProcessor",
        },
        "generation": {
            "device": "cuda",
            "dtype": "bfloat16",
            "batch_size": 1,
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": 256,
            "python_seed": 0,
            "torch_seed": 0,
            "cuda_seed": 0,
        },
        "contract": {
            "contract_version": contract["packet_version"],
            "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "schema_sha256": hashlib.sha256(schema_json.encode()).hexdigest(),
            "max_caption_chars": 900,
        },
    }


@pytest.fixture
def identities(tmp_path: Path) -> tuple[Path, Path, Path]:
    prompt = "exact prompt"
    contract = {
        "packet_version": "M1-FINMULTITIME-v1.0.1",
        "qwen_image_adapter": {
            "prompt": prompt,
            "caption_schema": list(CAPTION_FIELDS),
            "max_caption_chars": 900,
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    freeze = _freeze(contract, contract_path)
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze))
    qwen_manifest = {
        "evidence_contract_version": contract["packet_version"],
        "evidence_contract_sha256": freeze["contract"]["contract_sha256"],
        "prompt_sha256": freeze["contract"]["prompt_sha256"],
        "caption_schema": list(CAPTION_FIELDS),
        "caption_schema_sha256": freeze["contract"]["schema_sha256"],
        "max_caption_chars": 900,
        "caption_status_counts": {"GENERATED": 0, "PENDING": 52, "NOT_APPLICABLE": 26},
        "images": [],
    }
    manifest_path = tmp_path / "qwen.json"
    manifest_path.write_text(json.dumps(qwen_manifest))
    return contract_path, freeze_path, manifest_path


def test_environment_freeze_parser_accepts_exact_identities(identities: tuple[Path, Path, Path]):
    contract_path, freeze_path, manifest_path = identities
    contract, freeze, manifest = load_and_validate_identities(
        contract_path, freeze_path, manifest_path, mode="SMOKE"
    )
    assert contract["packet_version"] == "M1-FINMULTITIME-v1.0.1"
    assert freeze["qwen"]["revision"] == MODEL_REVISION
    assert manifest["caption_status_counts"]["GENERATED"] == 0


def test_environment_freeze_requires_exact_revision(identities: tuple[Path, Path, Path]):
    contract_path, freeze_path, manifest_path = identities
    freeze = json.loads(freeze_path.read_text())
    freeze["qwen"]["revision"] = "main"
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(FreezeValidationError, match="revision"):
        load_and_validate_identities(contract_path, freeze_path, manifest_path, mode="SMOKE")


def test_environment_freeze_rejects_contract_or_prompt_hash_mismatch(
    identities: tuple[Path, Path, Path],
):
    contract_path, freeze_path, manifest_path = identities
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FreezeValidationError, match="prompt_sha256"):
        load_and_validate_identities(contract_path, freeze_path, manifest_path, mode="SMOKE")


def test_model_input_builder_contains_only_image_and_exact_prompt(tmp_path: Path):
    image = tmp_path / "synthetic.png"
    image.write_bytes(build_png())
    messages = build_model_messages(image, "exact frozen prompt")
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image.resolve())},
                {"type": "text", "text": "exact frozen prompt"},
            ],
        }
    ]
    serialized = json.dumps(messages)
    assert "ticker" not in serialized.casefold()
    assert "company" not in serialized.casefold()


def test_canonical_parser_preserves_exact_nine_field_order():
    parsed = parse_canonical_caption(_caption(trend="steady rise"))
    assert list(json.loads(parsed)) == list(CAPTION_FIELDS)
    assert json.loads(parsed)["trend"] == "steady rise"


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.replace('"confidence": "none visible", ', ""), "missing"),
        (lambda value: value[:-1] + ', "unknown": "x"}', "unknown"),
        (
            lambda value: value[:-1] + ', "trend": "duplicate"}',
            "duplicate",
        ),
    ],
)
def test_canonical_parser_rejects_missing_unknown_or_duplicate(mutator, message: str):
    raw = json.dumps(dict.fromkeys(CAPTION_FIELDS, "none visible"))
    if message == "missing":
        value = json.loads(raw)
        value.pop("confidence")
        raw = json.dumps(value)
    else:
        raw = mutator(raw)
    with pytest.raises(CaptionValidationError, match=message):
        parse_canonical_caption(raw)


def test_canonical_parser_rejects_more_than_900_chars():
    with pytest.raises(CaptionValidationError, match="900"):
        parse_canonical_caption(_caption(trend="x" * 900))


def test_smoke_refuses_formal_image_and_formal_requires_frozen_code(tmp_path: Path):
    formal = tmp_path / "formal.png"
    formal.write_bytes(b"formal")
    formal_sha = hashlib.sha256(formal.read_bytes()).hexdigest()
    manifest = {"images": [{"sha256": formal_sha}]}
    environment = {"status": "FROZEN_FOR_SMOKE", "smoke": {}}
    with pytest.raises(FreezeValidationError, match="formal image"):
        validate_mode_and_image(
            "SMOKE",
            formal,
            environment,
            manifest,
            captioning_code_sha=None,
            repository=tmp_path,
        )
    with pytest.raises(FreezeValidationError, match="captioning_code_sha"):
        validate_mode_and_image(
            "FORMAL",
            formal,
            environment,
            manifest,
            captioning_code_sha=None,
            repository=tmp_path,
        )


def test_synthetic_png_is_deterministic_and_has_no_formal_metadata():
    first = build_png()
    second = build_png()
    assert first == second
    assert first.startswith(b"\x89PNG")
    assert b"AMZN" not in first
    assert b"JPM" not in first
