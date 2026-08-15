"""Strict structural comparison with tolerance only for finite float leaves."""

from __future__ import annotations

import math
from typing import Any

FLOAT_ABS_TOLERANCE = 1e-12
FLOAT_REL_TOLERANCE = 1e-12


def assert_semantically_equivalent(actual: Any, expected: Any, path: str = "$") -> None:
    """Assert exact JSON structure and tolerant finite floating-point values."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        assert type(actual) is bool and type(expected) is bool, f"type drift at {path}"
        assert actual is expected, f"boolean drift at {path}"
        return

    if actual is None or expected is None:
        assert actual is None and expected is None, f"null drift at {path}"
        return

    assert type(actual) is type(expected), f"type drift at {path}"

    if isinstance(actual, dict):
        assert actual.keys() == expected.keys(), f"dictionary keys drift at {path}"
        for key in actual:
            assert_semantically_equivalent(actual[key], expected[key], f"{path}.{key}")
        return

    if isinstance(actual, list):
        assert len(actual) == len(expected), f"list length drift at {path}"
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            assert_semantically_equivalent(
                actual_item, expected_item, f"{path}[{index}]"
            )
        return

    if isinstance(actual, float):
        assert math.isfinite(actual) and math.isfinite(expected), (
            f"non-finite float at {path}"
        )
        assert math.isclose(
            actual,
            expected,
            abs_tol=FLOAT_ABS_TOLERANCE,
            rel_tol=FLOAT_REL_TOLERANCE,
        ), f"float drift at {path}: {actual!r} != {expected!r}"
        return

    if isinstance(actual, (str, int)):
        assert actual == expected, f"value drift at {path}"
        return

    raise AssertionError(f"unsupported semantic value at {path}: {type(actual).__name__}")
