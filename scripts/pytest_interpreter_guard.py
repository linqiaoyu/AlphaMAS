"""Pure interpreter guard used before pytest begins collection."""

from collections.abc import Sequence


def is_known_bad_pytest_interpreter(version: Sequence[int]) -> bool:
    """Return whether *version* is the diagnosed Python 3.14 environment."""
    return tuple(version[:2]) == (3, 14)
