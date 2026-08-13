from scripts.pytest_interpreter_guard import is_known_bad_pytest_interpreter


def test_only_python_314_is_rejected_by_collection_guard() -> None:
    expected = {
        (3, 10): False,
        (3, 11): False,
        (3, 12): False,
        (3, 13): False,
        (3, 14): True,
    }

    assert {
        version: is_known_bad_pytest_interpreter(version)
        for version in expected
    } == expected
