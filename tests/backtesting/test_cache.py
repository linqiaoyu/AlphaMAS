import json

from tradingagents.backtesting.cache import DecisionCache, cache_key


def test_cache_key_changes_with_state_model_and_git():
    base = {"portfolio": "a", "model": "m", "git_sha": "1"}
    key = cache_key(base)
    assert key != cache_key({**base, "portfolio": "b"})
    assert key != cache_key({**base, "model": "x"})
    assert key != cache_key({**base, "git_sha": "2"})


def test_only_complete_success_is_a_hit(tmp_path):
    cache = DecisionCache(tmp_path)
    key = cache_key({"case": 1})
    assert cache.load(key) is None
    cache.save_success(key, {"action": "BUY"})
    assert cache.load(key) == {"action": "BUY"}
    (cache.case_dir(key) / "run_status.json").write_text("{", encoding="utf-8")
    assert cache.load(key) is None
    (cache.case_dir(key) / "run_status.json").write_text(
        json.dumps({"status": "failed"}), encoding="utf-8"
    )
    assert cache.load(key) is None
