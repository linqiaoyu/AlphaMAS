"""yfinance-based news data fetching functions."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
from dateutil.relativedelta import relativedelta

from .config import get_config
from .stockstats_utils import yf_retry

_LIVE_REPLAY_GRACE_DAYS = 2


def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()


def _cache_root() -> Path:
    root = Path(get_config()["data_cache_dir"]) / "yfinance_news"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path(namespace: str, cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    directory = _cache_root() / namespace
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


def _live_fetch_allowed(requested_end_date: date) -> bool:
    cutoff = _today_utc_date() - timedelta(days=_LIVE_REPLAY_GRACE_DAYS)
    return requested_end_date >= cutoff


def _serialize_articles(articles: list[dict]) -> list[dict]:
    serialized = []
    for article in articles:
        pub_date = article.get("pub_date")
        serialized.append(
            {
                "title": article.get("title", "No title"),
                "summary": article.get("summary", ""),
                "publisher": article.get("publisher", "Unknown"),
                "link": article.get("link", ""),
                "pub_date": pub_date.isoformat() if isinstance(pub_date, datetime) else None,
            }
        )
    return serialized


def _deserialize_articles(payload_articles: list[dict]) -> list[dict]:
    deserialized = []
    for article in payload_articles:
        pub_date = None
        pub_date_str = article.get("pub_date")
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str)
            except ValueError:
                pub_date = None

        deserialized.append(
            {
                "title": article.get("title", "No title"),
                "summary": article.get("summary", ""),
                "publisher": article.get("publisher", "Unknown"),
                "link": article.get("link", ""),
                "pub_date": pub_date,
            }
        )
    return deserialized


def _load_recent_cache(namespace: str, cache_key: str) -> list[dict] | None:
    path = _cache_path(namespace, cache_key)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    return _deserialize_articles(payload.get("articles", []))


def _save_recent_cache(namespace: str, cache_key: str, articles: list[dict]) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "articles": _serialize_articles(articles),
    }
    _cache_path(namespace, cache_key).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _historical_news_disabled_message(scope: str, start_date: str, end_date: str) -> str:
    return (
        f"[NEWS_UNAVAILABLE] Historical yfinance news unavailable for {scope} between {start_date} and {end_date}. "
        "This simplified baseline disables historical news replay because Yahoo Finance live "
        "news/search endpoints are not reliable historical archives."
    )


def _extract_article_data(article: dict) -> dict:
    """Extract article data from yfinance news format (handles nested 'content' structure)."""
    if "content" in article:
        content = article["content"]
        title = content.get("title", "No title")
        summary = content.get("summary", "")
        provider = content.get("provider", {})
        publisher = provider.get("displayName", "Unknown")

        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "")

        pub_date_str = content.get("pubDate", "")
        pub_date = None
        if pub_date_str:
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_date = None

        return {
            "title": title,
            "summary": summary,
            "publisher": publisher,
            "link": link,
            "pub_date": pub_date,
        }

    pub_date = None
    pub_date_str = article.get("pubDate")
    if pub_date_str:
        try:
            pub_date = datetime.fromisoformat(str(pub_date_str).replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            pub_date = None

    if pub_date is None:
        provider_publish_time = article.get("providerPublishTime")
        if provider_publish_time is not None:
            try:
                pub_ts = float(provider_publish_time)
                pub_date = datetime.fromtimestamp(pub_ts, tz=timezone.utc)
            except (ValueError, TypeError, OSError):
                pub_date = None

    return {
        "title": article.get("title", "No title"),
        "summary": article.get("summary", ""),
        "publisher": article.get("publisher", "Unknown"),
        "link": article.get("link", ""),
        "pub_date": pub_date,
    }


def _filter_articles_by_day(
    articles: list[dict],
    start_day: date,
    end_day: date,
    limit: int | None = None,
) -> list[dict]:
    filtered = []
    for article in articles:
        pub_date = article.get("pub_date")
        if not pub_date:
            continue

        pub_day = pub_date.replace(tzinfo=None).date()
        if start_day <= pub_day <= end_day:
            filtered.append(article)

    if limit is not None:
        return filtered[:limit]
    return filtered


def _format_articles(header: str, articles: list[dict]) -> str:
    body = ""
    for article in articles:
        body += f"### {article['title']} (source: {article['publisher']})\n"
        if article.get("summary"):
            body += f"{article['summary']}\n"
        if article.get("link"):
            body += f"Link: {article['link']}\n"
        body += "\n"
    return f"{header}\n\n{body}"


def get_news_yfinance(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """
    Retrieve news for a specific stock ticker using yfinance.

    Historical rigor note:
    This simplified baseline does not replay historical Yahoo news.
    Older dates fail closed instead of fabricating a historical feed.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    ticker_upper = ticker.upper()
    cache_key = f"{ticker_upper}|{start_date}|{end_date}"

    if not _live_fetch_allowed(end_dt.date()):
        return _historical_news_disabled_message(ticker_upper, start_date, end_date)

    try:
        articles = _load_recent_cache("ticker", cache_key)
        if articles is None:
            stock = yf.Ticker(ticker_upper)
            raw_articles = yf_retry(lambda: stock.get_news(count=20)) or []
            articles = [_extract_article_data(article) for article in raw_articles]
            _save_recent_cache("ticker", cache_key, articles)

        filtered_articles = _filter_articles_by_day(
            articles,
            start_dt.date(),
            end_dt.date(),
        )
        if not filtered_articles:
            return f"[NEWS_EMPTY] No news found for {ticker_upper} between {start_date} and {end_date}"

        return _format_articles(
            f"## {ticker_upper} News, from {start_date} to {end_date}:",
            filtered_articles,
        )

    except Exception as e:
        return f"Error fetching news for {ticker_upper}: {str(e)}"


def get_global_news_yfinance(
    curr_date: str,
    look_back_days: int = 7,
    limit: int = 10,
) -> str:
    """
    Retrieve global/macro economic news using yfinance Search.

    Historical rigor note:
    This simplified baseline does not replay historical Yahoo macro news.
    Older dates fail closed instead of fabricating a historical feed.
    """
    search_queries = [
        "stock market economy",
        "Federal Reserve interest rates",
        "inflation economic outlook",
        "global markets trading",
    ]

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")
    cache_key = f"{curr_date}|{look_back_days}|{limit}"

    if not _live_fetch_allowed(curr_dt.date()):
        return _historical_news_disabled_message(
            "global market news", start_date, curr_date
        ).replace("[NEWS_UNAVAILABLE]", "[GLOBAL_NEWS_UNAVAILABLE]", 1)

    try:
        articles = _load_recent_cache("global", cache_key)
        if articles is None:
            all_articles = []
            seen_titles = set()

            for query in search_queries:
                search = yf_retry(
                    lambda q=query: yf.Search(
                        query=q,
                        news_count=limit,
                        enable_fuzzy_query=True,
                    )
                )

                for article in getattr(search, "news", []) or []:
                    data = _extract_article_data(article)
                    title = data["title"]
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_articles.append(data)

                if len(all_articles) >= limit:
                    break

            articles = all_articles
            _save_recent_cache("global", cache_key, articles)

        filtered_articles = _filter_articles_by_day(
            articles,
            start_dt.date(),
            curr_dt.date(),
            limit=limit,
        )
        if not filtered_articles:
            return f"[GLOBAL_NEWS_EMPTY] No global news found for {curr_date}"

        return _format_articles(
            f"## Global Market News, from {start_date} to {curr_date}:",
            filtered_articles,
        )

    except Exception as e:
        return f"Error fetching global news: {str(e)}"
