"""
AI News Tool
Fetches the latest AI and ML news using DuckDuckGo.
No API key required.

Two public functions:
  run_ai_news(message)  -> plain string for chat bubble (router calls this)
  get_ai_news_feed()    -> list[dict] for the sidebar news tab (app.py calls this)

Both share the same underlying _fetch_raw() fetch logic.
"""

from datetime import datetime, timezone
from urllib.parse import urlparse

from ddgs import DDGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURE = "Sorry I could not fetch AI news right now. Please try again."
_SUMMARY_MAX = 180
_FEED_MAX = 8    # max cards in the news tab
_CHAT_MAX = 5    # max items in chat response

# News search queries — ddgs.news() gives date + source fields
_NEWS_QUERIES = [
    "artificial intelligence news",
    "large language model news",
    "machine learning research",
]

# Tag detection rules — checked in order, first match wins.
# Anthropic BEFORE OpenAI to avoid 'Claude beats GPT-4' -> OpenAI misfire.
_TAG_RULES: list[tuple[list[str], str]] = [
    (["anthropic", "claude"],                                    "Anthropic"),
    (["openai", "chatgpt", "gpt-4", "gpt-5", "gpt4",
      "gpt5", "sora", "dall-e"],                                 "OpenAI"),
    (["meta ", "llama", "facebook ai"],                          "Meta"),
    (["google", "gemini", "deepmind", "bard"],                   "Google"),
    (["microsoft", "copilot", "phi-3", "phi-4", "azure ai"],     "Microsoft"),
    (["mistral", "mixtral", "huggingface", "hugging face",
      "deepseek", "qwen", "alibaba"],                            "Research"),
]

# Tag -> color hex (used by app.py to render colored badges)
TAG_COLORS: dict[str, str] = {
    "Anthropic":  "#a78bfa",
    "OpenAI":     "#ef4444",
    "Meta":       "#3b82f6",
    "Google":     "#22c55e",
    "Microsoft":  "#60a5fa",
    "Research":   "#D97706",
    "AI News":    "#6b7280",
}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, max_len: int = _SUMMARY_MAX) -> str:
    """Truncate to max_len chars, stripping newlines."""
    t = (text or "").strip().replace("\n", " ")
    return (t[:max_len] + "...") if len(t) > max_len else t


def _detect_tag(title: str) -> str:
    """
    Detect the news category from the headline title only.
    Title-only (not body) avoids 'Anthropic Claude beats GPT-4' -> OpenAI misfire.
    """
    text = title.lower()
    for keywords, tag in _TAG_RULES:
        if any(kw in text for kw in keywords):
            return tag
    return "AI News"


def _extract_source(url: str, fallback: str = "") -> str:
    """Extract clean domain name from URL e.g. 'techcrunch.com'."""
    if not url:
        return fallback or "unknown"
    try:
        host = urlparse(url).netloc
        # strip www. prefix
        return host.removeprefix("www.") or fallback or "unknown"
    except Exception:
        return fallback or "unknown"


def _format_time(date_str: str) -> str:
    """
    Convert ddgs date string to relative time label.
    ddgs.news() returns ISO-8601 strings like '2024-01-15T10:30:00+00:00'
    or sometimes plain strings like 'Mon, 15 Jan 2024 10:30:00 +0000'.
    Falls back to 'recently' on any parse error.
    """
    if not date_str:
        return "recently"
    try:
        # Try ISO format first
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 3600:
            m = max(1, seconds // 60)
            return f"{m}m ago"
        elif seconds < 86400:
            return f"{seconds // 3600}h ago"
        else:
            return f"{seconds // 86400}d ago"
    except Exception:
        return "recently"


def _fetch_raw(max_results: int = _FEED_MAX) -> list[dict]:
    """
    Shared fetch logic used by both run_ai_news() and get_ai_news_feed().
    Uses ddgs.news() for structured date + source fields.
    Deduplicates by URL. Returns raw DDGS dicts.
    """
    results: list[dict] = []
    seen_urls: set[str] = set()

    with DDGS() as ddgs:
        for query in _NEWS_QUERIES:
            try:
                chunks = list(ddgs.news(query, max_results=5))
                for item in chunks:
                    url = item.get("url", "") or item.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(item)
            except Exception:
                continue
            if len(results) >= max_results:
                break

    return results[:max_results]


# ---------------------------------------------------------------------------
# Public: news tab feed
# ---------------------------------------------------------------------------

def get_ai_news_feed() -> list[dict]:
    """
    Fetch structured news for the sidebar AI News tab.

    Returns a list of dicts, each with:
        title   : str  - headline
        summary : str  - short description (~180 chars)
        source  : str  - domain e.g. 'techcrunch.com'
        time    : str  - relative time e.g. '2h ago'
        tag     : str  - category e.g. 'OpenAI', 'Meta', 'Research'
        color   : str  - hex color for the tag badge
        url     : str  - article link

    Returns empty list on failure (caller should handle gracefully).
    """
    try:
        raw = _fetch_raw(max_results=_FEED_MAX)
        feed = []

        for item in raw:
            title   = (item.get("title") or "").strip()
            body    = (item.get("body") or "").strip()
            url     = item.get("url") or item.get("href", "")
            source  = item.get("source") or _extract_source(url)
            date    = item.get("date") or item.get("published", "")
            tag     = _detect_tag(title)

            if not title:
                continue

            feed.append({
                "title":   title,
                "summary": _truncate(body, _SUMMARY_MAX),
                "source":  source,
                "time":    _format_time(date),
                "tag":     tag,
                "color":   TAG_COLORS.get(tag, "#6b7280"),
                "url":     url,
            })

        return feed

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public: chat tool (unchanged behavior)
# ---------------------------------------------------------------------------

def run_ai_news(message: str) -> str:
    """
    Fetch latest AI news for the chat bubble.
    Called by the router when user asks about AI news.

    Args:
        message: User message (not used directly)

    Returns:
        Formatted plain string with latest AI headlines.
    """
    try:
        raw = _fetch_raw(max_results=_CHAT_MAX)

        if not raw:
            return _FAILURE

        lines = ["Here is the latest AI news:", ""]

        for i, item in enumerate(raw[:_CHAT_MAX], start=1):
            title   = (item.get("title") or "").strip() or "(no title)"
            body    = (item.get("body") or "").strip()
            source  = item.get("source") or _extract_source(item.get("url","") or item.get("href",""))
            date    = item.get("date") or item.get("published", "")
            time    = _format_time(date)
            summary = _truncate(body, _SUMMARY_MAX)

            lines.append(f"{i}. {title}")
            if source and time:
                lines.append(f"   {source} · {time}")
            if summary:
                lines.append(f"   {summary}")
            if i < min(_CHAT_MAX, len(raw)):
                lines.append("")

        return "\n".join(lines)

    except Exception:
        return _FAILURE


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CHAT RESPONSE:")
    print("=" * 60)
    print(run_ai_news("latest AI news"))

    print()
    print("=" * 60)
    print("NEWS FEED (structured):")
    print("=" * 60)
    feed = get_ai_news_feed()
    if feed:
        for i, card in enumerate(feed, 1):
            print(f"{i}. [{card['tag']}] {card['title']}")
            print(f"   {card['source']} · {card['time']}")
            print(f"   {card['summary'][:80]}...")
            print()
    else:
        print("No feed results")
