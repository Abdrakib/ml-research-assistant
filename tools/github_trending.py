"""
GitHub Trending ML Repositories Tool
Finds trending Machine Learning repositories on GitHub.
Uses DuckDuckGo to find trending repos. No API key required.
"""

from ddgs import DDGS

_FAILURE = "Sorry I could not fetch trending ML repositories right now. Please try again."

_FILLER_WORDS = (
    "github trending",
    "trending github",
    "trending ml",
    "trending ai",
    "trending machine learning",
    "trending deep learning",
    "trending repos",
    "trending repositories",
    "popular repos",
    "popular repositories",
    "latest ml repos",
    "latest ai repos",
    "new ml repos",
    "hot repos",
    "find trending",
    "show trending",
    "what is trending",
    "whats trending",
)

_SEARCH_QUERIES = [
    "trending machine learning github repositories this week",
    "popular new AI github repos 2025",
    "trending deep learning pytorch huggingface github",
]


def _clean_query(message: str) -> str:
    """Remove filler words to get clean search query."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q


def _is_github_repo_url(url: str) -> bool:
    """Check if URL is a GitHub repository page."""
    if "github.com" not in url:
        return False
    parts = url.replace("https://", "").replace("http://", "").split("/")
    # Valid repo URL has format: github.com/owner/repo
    return len(parts) >= 3 and parts[0] == "github.com" and parts[1] and parts[2]


def run_github_trending(message: str) -> str:
    """
    Find trending ML repositories on GitHub.

    Args:
        message: User message about trending repos

    Returns:
        Formatted string with trending ML GitHub repositories
    """
    try:
        extra = _clean_query(message)
        results = []
        seen_urls = set()

        with DDGS() as ddgs:
            for query in _SEARCH_QUERIES:
                # Add user specified topic if any
                if extra:
                    query = f"{extra} {query}"
                chunks = list(ddgs.text(query, max_results=8))
                for item in chunks:
                    url = item.get("href", "")
                    if _is_github_repo_url(url) and url not in seen_urls:
                        seen_urls.add(url)
                        results.append(item)
                if len(results) >= 5:
                    break

        if not results:
            return _FAILURE

        lines = ["Here are trending ML repositories on GitHub:", ""]

        for i, item in enumerate(results[:5], start=1):
            title = (item.get("title") or "").strip()
            body = (item.get("body") or "").strip()
            url = item.get("href", "")

            # Clean up title - remove " - GitHub" suffix
            title = title.replace(" - GitHub", "").strip()

            # Extract owner/repo from URL
            parts = url.replace("https://github.com/", "").split("/")
            repo_name = f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else title

            description = body[:200] if body else "No description available"

            lines.append(f"{i}. {repo_name}")
            lines.append(f"   Description: {description}")
            lines.append(f"   Link: {url}")
            if i < min(5, len(results)):
                lines.append("")

        return "\n".join(lines)

    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print(run_github_trending("trending ml repos"))
    print("\n---\n")
    print(run_github_trending("trending pytorch repos"))
