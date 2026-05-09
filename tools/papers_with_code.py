"""
Papers With Code Tool
Step 1: Search ArXiv for papers (reliable, free API)
Step 2: Search GitHub for implementations using paper title
No API key required for either.
"""

import re
import time
import arxiv
import requests

_FAILURE = "Sorry I could not search for papers right now. Please try again."
_GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"
_GITHUB_HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "ML-Research-Assistant/1.0",
}

_FILLER_WORDS = (
    "find paper with code",
    "find papers with code",
    "search paper with code",
    "paper with code",
    "papers with code",
    "paper implementation",
    "paper code",
    "find implementation",
    "code implementation",
    "github implementation",
    "find code for paper",
    "paper and code",
    "find paper",
    "search paper",
    "find papers",
    "search papers",
)

_LEADING_STOPWORDS = re.compile(
    r"^(for|about|on|of|with|the|a|an)\s+", re.IGNORECASE
)

# ArXiv requires at least 15 seconds between requests
_last_arxiv_call: float = 0.0
_ARXIV_MIN_INTERVAL = 15.0


def _clean_query(message: str) -> str:
    """Remove filler words and leftover stopwords to get a clean search query."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    q = _LEADING_STOPWORDS.sub("", q).strip()
    return q if q else message.strip()


def _respect_rate_limit() -> None:
    """Enforce ArXiv's rate limit — max one request per 15 seconds."""
    global _last_arxiv_call
    elapsed = time.time() - _last_arxiv_call
    if elapsed < _ARXIV_MIN_INTERVAL:
        time.sleep(_ARXIV_MIN_INTERVAL - elapsed)
    _last_arxiv_call = time.time()


def _get_github_repos(paper_title: str) -> list[str]:
    """
    Search GitHub for repositories implementing this paper.
    Uses the paper title + 'implementation' as the query.
    Returns formatted strings like 'url ⭐stars'.
    Gracefully returns empty list on any failure.
    """
    try:
        # Build a focused GitHub search query from the paper title
        # Keep only first 6 words to avoid over-specific queries
        short_title = " ".join(paper_title.split()[:6])
        github_query = f"{short_title} implementation"

        response = requests.get(
            _GITHUB_SEARCH_URL,
            params={
                "q": github_query,
                "sort": "stars",
                "order": "desc",
                "per_page": 2,
            },
            headers=_GITHUB_HEADERS,
            timeout=8,
        )
 
        if response.status_code != 200:
            return []

        items = response.json().get("items", [])
        repo_urls = []
        for repo in items[:2]:
            url = repo.get("html_url", "")
            stars = repo.get("stargazers_count", 0)
            if url:
                repo_urls.append(f"{url} ⭐{stars}")

        return repo_urls

    except Exception:
        return []


def run_papers_with_code(message: str) -> str:
    """
    Search for ML papers with GitHub implementations.

    Step 1 — ArXiv:  search by topic → title, abstract, arxiv_id
    Step 2 — GitHub: search by paper title → repo links + stars

    Args:
        message: User message containing the search topic

    Returns:
        Formatted string with paper titles, abstracts, links, and GitHub repos
    """
    query = _clean_query(message)
    if not query:
        return _FAILURE

    try:
        _respect_rate_limit()

        client = arxiv.Client(page_size=3, num_retries=2, delay_seconds=15)
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        arxiv_results = list(client.results(search))

    except arxiv.HTTPError as e:
        if "429" in str(e):
            return "ArXiv is rate limiting requests right now. Please wait a moment and try again."
        return _FAILURE

    except Exception:
        return _FAILURE

    if not arxiv_results:
        return f"Sorry, I could not find any papers for '{query}'."

    lines = [f"Here are papers with code for '{query}':", ""]

    for i, paper in enumerate(arxiv_results, start=1):
        title = paper.title.strip()

        abstract = paper.summary.strip().replace("\n", " ")
        if len(abstract) > 250:
            abstract = abstract[:250] + "..."

        arxiv_id = paper.entry_id.split("/")[-1]
        clean_id = arxiv_id.split("v")[0]

        paper_url = f"https://arxiv.org/abs/{clean_id}"
        pdf_url = f"https://arxiv.org/pdf/{clean_id}"

        authors = [a.name for a in paper.authors[:3]]
        if len(paper.authors) > 3:
            authors.append("et al.")
        authors_str = ", ".join(authors)
        published = paper.published.strftime("%b %Y") if paper.published else "Unknown"

        # GitHub search using paper title
        repo_urls = _get_github_repos(title)

        lines.append(f"{i}. {title}")
        lines.append(f"   Authors: {authors_str} ({published})")
        if abstract:
            lines.append(f"   Abstract: {abstract}")
        lines.append(f"   Paper: {paper_url}")
        lines.append(f"   PDF:   {pdf_url}")
        lines.append(f"   Code:  {' | '.join(repo_urls) if repo_urls else 'No implementation found yet'}")
        if i < len(arxiv_results):
            lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Query 1: transformer attention")
    print(run_papers_with_code("find papers with code for transformer attention"))

    print("\n---\n")

    print("Query 2: BERT")
    print(run_papers_with_code("paper implementation for BERT"))
