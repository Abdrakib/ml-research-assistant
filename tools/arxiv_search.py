"""
ArXiv Search Tool
Searches for latest ML and AI research papers on ArXiv.
Uses the arxiv Python library. No API key required.
"""

import arxiv

_FAILURE = "Sorry I could not find any papers right now. Please try again."
_ABSTRACT_MAX = 300


def _truncate(text: str, max_len: int = _ABSTRACT_MAX) -> str:
    """Truncate text to max_len characters."""
    t = text.strip().replace("\n", " ")
    return (t[:max_len] + "...") if len(t) > max_len else t


def _clean_query(message: str) -> str:
    """Remove filler words from user message to get clean search query."""
    fillers = (
        "find paper",
        "find papers",
        "search paper",
        "search papers",
        "latest paper",
        "latest papers",
        "research paper",
        "research papers",
        "paper about",
        "paper on",
        "papers about",
        "papers on",
        "find research",
        "find study",
        "arxiv",
        "recent paper",
        "recent papers",
        "published paper",
        "published papers",
        "study on",
        "ml paper",
        "ai paper",
        "deep learning paper",
        "machine learning paper",
    )
    q = message.strip().lower()
    for filler in fillers:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q if q else message.strip()


def run_arxiv_search(message: str) -> str:
    """
    Search ArXiv for research papers matching the user query.

    Args:
        message: User message containing the search topic

    Returns:
        Formatted string with paper titles, authors, abstracts and links
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = list(client.results(search))

        if not results:
            return f"Sorry I could not find any papers about '{query}' on ArXiv."

        lines = [f"Here are the latest ArXiv papers on '{query}':", ""]

        for i, paper in enumerate(results, start=1):
            title = paper.title.strip()
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            abstract = _truncate(paper.summary)
            published = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown"
            link = paper.entry_id

            lines.append(f"{i}. {title}")
            lines.append(f"   Authors: {authors}")
            lines.append(f"   Published: {published}")
            lines.append(f"   Abstract: {abstract}")
            lines.append(f"   Link: {link}")
            if i < len(results):
                lines.append("")

        return "\n".join(lines)

    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print(run_arxiv_search("find papers about retrieval augmented generation"))
