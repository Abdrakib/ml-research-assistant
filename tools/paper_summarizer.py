"""
Paper Summarizer Tool
Summarizes ML research papers from ArXiv using the LLM.
Fetches paper abstract from ArXiv and summarizes it in simple terms.
No external API key required.
"""

import arxiv

_FAILURE = "Sorry I could not summarize that paper right now. Please try again."
_ABSTRACT_MAX = 2000

_FILLER_WORDS = (
    "summarize paper",
    "summarize the paper",
    "summarize arxiv",
    "summarize this paper",
    "paper summary",
    "give me a summary",
    "give summary",
    "explain paper",
    "explain the paper",
    "explain arxiv",
    "tldr",
    "tl;dr",
    "summarize",
    "summary of",
    "explain this paper",
    "what is this paper about",
    "what does this paper say",
    "break down paper",
    "break down the paper",
    "simplify paper",
)

_SUMMARIZE_PROMPT = """You are an ML research assistant who explains papers clearly.

Given this research paper abstract, provide a simple summary that includes:
1. What problem the paper solves
2. The key idea or method proposed
3. The main results or contributions
4. Who should read this paper

Keep the summary concise and easy to understand for ML engineers.

Paper Abstract:
{abstract}

Summary:"""


def _clean_query(message: str) -> str:
    """Remove filler words to get clean paper title or arxiv ID."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q if q else message.strip()


def _is_arxiv_id(query: str) -> bool:
    """Check if query looks like an ArXiv ID like 2303.08774."""
    import re
    return bool(re.match(r"^\d{4}\.\d{4,5}$", query.strip()))


def run_paper_summarizer(message: str) -> str:
    """
    Summarize an ML research paper from ArXiv.
    Accepts either an ArXiv ID or a paper title/topic.

    Args:
        message: User message containing paper title or ArXiv ID

    Returns:
        Simple summary of the paper
    """
    try:
        from model import generate_response

        query = _clean_query(message)
        if not query:
            return _FAILURE

        client = arxiv.Client()

        # Search by ArXiv ID or by title
        if _is_arxiv_id(query):
            search = arxiv.Search(id_list=[query])
        else:
            search = arxiv.Search(
                query=query,
                max_results=1,
                sort_by=arxiv.SortCriterion.Relevance,
            )

        results = list(client.results(search))

        if not results:
            return f"Sorry I could not find a paper matching '{query}' on ArXiv."

        paper = results[0]
        title = paper.title.strip()
        authors = ", ".join(a.name for a in paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        published = paper.published.strftime("%Y-%m-%d") if paper.published else "Unknown"
        abstract = paper.summary.strip()[:_ABSTRACT_MAX]
        link = paper.entry_id

        # Use LLM to summarize
        prompt = _SUMMARIZE_PROMPT.format(abstract=abstract)
        summary = generate_response(prompt).strip()

        lines = [
            f"📄 Paper: {title}",
            f"   Authors: {authors}",
            f"   Published: {published}",
            f"   Link: {link}",
            "",
            "📝 Summary:",
            summary,
        ]

        return "\n".join(lines)

    except Exception:
        return _FAILURE


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    # Test with a known ArXiv ID (Attention Is All You Need)
    query = "1706.03762"
    cleaned = _clean_query(f"summarize arxiv {query}")
    print(f"Cleaned query: '{cleaned}'")
    print(f"Is ArXiv ID: {_is_arxiv_id(cleaned)}")
    print("To test full summary run the app with model loaded")
