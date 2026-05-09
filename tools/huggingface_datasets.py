"""
HuggingFace Datasets Search Tool
Searches for ML datasets on HuggingFace Hub.
Uses HuggingFace public REST API. No API key required.
"""

import requests

_FAILURE = "Sorry I could not search HuggingFace datasets right now. Please try again."
_HF_DATASETS_URL = "https://huggingface.co/api/datasets"

_FILLER_WORDS = (
    "find dataset",
    "find datasets",
    "search dataset",
    "search datasets",
    "dataset for",
    "datasets for",
    "give me dataset",
    "give me datasets",
    "show dataset",
    "show datasets",
    "huggingface dataset",
    "huggingface datasets",
    "training data for",
    "data for training",
    "training dataset",
    "benchmark dataset",
    "public dataset",
    "find data",
    "search data",
)


def _clean_query(message: str) -> str:
    """Remove filler words to get clean search query."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q if q else message.strip()


def _format_number(n: int) -> str:
    """Format large numbers nicely. 1000000 -> 1M"""
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def run_huggingface_datasets(message: str) -> str:
    """
    Search HuggingFace for datasets matching the user query.

    Args:
        message: User message containing the search topic

    Returns:
        Formatted string with dataset names, downloads and descriptions
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        response = requests.get(
            _HF_DATASETS_URL,
            params={
                "search": query,
                "limit": 5,
                "sort": "downloads",
                "direction": -1,
            },
            timeout=10,
        )
        response.raise_for_status()
        datasets = response.json()

        if not datasets:
            return f"Sorry I could not find any datasets for '{query}' on HuggingFace."

        lines = [f"Here are the top HuggingFace datasets for '{query}':", ""]

        for i, dataset in enumerate(datasets[:5], start=1):
            name = dataset.get("id", "Unknown")
            downloads = _format_number(dataset.get("downloads", 0))
            likes = dataset.get("likes", 0)
            task_categories = dataset.get("task_categories", [])
            task = ", ".join(task_categories) if task_categories else "Not specified"
            link = f"https://huggingface.co/datasets/{name}"

            lines.append(f"{i}. {name}")
            lines.append(f"   Task: {task}")
            lines.append(f"   Downloads: {downloads}")
            lines.append(f"   Likes: {likes}")
            lines.append(f"   Link: {link}")
            if i < min(5, len(datasets)):
                lines.append("")

        return "\n".join(lines)

    except requests.RequestException:
        return _FAILURE
    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print(run_huggingface_datasets("find dataset for sentiment analysis"))
    print("\n---\n")
    print(run_huggingface_datasets("dataset for image classification"))
