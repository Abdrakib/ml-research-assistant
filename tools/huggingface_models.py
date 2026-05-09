"""
HuggingFace Models Search Tool
Searches for ML models on HuggingFace Hub.
Uses HuggingFace public REST API. No API key required.
"""

import requests

_FAILURE = "Sorry I could not search HuggingFace models right now. Please try again."
_HF_MODELS_URL = "https://huggingface.co/api/models"

_FILLER_WORDS = (
    "find model",
    "find models",
    "search model",
    "search models",
    "best model for",
    "top model for",
    "show model",
    "show models",
    "give me model",
    "give me models",
    "huggingface model",
    "huggingface models",
    "model for",
    "models for",
    "what model",
    "which model",
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


def run_huggingface_models(message: str) -> str:
    """
    Search HuggingFace for ML models matching the user query.

    Args:
        message: User message containing the search topic

    Returns:
        Formatted string with model names, downloads and descriptions
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        response = requests.get(
            _HF_MODELS_URL,
            params={
                "search": query,
                "limit": 5,
                "sort": "downloads",
                "direction": -1,
            },
            timeout=10,
        )
        response.raise_for_status()
        models = response.json()

        if not models:
            return f"Sorry I could not find any models for '{query}' on HuggingFace."

        lines = [f"Here are the top HuggingFace models for '{query}':", ""]

        for i, model in enumerate(models[:5], start=1):
            name = model.get("id", "Unknown")
            downloads = _format_number(model.get("downloads", 0))
            likes = model.get("likes", 0)
            pipeline = model.get("pipeline_tag", "Not specified")
            link = f"https://huggingface.co/{name}"

            lines.append(f"{i}. {name}")
            lines.append(f"   Task: {pipeline}")
            lines.append(f"   Downloads: {downloads}")
            lines.append(f"   Likes: {likes}")
            lines.append(f"   Link: {link}")
            if i < min(5, len(models)):
                lines.append("")

        return "\n".join(lines)

    except requests.RequestException:
        return _FAILURE
    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print(run_huggingface_models("find models for text classification"))
    print("\n---\n")
    print(run_huggingface_models("best model for image classification"))
