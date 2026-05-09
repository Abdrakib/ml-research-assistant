"""
Model Benchmarks Tool
Fetches REAL benchmark scores for ML models.

Data sources (in priority order):
  1. HuggingFace Model Card metadata — actual eval scores stored in model cards
     (model_index field contains metric names + values e.g. F1: 93.2, Accuracy: 91.5)
  2. DDGS leaderboard search — targeted search for actual score tables
     with strict deduplication and boilerplate filtering

No API key required.
"""

import re
from urllib.parse import urlparse, urlunparse

import requests
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FAILURE = "Sorry I could not fetch model benchmarks right now. Please try again."

_HF_MODELS_URL = "https://huggingface.co/api/models"
_HF_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "ML-Research-Assistant/1.0",
}

_FILLER_WORDS = (
    "benchmark",
    "benchmarks",
    "leaderboard",
    "leaderboards",
    "best model for",
    "top model for",
    "compare model",
    "compare models",
    "model performance",
    "model comparison",
    "state of the art",
    "sota",
    "which model is best",
    "which model is better",
    "model accuracy",
    "model ranking",
    "best performing",
    "highest accuracy",
    "model evaluation",
    "evaluate model",
)

_LEADING_STOPWORDS = re.compile(
    r"^(for|about|on|of|with|the|a|an)\s+", re.IGNORECASE
)

_TASK_MAP = {
    "text classification": "text-classification",
    "sentiment analysis": "text-classification",
    "image classification": "image-classification",
    "object detection": "object-detection",
    "machine translation": "translation",
    "translation": "translation",
    "question answering": "question-answering",
    "summarization": "summarization",
    "text summarization": "summarization",
    "speech recognition": "automatic-speech-recognition",
    "asr": "automatic-speech-recognition",
    "named entity recognition": "token-classification",
    "ner": "token-classification",
    "text generation": "text-generation",
    "language model": "text-generation",
    "image segmentation": "image-segmentation",
    "zero shot classification": "zero-shot-classification",
    "fill mask": "fill-mask",
}

_BOILERPLATE_PHRASES = (
    "if a benchmark already exists",
    "you'll see a link appear",
    "signifying a new leaderboard",
    "you can create a new dataset",
    "to prevent duplication",
    "paper tables with annotated results",
    "submit your results",
    "add a result",
)


# ---------------------------------------------------------------------------
# Query cleaning
# ---------------------------------------------------------------------------

def _clean_query(message: str) -> str:
    """Remove filler words and leftover prepositions."""
    q = message.strip().lower()
    for filler in sorted(_FILLER_WORDS, key=len, reverse=True):
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    q = _LEADING_STOPWORDS.sub("", q).strip()
    return q if q else message.strip()


def _extract_hf_pipeline(query: str) -> str | None:
    """Map natural language to HuggingFace pipeline_tag."""
    q = query.lower()
    for key, value in _TASK_MAP.items():
        if key in q:
            return value
    return None


# ---------------------------------------------------------------------------
# Source 1: HuggingFace Model Card eval results
# ---------------------------------------------------------------------------

def _parse_model_card_metrics(card_data: dict) -> list[dict]:
    """
    Extract actual benchmark scores from HuggingFace model card metadata.

    Model cards store eval results in this structure:
    {
        "model-index": [{
            "results": [{
                "dataset": {"name": "SQuAD"},
                "metrics": [{"name": "F1", "value": 93.2}]
            }]
        }]
    }
    """
    scores = []
    if not card_data or not isinstance(card_data, dict):
        return scores

    model_index = card_data.get("model-index") or card_data.get("model_index") or []
    if not isinstance(model_index, list):
        return scores

    for model_entry in model_index:
        results = model_entry.get("results", [])
        for result in results:
            dataset_info = result.get("dataset", {})
            dataset_name = ""
            if isinstance(dataset_info, dict):
                dataset_name = dataset_info.get("name", "") or dataset_info.get("type", "")
            elif isinstance(dataset_info, str):
                dataset_name = dataset_info

            metrics = result.get("metrics", [])
            for metric in metrics:
                if not isinstance(metric, dict):
                    continue
                name = metric.get("name", "") or metric.get("type", "")
                value = metric.get("value")
                if name and value is not None:
                    if isinstance(value, float):
                        value = round(value, 2)
                        if 0 < value <= 1.0:
                            value = f"{value * 100:.1f}%"
                        else:
                            value = f"{value}"
                    scores.append({
                        "dataset": dataset_name,
                        "metric": name,
                        "value": str(value),
                    })

    return scores[:4]


def _get_hf_benchmarks(query: str) -> str:
    """
    Fetch models with real benchmark scores from HuggingFace model card metadata.
    Layout: downloads/likes/tags first, then benchmark scores below.
    """
    pipeline = _extract_hf_pipeline(query)
    if not pipeline:
        return ""

    try:
        response = requests.get(
            _HF_MODELS_URL,
            params={
                "filter": pipeline,
                "sort": "downloads",
                "direction": -1,
                "limit": 10,
                "full": True,
                "cardData": True,
            },
            headers=_HF_HEADERS,
            timeout=12,
        )
        if response.status_code != 200:
            return ""
        if "application/json" not in response.headers.get("Content-Type", ""):
            return ""

        models = response.json()
        if not models:
            return ""

        lines = [f"Here are benchmark results for '{query}':", ""]
        shown = 0

        for model in models:
            if shown >= 5:
                break

            model_id    = model.get("id", "Unknown")
            downloads   = model.get("downloads", 0)
            likes       = model.get("likes", 0)
            hf_url      = f"https://huggingface.co/{model_id}"

            # Filter tags: remove noise, keep meaningful architecture/language tags
            raw_tags = model.get("tags", [])
            tags = [
                t for t in raw_tags
                if t not in ("pytorch", "transformers", "tf", "jax", pipeline, "gguf")
                and not t.startswith("license:")
                and not t.startswith("dataset:")
                and not t.startswith("arxiv:")
                and not t.startswith("base_model:")
                and not t.startswith("region:")
                and len(t) < 30
            ][:3]

            card_data = model.get("cardData") or {}
            scores = _parse_model_card_metrics(card_data)

            shown += 1
            lines.append(f"{shown}. {model_id}")

            # Context (secondary)
            lines.append(f"   Downloads: {downloads:,}   Likes: {likes:,}")
            if tags:
                lines.append(f"   Tags: {', '.join(tags)}")

            # Benchmark Scores (primary)
            lines.append(f"   {'─' * 20}")
            if scores:
                for s in scores:
                    dataset_label = f" ({s['dataset']})" if s["dataset"] else ""
                    lines.append(f"   {s['metric']}{dataset_label}: {s['value']}")
            else:
                lines.append("   No public benchmark scores in model card")

            lines.append(f"   Link: {hf_url}")
            if shown < 5:
                lines.append("")

        if shown == 0:
            return ""

        lines.append("")
        lines.append(
            "Tip: Models with no scores haven't published eval results publicly. "
            "Click the link → Model Card → Evaluation Results for full details."
        )
        return "\n".join(lines)

    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Source 2: DDGS fallback
# ---------------------------------------------------------------------------

def _base_url(url: str) -> str:
    """Strip query params and fragments for deduplication."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _is_boilerplate(text: str) -> bool:
    """Return True if snippet is Papers With Code UI boilerplate."""
    t = text.lower()
    return any(phrase in t for phrase in _BOILERPLATE_PHRASES)


def _search_ddg_benchmarks(query: str) -> str:
    """
    Search DuckDuckGo for SOTA leaderboard scores.
    Deduplicates by base URL and filters boilerplate.
    """
    try:
        search_query = (
            f"state of the art {query} benchmark accuracy score leaderboard results 2024"
        )

        with DDGS() as ddgs:
            raw = list(ddgs.text(search_query, max_results=10))

        if not raw:
            return _FAILURE

        seen_urls: set[str] = set()
        clean_results = []

        for item in raw:
            url   = item.get("href", "")
            body  = (item.get("body") or "").strip()
            title = (item.get("title") or "").strip()

            base = _base_url(url)
            if base in seen_urls:
                continue
            if _is_boilerplate(body):
                continue
            if not body or not title:
                continue

            seen_urls.add(base)
            clean_results.append(item)

            if len(clean_results) == 4:
                break

        if not clean_results:
            return _FAILURE

        lines = [f"Here are benchmark results for '{query}':", ""]

        for i, item in enumerate(clean_results, start=1):
            title = (item.get("title") or "").strip()
            body  = (item.get("body") or "").strip()
            if len(body) > 300:
                body = body[:300] + "..."
            url = item.get("href", "")

            lines.append(f"{i}. {title}")
            if body:
                lines.append(f"   Info: {body}")
            if url:
                lines.append(f"   Source: {url}")
            if i < len(clean_results):
                lines.append("")

        return "\n".join(lines)

    except Exception:
        return _FAILURE


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_model_benchmarks(message: str) -> str:
    """
    Find SOTA benchmark scores for a given ML task.

    Args:
        message: User message about an ML task benchmark

    Returns:
        Formatted string with real benchmark scores where available
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        hf_result = _get_hf_benchmarks(query)
        if hf_result:
            return hf_result

        return _search_ddg_benchmarks(query)

    except Exception:
        return _FAILURE


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "best model for text classification",
        "state of the art for image classification",
        "benchmark for question answering",
    ]
    for i, test in enumerate(tests, start=1):
        print(f"Query {i}: {test}")
        print(run_model_benchmarks(test))
        if i < len(tests):
            print("\n---\n")
