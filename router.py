import re
from typing import Any, Dict

from thefuzz import fuzz

# ── Existing tool keywords ────────────────────────────────────────────────────

_WEATHER_KEYS = (
    "weather",
    "temperature",
    "forecast",
    "wear",
    "cold",
    "hot",
    "raining",
    "sunny",
    "umbrella",
    "coat",
    "jacket",
    "outside",
    "humid",
    "wind",
)

_DEEP_SEARCH_KEYS = (
    "deep search",
    "explain in detail",
    "comprehensive",
    "everything about",
    "full explanation",
    "in depth",
    "deep dive",
)

_SEARCH_KEYS = (
    "search",
    "look up",
    "who is",
    "what is",
    "tell me about",
    "current events",
)

_CALC_WORD_KEYS = (
    "calculate",
    "plus",
    "minus",
    "times",
    "divided",
    "multiply",
    "percent",
    "square root",
)

_MEMORY_KEYS = (
    "remember",
    "my name is",
    "i live",
    "i like",
    "i work",
    "what do you know",
    "do you remember",
    "what is my name",
    "what is my favorite",
    "what is my age",
    "what is my job",
    "what is my car",
    "what do you know about me",
    "my favorite",
    "i love",
    "i hate",
    "i prefer",
    "i enjoy",
    "my car is",
    "my age is",
    "my job is",
    "my major is",
    "i study",
    "i was born",
    "my birthday is",
    "my dog is",
    "my cat is",
    "my phone is",
    "my email is",
)

_GITHUB_KEYS = (
    "my repos",
    "my repositories",
    "my profile",
    "my code",
    "my github",
    "github repos",
    "github profile",
)

# ── New ML Specialized tool keywords ─────────────────────────────────────────

_ARXIV_KEYS = (
    "arxiv",
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
    "recent paper",
    "recent papers",
    "published paper",
    "ml paper",
    "ai paper",
    "deep learning paper",
    "machine learning paper",
    "find study",
    "study on",
)

_HF_MODEL_KEYS = (
    "find model",
    "find models",
    "search model",
    "search models",
    "show model",
    "show models",
    "give me model",
    "give me models",
    "huggingface model",
    "huggingface models",
    "model for",
    "models for",
)

_HF_DATASET_KEYS = (
    "find dataset",
    "find datasets",
    "search dataset",
    "search datasets",
    "dataset for",
    "datasets for",
    "give me dataset",
    "give me datasets",
    "huggingface dataset",
    "training data for",
    "data for training",
    "training dataset",
    "benchmark dataset",
    "public dataset",
)

_CODE_KEYS = (
    "write code",
    "generate code",
    "implement",
    "fine-tune",
    "finetune",
    "train a model",
    "build a model",
    "create a function",
    "write a function",
    "write a script",
    "code for",
    "python code",
    "pytorch code",
    "tensorflow code",
    "keras code",
    "numpy code",
    "sklearn code",
    "show me the code",
    "write me a",
    "give me code",
    "how to code",
)

_AI_NEWS_KEYS = (
    "ai news",
    "latest ai",
    "what happened in ai",
    "ml news",
    "artificial intelligence news",
    "deep learning news",
    "openai news",
    "llm news",
    "chatgpt news",
    "what is new in ai",
    "recent ai",
    "ai this week",
    "ai today",
    "machine learning news",
)

_PWC_KEYS = (
    "paper with code",
    "papers with code",
    "paper implementation",
    "find implementation",
    "code implementation",
    "github implementation",
    "find code for paper",
    "paper and code",
)

_PYPI_KEYS = (
    "python package",
    "python packages",
    "python library",
    "python libraries",
    "pip package",
    "pip install",
    "find library",
    "find libraries",
    "package for",
    "library for",
    "packages for",
    "libraries for",
    "best package",
    "best library",
    "find package",
    "find packages",
)

_PAPER_SUMMARIZER_KEYS = (
    "summarize paper",
    "summarize the paper",
    "summarize arxiv",
    "paper summary",
    "give me a summary of",
    "explain paper",
    "explain the paper",
    "tldr",
    "tl;dr",
    "summary of paper",
    "explain this paper",
    "what does this paper say",
    "break down paper",
    "simplify paper",
)

_GITHUB_TRENDING_KEYS = (
    "github trending",
    "trending github",
    "trending ml repos",
    "trending ai repos",
    "trending machine learning",
    "trending repos",
    "trending repositories",
    "popular repos",
    "popular ml repos",
    "latest ml repos",
    "hot repos",
    "what is trending on github",
)

_BENCHMARK_KEYS = (
    "benchmark",
    "leaderboard",
    "best model for",
    "top model for",
    "compare models",
    "model performance",
    "state of the art",
    "sota",
    "which model is best",
    "which model is better",
    "model accuracy",
    "model ranking",
    "best performing model",
    "highest accuracy model",
    "model comparison",
)


_LLM_LEADERBOARD_KEYS = (
    # Direct comparison phrases
    "compare llm", "llm comparison", "llm vs", "vs llm",
    "compare gpt", "compare claude", "compare llama", "compare mistral",
    "compare gemma", "compare qwen", "compare phi", "compare deepseek",
    # X vs Y model name patterns
    "gpt vs", "gpt-4 vs", "gpt-4o vs", "gpt4 vs",
    "claude vs", "claude 3", "claude 3.5",
    "llama vs", "llama 3", "llama 2",
    "mistral vs", "mixtral vs", "gemma vs", "gemma 2",
    "phi vs", "phi-3", "deepseek vs",
    "vs gpt", "vs claude", "vs llama", "vs mistral",
    "vs gemma", "vs phi", "vs deepseek", "vs mixtral",
    # LLM leaderboard phrases
    "llm leaderboard", "llm ranking", "llm benchmark",
    "best llm", "top llm", "which llm", "llm score",
    "open llm leaderboard",
    # Standard LLM benchmark names
    "mmlu", "gsm8k", "humaneval", "hellaswag",
    # Performance questions about specific LLMs
    "how does gpt", "how does claude", "how does llama",
    "how does mistral", "how does gemma",
)


# ── Helper functions ──────────────────────────────────────────────────────────

def _contains_any(haystack: str, needles: tuple) -> bool:
    low = haystack.lower()
    return any(n in low for n in needles)


def _fuzzy_contains(message: str, keywords: tuple, threshold: int = 80) -> bool:
    words = message.lower().split()
    for word in words:
        for keyword in keywords:
            if len(keyword.split()) == 1:
                if fuzz.ratio(word, keyword) >= threshold:
                    return True
            else:
                if fuzz.partial_ratio(message.lower(), keyword) >= threshold:
                    return True
    return False


def _is_calc(message: str) -> bool:
    """Calculator intent: math keywords or digit+operator+digit patterns.

    Requires the operator to sit between two digits so that hyphens in
    model names (gpt-4o, llama-3, phi-3) and version numbers (3.5) never
    trigger the calculator by accident.
    """
    if _contains_any(message, _CALC_WORD_KEYS):
        return True
    # Operator must be flanked by digits — excludes model-name hyphens
    if re.search(r"\d\s*[+\-*/×÷]\s*\d", message):
        return True
    low = message.lower()
    if "what is" in low and re.search(r"\d+\s*[+\-*/×÷]\s*\d", message):
        return True
    return False


# ── Main detection function ───────────────────────────────────────────────────

def _detect_tool(message: str) -> str:
    """
    Check message against all keyword lists in priority order.
    First match wins.
    """
    # ── General tools ──────────────────────────────────────────────────────
    if _contains_any(message, _WEATHER_KEYS) or _fuzzy_contains(message, _WEATHER_KEYS, threshold=80):
        return "weather"

    if _contains_any(message, _MEMORY_KEYS) or _fuzzy_contains(message, _MEMORY_KEYS, threshold=90):
        return "memory"

    # github_trending BEFORE github — 'trending ml repos on github' has 'repos'
    # which matches _GITHUB_KEYS if checked first
    if _contains_any(message, _GITHUB_TRENDING_KEYS) or _fuzzy_contains(message, _GITHUB_TRENDING_KEYS, threshold=92):
        return "github_trending"

    if _contains_any(message, _GITHUB_KEYS) or _fuzzy_contains(message, _GITHUB_KEYS, threshold=85):
        return "github"

    if _is_calc(message):
        return "calc"

    # ── ML Specialized tools (checked before general search) ───────────────
    # llm_leaderboard BEFORE model_benchmarks — more specific keywords
    if _contains_any(message, _LLM_LEADERBOARD_KEYS) or _fuzzy_contains(message, _LLM_LEADERBOARD_KEYS, threshold=90):
        return "llm_leaderboard"

    if _contains_any(message, _BENCHMARK_KEYS) or _fuzzy_contains(message, _BENCHMARK_KEYS, threshold=85):
        return "model_benchmarks"

    if _contains_any(message, _PAPER_SUMMARIZER_KEYS) or _fuzzy_contains(message, _PAPER_SUMMARIZER_KEYS, threshold=85):
        return "paper_summarizer"

    if _contains_any(message, _PWC_KEYS) or _fuzzy_contains(message, _PWC_KEYS, threshold=85):
        return "papers_with_code"

    if _contains_any(message, _ARXIV_KEYS) or _fuzzy_contains(message, _ARXIV_KEYS, threshold=85):
        return "arxiv"

    if _contains_any(message, _HF_MODEL_KEYS) or _fuzzy_contains(message, _HF_MODEL_KEYS, threshold=85):
        return "huggingface_models"

    if _contains_any(message, _HF_DATASET_KEYS) or _fuzzy_contains(message, _HF_DATASET_KEYS, threshold=85):
        return "huggingface_datasets"

    if _contains_any(message, _CODE_KEYS) or _fuzzy_contains(message, _CODE_KEYS, threshold=85):
        return "code_generator"

    if _contains_any(message, _AI_NEWS_KEYS) or _fuzzy_contains(message, _AI_NEWS_KEYS, threshold=85):
        return "ai_news"

    if _contains_any(message, _PYPI_KEYS) or _fuzzy_contains(message, _PYPI_KEYS, threshold=85):
        return "python_packages"

    if _contains_any(message, _GITHUB_TRENDING_KEYS) or _fuzzy_contains(message, _GITHUB_TRENDING_KEYS, threshold=92):
        return "github_trending"

    # ── General search and deep search (last resort) ────────────────────────
    if _contains_any(message, _DEEP_SEARCH_KEYS) or _fuzzy_contains(message, _DEEP_SEARCH_KEYS, threshold=85):
        return "deep_search"

    if _contains_any(message, _SEARCH_KEYS) or _fuzzy_contains(message, _SEARCH_KEYS, threshold=85):
        return "search"

    return "none"


def route(message: str, active_tools: Dict[str, Any]) -> Dict[str, Any]:
    tool = _detect_tool(message)
    if tool == "none":
        return {"tool": "none", "auto_enabled": False}

    was_off = not bool(active_tools.get(tool, False))
    if was_off:
        active_tools[tool] = True

    return {"tool": tool, "auto_enabled": was_off}


if __name__ == "__main__":
    _all_on = {
        "search": True,
        "deep_search": True,
        "weather": True,
        "calc": True,
        "memory": True,
        "github": True,
        "arxiv": True,
        "huggingface_models": True,
        "huggingface_datasets": True,
        "code_generator": True,
        "ai_news": True,
        "papers_with_code": True,
        "python_packages": True,
        "paper_summarizer": True,
        "github_trending": True,
        "model_benchmarks": True,
        "llm_leaderboard": True,
    }

    _msgs = (
        # Existing tools
        "what is the weather in Philadelphia",
        "what is 25 times 4",
        "my name is Rakib",
        "show my github repos",
        "hello how are you",
        "deep dive into neural networks",
        # New ML tools
        "find papers about RAG",
        "find models for text classification",
        "find dataset for sentiment analysis",
        "write pytorch code for training loop",
        "latest AI news today",
        "papers with code for BERT",
        "find python package for data augmentation",
        "summarize arxiv paper 1706.03762",
        "trending ml repos on github",
        "best model for text classification benchmark",
        "state of the art for image classification",
        # Edge cases
        "wheather in philadelphia",
        "serach for AI news",
        "what is my favorite food",
        "what is transfer learning",
    )

    for m in _msgs:
        tools = dict(_all_on)
        print(repr(m), "->", route(m, tools))
