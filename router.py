import re
from typing import Any, Dict

from thefuzz import fuzz

# ── General tool keywords ─────────────────────────────────────────────────────

_WEATHER_KEYS = (
    "weather", "temperature", "forecast", "umbrella", "humid", "humidity",
    # Common misspellings handled by exact match (no fuzzy needed)
    "wheather", "wether", "waether", "forcast", "tempature", "tempurature",
)

# Short weather words that collide as substrings:
# 'raining' inside 'training', 'hot' inside 'short', 'wind' inside 'window'
# Matched with word boundaries only via _is_weather()
_WEATHER_WORD_KEYS = (
    "raining", "sunny", "cold", "hot", "wind", "wear",
    "coat", "jacket", "outside",
)

_DEEP_SEARCH_KEYS = (
    "deep search", "explain in detail", "comprehensive",
    "everything about", "full explanation", "in depth", "deep dive",
)

_SEARCH_KEYS = (
    "search", "look up", "who is", "what is",
    "tell me about", "current events",
)

_CALC_WORD_KEYS = (
    "calculate", "plus", "minus", "times", "divided",
    "multiply", "percent", "square root",
)

_MEMORY_KEYS = (
    "remember", "my name is", "i live", "i like", "i work",
    "what do you know", "do you remember", "what is my name",
    "what is my favorite", "what is my age", "what is my job",
    "what is my car", "what do you know about me", "my favorite",
    "i love", "i hate", "i prefer", "i enjoy", "my car is",
    "my age is", "my job is", "my major is", "i study", "i was born",
    "my birthday is", "my dog is", "my cat is", "my phone is",
    "my email is",
)

_GITHUB_KEYS = (
    "my repos", "my repositories", "my profile", "my code",
)

# ── ML specialized tool keywords ─────────────────────────────────────────────

_ARXIV_KEYS = (
    "arxiv", "find paper", "find papers", "search paper", "search papers",
    "latest paper", "latest papers", "research paper", "research papers",
    "paper about", "paper on", "papers about", "papers on",
    "recent paper", "recent papers", "published paper",
    "ml paper", "ai paper", "deep learning paper", "machine learning paper",
    "find study", "study on",
)

_HF_MODEL_KEYS = (
    "find model", "find models", "search model", "search models",
    "show model", "show models", "give me model", "give me models",
    "huggingface model", "huggingface models", "model for", "models for",
)

_HF_DATASET_KEYS = (
    "find dataset", "find datasets", "search dataset", "search datasets",
    "dataset for", "datasets for", "give me dataset", "give me datasets",
    "huggingface dataset", "training data for", "data for training",
    "training dataset", "benchmark dataset", "public dataset",
)

_CODE_KEYS = (
    "write code", "generate code", "implement", "fine-tune", "finetune",
    "train a model", "build a model", "create a function",
    "write a function", "write a script", "code for", "python code",
    "pytorch code", "tensorflow code", "keras code", "numpy code",
    "sklearn code", "show me the code", "write me a", "give me code",
    "how to code",
)

_AI_NEWS_KEYS = (
    "ai news", "latest ai", "what happened in ai", "ml news",
    "artificial intelligence news", "deep learning news", "openai news",
    "llm news", "chatgpt news", "what is new in ai", "recent ai",
    "ai this week", "ai today", "machine learning news",
)

_PWC_KEYS = (
    "paper with code", "papers with code", "paper implementation",
    "find implementation", "code implementation", "github implementation",
    "find code for paper", "paper and code",
)

_PYPI_KEYS = (
    "python package", "python packages", "python library", "python libraries",
    "pip package", "pip install", "find library", "find libraries",
    "package for", "library for", "packages for", "libraries for",
    "best package", "best library", "find package", "find packages",
)

_PAPER_SUMMARIZER_KEYS = (
    "summarize paper", "summarize the paper", "summarize arxiv",
    "paper summary", "give me a summary of", "explain paper",
    "explain the paper", "tldr", "tl;dr", "summary of paper",
    "explain this paper", "what does this paper say",
    "break down paper", "simplify paper",
)

_GITHUB_TRENDING_KEYS = (
    "github trending", "trending github", "trending ml repos",
    "trending ai repos", "trending machine learning", "trending repos",
    "trending repositories", "popular repos", "popular ml repos",
    "latest ml repos", "hot repos", "what is trending on github",
)

# ── LLM Leaderboard keywords ──────────────────────────────────────────────────
# Catches queries about comparing specific large language models by name
# and standard LLM benchmark names (MMLU, GSM8K, HumanEval, HellaSwag).
# Must be checked BEFORE _BENCHMARK_KEYS to avoid model comparison queries
# being swallowed by the more generic benchmark tool.

_LLM_LEADERBOARD_KEYS = (
    # Direct LLM comparison phrases
    "compare llm", "llm comparison", "llm vs", "vs llm",
    "compare gpt", "compare claude", "compare llama", "compare mistral",
    "compare gemma", "compare qwen", "compare phi", "compare deepseek",
    # "X vs Y" patterns for named LLMs
    "gpt vs", "gpt-4 vs", "gpt-4o vs", "gpt4 vs",
    "claude vs", "claude 3", "claude 3.5",
    "llama vs", "llama 3", "llama 2",
    "mistral vs", "mixtral vs",
    "gemma vs", "gemma 2",
    "phi vs", "phi-3",
    "deepseek vs",
    "vs gpt", "vs claude", "vs llama", "vs mistral",
    "vs gemma", "vs phi", "vs deepseek", "vs mixtral",
    # Leaderboard / ranking phrases specific to LLMs
    "llm leaderboard", "llm ranking", "llm benchmark",
    "best llm", "top llm", "which llm", "llm score",
    "open llm leaderboard",
    # Standard LLM benchmark names — very specific, no ambiguity
    "mmlu", "gsm8k", "humaneval", "hellaswag",
    # General LLM performance questions
    "how does gpt", "how does claude", "how does llama",
    "how does mistral", "how does gemma",
)

# ── Model Benchmarks keywords ─────────────────────────────────────────────────
# Catches task-specific benchmark queries (text classification, QA, CV etc.)
# NOT for LLM-vs-LLM comparisons — those go to llm_leaderboard above.
# Removed 'leaderboard' and 'compare models' to avoid stealing llm queries.

_BENCHMARK_KEYS = (
    "benchmark for", "benchmarks for",
    "best model for", "top model for",
    "model performance on", "state of the art for",
    "sota for", "which model is best for",
    "which model is better for", "model accuracy on",
    "model ranking for", "best performing model for",
    "highest accuracy model for", "model comparison for",
    # Task-specific benchmark triggers
    "text classification benchmark", "image classification benchmark",
    "object detection benchmark", "question answering benchmark",
    "translation benchmark", "summarization benchmark",
    "speech recognition benchmark", "ner benchmark",
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


def _is_weather(message: str) -> bool:
    """Word-boundary match for short weather words to avoid substring collisions."""
    msg = message.lower()
    for kw in _WEATHER_WORD_KEYS:
        if re.search(r"\b" + re.escape(kw) + r"\b", msg):
            return True
    return False


def _is_calc(message: str) -> bool:
    """Calculator intent: math keywords or digit+operator patterns.

    Requires operator to sit between digits (e.g. '3 + 4', '25*4') so that
    hyphens in model names like 'gpt-4o' or 'phi-3' never trigger calc.
    """
    if _contains_any(message, _CALC_WORD_KEYS):
        return True
    # Operator must be flanked by digits — excludes 'gpt-4', 'llama-3' etc.
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

    Priority rationale:
      1. General tools (weather, memory, github, calc) — high precision,
         unlikely to collide with ML queries.
      2. llm_leaderboard — checked BEFORE model_benchmarks because it has
         more specific keywords (model names, MMLU/GSM8K etc.) that would
         otherwise be caught by the broader benchmark keywords.
      3. ML specialized tools ordered by specificity (most specific first).
      4. General search / deep search — last resort fallback.
    """
    # ── 1. General tools ───────────────────────────────────────────────────
    # Weather: exact match only (fuzzy disabled — 'training'~'raining'=93 false positive).
    # Long keys use substring match; short ambiguous keys use word boundaries.
    if _contains_any(message, _WEATHER_KEYS) or _is_weather(message):
        return "weather"

    if _contains_any(message, _MEMORY_KEYS) or _fuzzy_contains(message, _MEMORY_KEYS, threshold=90):
        return "memory"

    # github_trending BEFORE github — 'trending ml repos on github' contains
    # 'repos' which is in _GITHUB_KEYS; check trending first (more specific)
    if _contains_any(message, _GITHUB_TRENDING_KEYS) or _fuzzy_contains(message, _GITHUB_TRENDING_KEYS, threshold=85):
        return "github_trending"

    if _contains_any(message, _GITHUB_KEYS) or _fuzzy_contains(message, _GITHUB_KEYS, threshold=85):
        return "github"

    if _is_calc(message):
        return "calc"

    # ── 2. ML specialized tools ────────────────────────────────────────────

    # llm_leaderboard BEFORE model_benchmarks — more specific keywords.
    # Fuzzy threshold raised to 90 to prevent 'best model'~'best llm'=86
    # and 'classification benchmark'~'llm benchmark'=87 false positives.
    if _contains_any(message, _LLM_LEADERBOARD_KEYS) or _fuzzy_contains(message, _LLM_LEADERBOARD_KEYS, threshold=90):
        return "llm_leaderboard"

    if _contains_any(message, _BENCHMARK_KEYS) or _fuzzy_contains(message, _BENCHMARK_KEYS, threshold=85):
        return "model_benchmarks"

    if _contains_any(message, _PAPER_SUMMARIZER_KEYS) or _fuzzy_contains(message, _PAPER_SUMMARIZER_KEYS, threshold=85):
        return "paper_summarizer"

    # papers_with_code BEFORE arxiv — both match paper queries but pwc is more specific
    if _contains_any(message, _PWC_KEYS) or _fuzzy_contains(message, _PWC_KEYS, threshold=85):
        return "papers_with_code"

    if _contains_any(message, _ARXIV_KEYS) or _fuzzy_contains(message, _ARXIV_KEYS, threshold=85):
        return "arxiv"

    if _contains_any(message, _HF_MODEL_KEYS) or _fuzzy_contains(message, _HF_MODEL_KEYS, threshold=85):
        return "huggingface_models"

    if _contains_any(message, _HF_DATASET_KEYS) or _fuzzy_contains(message, _HF_DATASET_KEYS, threshold=85):
        return "huggingface_datasets"

    if _contains_any(message, _AI_NEWS_KEYS) or _fuzzy_contains(message, _AI_NEWS_KEYS, threshold=85):
        return "ai_news"

    if _contains_any(message, _PYPI_KEYS) or _fuzzy_contains(message, _PYPI_KEYS, threshold=85):
        return "python_packages"

    # code_generator last among ML tools — 'implement' is a broad keyword
    if _contains_any(message, _CODE_KEYS) or _fuzzy_contains(message, _CODE_KEYS, threshold=85):
        return "code_generator"

    # ── 3. General search (last resort) ────────────────────────────────────
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


# ── Tests ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _all_on = {
        "search": True, "deep_search": True, "weather": True,
        "calc": True, "memory": True, "github": True,
        "arxiv": True, "huggingface_models": True,
        "huggingface_datasets": True, "code_generator": True,
        "ai_news": True, "papers_with_code": True,
        "python_packages": True, "paper_summarizer": True,
        "github_trending": True, "model_benchmarks": True,
        "llm_leaderboard": True,                          # NEW
    }

    _tests = (
        # (message, expected_tool)
        # ── General tools ──────────────────────────────────────────────────
        ("what is the weather in Philadelphia",        "weather"),
        ("wheather in philadelphia",                   "weather"),   # fuzzy
        ("what is 25 times 4",                         "calc"),
        ("my name is Rakib",                           "memory"),
        ("what is my favorite food",                   "memory"),
        ("show my github repos",                       "github"),
        ("hello how are you",                          "none"),
        ("deep dive into neural networks",             "deep_search"),
        # ── LLM Leaderboard ────────────────────────────────────────────────
        ("compare gpt-4o vs claude 3.5 sonnet",        "llm_leaderboard"),
        ("how does llama 3 compare to mistral",        "llm_leaderboard"),
        ("show me the llm leaderboard",                "llm_leaderboard"),
        ("best llm for coding",                        "llm_leaderboard"),
        ("gpt-4 vs claude 3 opus",                     "llm_leaderboard"),
        ("mmlu scores for llama 3",                    "llm_leaderboard"),
        ("which llm is best",                          "llm_leaderboard"),
        ("humaneval benchmark results",                "llm_leaderboard"),
        ("compare llama 3 70b vs gpt-4",               "llm_leaderboard"),
        ("llm ranking",                                "llm_leaderboard"),
        # ── Model Benchmarks ───────────────────────────────────────────────
        ("best model for text classification",         "model_benchmarks"),
        ("state of the art for image classification",  "model_benchmarks"),
        ("benchmark for question answering",           "model_benchmarks"),
        ("text classification benchmark",              "model_benchmarks"),
        # ── Other ML tools ─────────────────────────────────────────────────
        ("find papers about RAG",                      "arxiv"),
        ("papers with code for BERT",                  "papers_with_code"),
        ("summarize arxiv paper 1706.03762",           "paper_summarizer"),
        ("find models for text classification",        "huggingface_models"),
        ("find dataset for sentiment analysis",        "huggingface_datasets"),
        ("write pytorch code for training loop",       "code_generator"),
        ("latest AI news today",                       "ai_news"),
        ("find python package for data augmentation",  "python_packages"),
        ("trending ml repos on github",                "github_trending"),
        # ── Edge cases ─────────────────────────────────────────────────────
        ("serach for AI news",                         "ai_news"),   # fuzzy typo
        ("what is transfer learning",                  "search"),
    )

    print(f"{'Message':<50} {'Expected':<20} {'Got':<20} {'Pass'}")
    print("─" * 100)

    passed = 0
    failed = 0
    for msg, expected in _tests:
        tools = dict(_all_on)
        got = route(msg, tools)["tool"]
        ok = got == expected
        if ok:
            passed += 1
        else:
            failed += 1
        status = "✅" if ok else "❌"
        print(f"{msg:<50} {expected:<20} {got:<20} {status}")

    print()
    print(f"Results: {passed}/{len(_tests)} passed, {failed} failed")
