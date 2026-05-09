"""
LLM Leaderboard Tool
Compares large language models across standard benchmarks:
  - MMLU       (Massive Multitask Language Understanding — general knowledge)
  - GSM8K      (Grade School Math — mathematical reasoning)
  - HumanEval  (Python code generation — coding ability)
  - HellaSwag  (Commonsense reasoning)

Data: Curated from official model release blogs and technical reports.
      Scores are fixed per model version — benchmarks don't change.
Fallback: DDGS search for models not in the database.

No API key required.
"""

import re
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Benchmark Database
# All scores sourced from official release blogs / technical reports:
#   OpenAI GPT-4 technical report, Meta Llama 3 blog, Mistral blog,
#   Google Gemma blog, Anthropic Claude 3 model card, etc.
# None = not officially reported by the model developer.
# ---------------------------------------------------------------------------

# Schema: model_id -> {
#   name, family, params,
#   mmlu, gsm8k, humaneval, hellaswag
# }
_LLM_DB: dict[str, dict] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o": {
        "name": "GPT-4o", "family": "OpenAI", "params": "unknown",
        "mmlu": 88.7, "gsm8k": 76.6, "humaneval": 90.2, "hellaswag": None,
    },
    "gpt-4": {
        "name": "GPT-4", "family": "OpenAI", "params": "unknown",
        "mmlu": 86.4, "gsm8k": 92.0, "humaneval": 67.0, "hellaswag": 95.3,
    },
    "gpt-3.5-turbo": {
        "name": "GPT-3.5 Turbo", "family": "OpenAI", "params": "unknown",
        "mmlu": 70.0, "gsm8k": 57.1, "humaneval": 48.1, "hellaswag": 85.5,
    },
    # ── Anthropic ───────────────────────────────────────────────────────────
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet", "family": "Anthropic", "params": "unknown",
        "mmlu": 88.7, "gsm8k": 96.4, "humaneval": 92.0, "hellaswag": None,
    },
    "claude-3-opus": {
        "name": "Claude 3 Opus", "family": "Anthropic", "params": "unknown",
        "mmlu": 86.8, "gsm8k": 95.0, "humaneval": 84.9, "hellaswag": None,
    },
    "claude-3-haiku": {
        "name": "Claude 3 Haiku", "family": "Anthropic", "params": "unknown",
        "mmlu": 75.2, "gsm8k": 88.9, "humaneval": 75.9, "hellaswag": None,
    },
    # ── Meta Llama ──────────────────────────────────────────────────────────
    "llama-3-70b": {
        "name": "Llama 3 70B Instruct", "family": "Meta", "params": "70B",
        "mmlu": 82.0, "gsm8k": 93.0, "humaneval": 81.7, "hellaswag": 88.0,
    },
    "llama-3-8b": {
        "name": "Llama 3 8B Instruct", "family": "Meta", "params": "8B",
        "mmlu": 68.4, "gsm8k": 79.6, "humaneval": 62.2, "hellaswag": 82.0,
    },
    "llama-2-70b": {
        "name": "Llama 2 70B", "family": "Meta", "params": "70B",
        "mmlu": 68.9, "gsm8k": 56.8, "humaneval": 29.9, "hellaswag": 87.3,
    },
    # ── Mistral ─────────────────────────────────────────────────────────────
    "mistral-7b": {
        "name": "Mistral 7B Instruct", "family": "Mistral", "params": "7B",
        "mmlu": 60.1, "gsm8k": 52.2, "humaneval": 30.5, "hellaswag": 81.3,
    },
    "mixtral-8x7b": {
        "name": "Mixtral 8x7B Instruct", "family": "Mistral", "params": "47B",
        "mmlu": 70.6, "gsm8k": 74.4, "humaneval": 40.2, "hellaswag": 86.7,
    },
    "mixtral-8x22b": {
        "name": "Mixtral 8x22B", "family": "Mistral", "params": "141B",
        "mmlu": 77.8, "gsm8k": 78.6, "humaneval": 53.1, "hellaswag": 88.6,
    },
    # ── Google ──────────────────────────────────────────────────────────────
    "gemma-7b": {
        "name": "Gemma 7B", "family": "Google", "params": "7B",
        "mmlu": 64.3, "gsm8k": 46.4, "humaneval": 32.3, "hellaswag": 80.9,
    },
    "gemma-2-9b": {
        "name": "Gemma 2 9B", "family": "Google", "params": "9B",
        "mmlu": 71.3, "gsm8k": 68.6, "humaneval": 40.2, "hellaswag": 87.2,
    },
    "gemma-2-27b": {
        "name": "Gemma 2 27B", "family": "Google", "params": "27B",
        "mmlu": 75.2, "gsm8k": 74.0, "humaneval": 51.8, "hellaswag": 88.8,
    },
    # ── Alibaba ─────────────────────────────────────────────────────────────
    "qwen2-72b": {
        "name": "Qwen2 72B Instruct", "family": "Alibaba", "params": "72B",
        "mmlu": 84.2, "gsm8k": 89.5, "humaneval": 86.0, "hellaswag": None,
    },
    # ── Microsoft ───────────────────────────────────────────────────────────
    "phi-3-mini": {
        "name": "Phi-3 Mini", "family": "Microsoft", "params": "3.8B",
        "mmlu": 68.8, "gsm8k": 82.5, "humaneval": 58.1, "hellaswag": None,
    },
    "phi-3-medium": {
        "name": "Phi-3 Medium", "family": "Microsoft", "params": "14B",
        "mmlu": 78.0, "gsm8k": 91.0, "humaneval": 62.3, "hellaswag": None,
    },
    # ── DeepSeek ────────────────────────────────────────────────────────────
    "deepseek-v2": {
        "name": "DeepSeek-V2", "family": "DeepSeek", "params": "236B MoE",
        "mmlu": 78.5, "gsm8k": 79.2, "humaneval": 81.1, "hellaswag": None,
    },
    # ── Cohere ──────────────────────────────────────────────────────────────
    "command-r-plus": {
        "name": "Command R+", "family": "Cohere", "params": "104B",
        "mmlu": 75.7, "gsm8k": 72.3, "humaneval": None, "hellaswag": None,
    },
}

# Aliases so users can type naturally
_ALIASES: dict[str, str] = {
    # GPT
    "gpt4o": "gpt-4o", "gpt 4o": "gpt-4o", "gpt-4o": "gpt-4o",
    "gpt4": "gpt-4", "gpt 4": "gpt-4", "gpt-4": "gpt-4",
    "gpt3.5": "gpt-3.5-turbo", "gpt 3.5": "gpt-3.5-turbo", "chatgpt": "gpt-3.5-turbo",
    # Claude
    "claude 3.5 sonnet": "claude-3.5-sonnet", "claude sonnet": "claude-3.5-sonnet",
    "claude 3 opus": "claude-3-opus", "claude opus": "claude-3-opus",
    "claude 3 haiku": "claude-3-haiku", "claude haiku": "claude-3-haiku",
    # Llama
    "llama3 70b": "llama-3-70b", "llama 3 70b": "llama-3-70b", "llama3-70b": "llama-3-70b",
    "llama3 8b": "llama-3-8b", "llama 3 8b": "llama-3-8b", "llama3-8b": "llama-3-8b",
    "llama2 70b": "llama-2-70b", "llama 2 70b": "llama-2-70b",
    # Mistral
    "mistral": "mistral-7b", "mistral 7b": "mistral-7b",
    "mixtral": "mixtral-8x7b", "mixtral 8x7b": "mixtral-8x7b",
    "mixtral 8x22b": "mixtral-8x22b",
    # Gemma
    "gemma": "gemma-7b", "gemma 7b": "gemma-7b",
    "gemma2 9b": "gemma-2-9b", "gemma 2 9b": "gemma-2-9b",
    "gemma2 27b": "gemma-2-27b", "gemma 2 27b": "gemma-2-27b",
    # Others
    "qwen2": "qwen2-72b", "qwen 2": "qwen2-72b", "qwen2 72b": "qwen2-72b",
    "phi3 mini": "phi-3-mini", "phi 3 mini": "phi-3-mini",
    "phi3 medium": "phi-3-medium", "phi 3 medium": "phi-3-medium",
    "deepseek": "deepseek-v2", "deepseek v2": "deepseek-v2",
    "command r+": "command-r-plus", "command r plus": "command-r-plus",
}

_BENCHMARKS = {
    "mmlu":      ("MMLU",      "General Knowledge"),
    "gsm8k":     ("GSM8K",     "Math Reasoning"),
    "humaneval": ("HumanEval", "Coding"),
    "hellaswag": ("HellaSwag", "Commonsense Reasoning"),
}

_FAILURE = "Sorry I could not fetch LLM leaderboard data right now. Please try again."

_FILLER_WORDS = (
    "compare", "comparison", "versus", "vs", "how does", "how do",
    "leaderboard", "benchmark", "benchmarks", "which is better",
    "which is best", "best llm", "top llm", "ranking", "rank",
    "performance", "performs", "perform", "score", "scores",
    "state of the art", "sota",
)

_LEADING_STOPWORDS = re.compile(
    r"^(for|about|on|of|with|the|a|an|and)\s+", re.IGNORECASE
)

# Benchmark keywords to detect which benchmark user is asking about
_BENCHMARK_KEYWORDS = {
    "mmlu":      ("mmlu", "general knowledge", "multitask", "knowledge"),
    "gsm8k":     ("gsm8k", "math", "maths", "mathematics", "reasoning", "arithmetic"),
    "humaneval": ("humaneval", "code", "coding", "programming", "python"),
    "hellaswag": ("hellaswag", "commonsense", "common sense"),
}


# ---------------------------------------------------------------------------
# Query parsing
# ---------------------------------------------------------------------------

def _clean_query(message: str) -> str:
    """Remove filler words and leftover prepositions."""
    q = message.strip().lower()
    for filler in sorted(_FILLER_WORDS, key=len, reverse=True):
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    q = _LEADING_STOPWORDS.sub("", q).strip()
    return q if q else message.strip()


def _find_models_in_query(query: str) -> list[str]:
    """
    Extract model IDs mentioned in the query.
    Returns list of model_ids from _LLM_DB.

    Uses regex with negative lookahead so 'gpt-4o' never consumes a
    separately typed 'gpt-4', allowing all explicitly named models to
    be detected correctly even when they share a common prefix.
    """
    q = query.lower()
    found = []

    # Match aliases longest-first with word-boundary awareness
    # Negative lookahead (?![\w\-.]) prevents 'gpt-4' matching inside 'gpt-4o'
    for alias in sorted(_ALIASES.keys(), key=len, reverse=True):
        pattern = re.escape(alias) + r"(?![\w\-.])"
        match = re.search(pattern, q)
        if match:
            model_id = _ALIASES[alias]
            if model_id not in found:
                found.append(model_id)
            # Blank out the matched span so it can't be re-matched
            q = q[:match.start()] + " " * (match.end() - match.start()) + q[match.end():]

    # Check direct model IDs against the remaining query
    for model_id in sorted(_LLM_DB.keys(), key=len, reverse=True):
        pattern = re.escape(model_id) + r"(?![\w\-.])"
        match = re.search(pattern, q)
        if match and model_id not in found:
            found.append(model_id)
            q = q[:match.start()] + " " * (match.end() - match.start()) + q[match.end():]

    return found


def _find_benchmark_in_query(query: str) -> str | None:
    """Detect if user is asking about a specific benchmark."""
    q = query.lower()
    for bench_id, keywords in _BENCHMARK_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return bench_id
    return None


def _find_family_in_query(query: str) -> list[str]:
    """Detect if user is asking about a model family e.g. 'all llama models'."""
    q = query.lower()
    families = {
        "openai": "OpenAI", "gpt": "OpenAI",
        "anthropic": "Anthropic", "claude": "Anthropic",
        "meta": "Meta", "llama": "Meta",
        "mistral": "Mistral", "mixtral": "Mistral",
        "google": "Google", "gemma": "Google",
        "microsoft": "Microsoft", "phi": "Microsoft",
        "alibaba": "Alibaba", "qwen": "Alibaba",
        "deepseek": "DeepSeek",
        "cohere": "Cohere", "command": "Cohere",
    }
    found_families = []
    for keyword, family in families.items():
        if keyword in q and family not in found_families:
            found_families.append(family)
    return found_families


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_score(value: float | None) -> str:
    """Format a benchmark score for display."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _format_model_row(model: dict) -> list[str]:
    """Format a single model's benchmark data as lines."""
    lines = []
    lines.append(
        f"   Downloads/Params: {model['params']}   Family: {model['family']}"
    )
    lines.append(f"   {'─' * 36}")
    lines.append(f"   MMLU       (General Knowledge) : {_fmt_score(model['mmlu'])}")
    lines.append(f"   GSM8K      (Math Reasoning)    : {_fmt_score(model['gsm8k'])}")
    lines.append(f"   HumanEval  (Coding)            : {_fmt_score(model['humaneval'])}")
    lines.append(f"   HellaSwag  (Commonsense)        : {_fmt_score(model['hellaswag'])}")
    return lines


# ---------------------------------------------------------------------------
# Core lookup functions
# ---------------------------------------------------------------------------

def _compare_specific_models(model_ids: list[str]) -> str:
    """Compare two or more specific models side by side."""
    models = [_LLM_DB[mid] for mid in model_ids]
    lines = [
        f"Comparing {' vs '.join(m['name'] for m in models)}:",
        "",
    ]

    for i, (mid, model) in enumerate(zip(model_ids, models), start=1):
        lines.append(f"{i}. {model['name']}")
        lines.extend(_format_model_row(model))
        if i < len(models):
            lines.append("")

    # Add winner summary if exactly 2 models
    if len(models) == 2:
        lines.append("")
        lines.append("── Summary ──")
        m1, m2 = models[0], models[1]
        bench_labels = {
            "mmlu": "General Knowledge",
            "gsm8k": "Math",
            "humaneval": "Coding",
            "hellaswag": "Commonsense",
        }
        for bench, label in bench_labels.items():
            s1, s2 = m1[bench], m2[bench]
            if s1 is not None and s2 is not None:
                winner = m1["name"] if s1 > s2 else m2["name"]
                diff = abs(s1 - s2)
                lines.append(f"   {label}: {winner} wins (+{diff:.1f}%)")

    lines.append("")
    lines.append("Note: Scores from official model release reports. N/A = not publicly reported.")
    return "\n".join(lines)


def _rank_by_benchmark(benchmark: str) -> str:
    """Rank all models in DB by a specific benchmark score."""
    bench_name, bench_label = _BENCHMARKS[benchmark]

    ranked = [
        (mid, m) for mid, m in _LLM_DB.items()
        if m[benchmark] is not None
    ]
    ranked.sort(key=lambda x: x[1][benchmark], reverse=True)

    lines = [
        f"LLM Leaderboard — {bench_name} ({bench_label}):",
        "",
    ]

    for i, (mid, model) in enumerate(ranked[:10], start=1):
        score = _fmt_score(model[benchmark])
        lines.append(
            f"{i:2}. {model['name']:<30} {score}   [{model['family']} · {model['params']}]"
        )

    lines.append("")
    lines.append("Note: Scores from official model release reports.")
    return "\n".join(lines)


def _show_family(families: list[str]) -> str:
    """Show all models from a specific family."""
    family_models = [
        (mid, m) for mid, m in _LLM_DB.items()
        if m["family"] in families
    ]
    family_models.sort(key=lambda x: x[1].get("mmlu") or 0, reverse=True)

    family_label = " / ".join(families)
    lines = [f"LLM Leaderboard — {family_label} Models:", ""]

    for i, (mid, model) in enumerate(family_models, start=1):
        lines.append(f"{i}. {model['name']}")
        lines.extend(_format_model_row(model))
        if i < len(family_models):
            lines.append("")

    lines.append("")
    lines.append("Note: Scores from official model release reports. N/A = not publicly reported.")
    return "\n".join(lines)


def _show_overall_leaderboard() -> str:
    """Show overall leaderboard ranked by MMLU (best general proxy)."""
    ranked = [
        (mid, m) for mid, m in _LLM_DB.items()
        if m["mmlu"] is not None
    ]
    ranked.sort(key=lambda x: x[1]["mmlu"], reverse=True)

    lines = ["LLM Leaderboard — Overall Rankings (sorted by MMLU):", ""]
    lines.append(
        f"{'#':<3} {'Model':<30} {'MMLU':>6} {'GSM8K':>7} {'HumanEval':>10} {'HellaSwag':>10}"
    )
    lines.append("─" * 70)

    for i, (mid, m) in enumerate(ranked, start=1):
        lines.append(
            f"{i:<3} {m['name']:<30} "
            f"{_fmt_score(m['mmlu']):>6} "
            f"{_fmt_score(m['gsm8k']):>7} "
            f"{_fmt_score(m['humaneval']):>10} "
            f"{_fmt_score(m['hellaswag']):>10}"
        )

    lines.append("")
    lines.append("Benchmarks: MMLU=General Knowledge | GSM8K=Math | HumanEval=Coding | HellaSwag=Commonsense")
    lines.append("Note: Scores from official model release reports. N/A = not publicly reported.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DDGS fallback for unknown models
# ---------------------------------------------------------------------------

def _ddg_fallback(query: str) -> str:
    """Search DuckDuckGo for benchmark scores of models not in the database."""
    try:
        search_query = f"{query} benchmark MMLU GSM8K HumanEval score results"
        with DDGS() as ddgs:
            raw = list(ddgs.text(search_query, max_results=5))

        if not raw:
            return _FAILURE

        lines = [f"Here are leaderboard results for '{query}':", ""]
        seen = set()

        for item in raw:
            url   = item.get("href", "")
            body  = (item.get("body") or "").strip()
            title = (item.get("title") or "").strip()
            if not body or not title or url in seen:
                continue
            seen.add(url)
            if len(body) > 300:
                body = body[:300] + "..."
            lines.append(f"• {title}")
            lines.append(f"  {body}")
            lines.append(f"  Source: {url}")
            lines.append("")

        return "\n".join(lines) if len(lines) > 2 else _FAILURE

    except Exception:
        return _FAILURE


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_llm_leaderboard(message: str) -> str:
    """
    Compare LLMs across standard benchmarks (MMLU, GSM8K, HumanEval, HellaSwag).

    Handles:
      - Specific model comparison: "compare Llama 3 70B vs Mistral 7B"
      - Benchmark ranking:         "best models for coding"
      - Family listing:            "all Google models"
      - Overall leaderboard:       "show me the leaderboard"
      - Unknown models:            DDGS fallback

    Args:
        message: User message about LLM benchmarks

    Returns:
        Formatted benchmark comparison string
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        raw = message.lower()

        # Case 1: Specific model comparison (2+ models named)
        model_ids = _find_models_in_query(raw)
        if len(model_ids) >= 2:
            return _compare_specific_models(model_ids)

        # Case 2: Single model — show its full benchmark card
        if len(model_ids) == 1:
            return _compare_specific_models(model_ids)

        # Case 3: Specific benchmark ranking
        benchmark = _find_benchmark_in_query(raw)
        if benchmark:
            return _rank_by_benchmark(benchmark)

        # Case 4: Model family
        families = _find_family_in_query(raw)
        if families:
            return _show_family(families)

        # Case 5: General leaderboard request
        general_keywords = ("leaderboard", "all models", "overall", "best", "top", "ranking")
        if any(kw in raw for kw in general_keywords):
            return _show_overall_leaderboard()

        # Case 6: Unknown model — DDGS fallback
        return _ddg_fallback(query)

    except Exception:
        return _FAILURE


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        "compare Llama 3 70B vs Mistral 7B",
        "best models for coding",
        "show me all Google models",
        "show me the overall leaderboard",
        "how does GPT-4o compare to Claude 3.5 Sonnet",
    ]
    for i, test in enumerate(tests, start=1):
        print(f"Query {i}: {test}")
        print(run_llm_leaderboard(test))
        print("\n" + "═" * 70 + "\n")
