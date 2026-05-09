"""
ML Code Generator Tool
Generates clean, well-commented Python/ML code using the LLM.
No external API needed - uses the model already running in the app.
"""

_FAILURE = "Sorry I could not generate code right now. Please try again."

_CODING_SYSTEM_PROMPT = """You are a senior ML engineer with 10 years of experience.
Your job is to write clean, well-commented Python code.

Rules:
- Return ONLY the code
- No explanations before or after the code
- Include all necessary imports at the top
- Add clear comments explaining each step
- Use best practices
- Keep it concise and readable
- Use PyTorch unless user specifies otherwise"""

_FILLER_WORDS = (
    "write code for",
    "write code to",
    "generate code for",
    "generate code to",
    "implement",
    "create a function",
    "write a function",
    "write a script",
    "give me code for",
    "give me code to",
    "show me code for",
    "show me how to code",
    "how do i code",
    "pytorch code for",
    "tensorflow code for",
    "python code for",
    "write me",
    "create",
    "build",
    "code for",
    "code to",
)


def _clean_query(message: str) -> str:
    """Remove filler words to get clean code request."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q if q else message.strip()


def run_code_generator(message: str) -> str:
    """
    Generate ML Python code based on user request.
    Uses the LLM with a specialized ML engineer system prompt.

    Args:
        message: User message describing what code to generate

    Returns:
        Generated Python code as a string
    """
    try:
        from model import generate_response

        query = _clean_query(message)
        if not query:
            return _FAILURE

        prompt = f"""{_CODING_SYSTEM_PROMPT}

Write Python code for: {query}

Code:"""

        code = generate_response(prompt).strip()

        if not code:
            return _FAILURE

        # Wrap in code block if not already wrapped
        if not code.startswith("```"):
            code = f"```python\n{code}\n```"

        return f"Here is the code for '{query}':\n\n{code}"

    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print("Testing with offline store...")
    import sys
    sys.path.insert(0, ".")

    # Simple test without loading the full model
    query = "PyTorch training loop"
    cleaned = _clean_query(f"write code for {query}")
    print(f"Cleaned query: '{cleaned}'")
    print(f"Would generate code for: '{cleaned}'")
    print("To test with real model run the full app")
