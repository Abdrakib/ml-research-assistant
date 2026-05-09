"""
Python Package Finder Tool
Searches for Python packages on PyPI (Python Package Index).
Uses PyPI public API. No API key required.
"""

import requests
from ddgs import DDGS

_FAILURE = "Sorry I could not search for Python packages right now. Please try again."
_PYPI_URL = "https://pypi.org/pypi/{package}/json"
_PYPI_SEARCH_URL = "https://pypi.org/search/"

_FILLER_WORDS = (
    "find package",
    "find packages",
    "search package",
    "search packages",
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
    "what package",
    "which package",
    "best package",
    "best library",
)


def _clean_query(message: str) -> str:
    """Remove filler words to get clean search query."""
    q = message.strip().lower()
    for filler in _FILLER_WORDS:
        q = q.replace(filler, "")
    q = q.strip().strip("?.,!")
    return q if q else message.strip()


def _get_package_info(package_name: str) -> dict:
    """Get detailed info about a specific package from PyPI."""
    try:
        response = requests.get(
            _PYPI_URL.format(package=package_name),
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            info = data.get("info", {})
            return {
                "name": info.get("name", package_name),
                "version": info.get("version", "Unknown"),
                "summary": info.get("summary", "No description"),
                "home_page": info.get("home_page") or info.get("project_url") or f"https://pypi.org/project/{package_name}",
                "author": info.get("author", "Unknown"),
            }
    except Exception:
        pass
    return {}


def run_python_packages(message: str) -> str:
    """
    Search PyPI for Python packages matching the user query.
    Uses DuckDuckGo to search PyPI since PyPI search
    does not have a public JSON API.

    Args:
        message: User message containing the search topic

    Returns:
        Formatted string with package names, versions and descriptions
    """
    try:
        query = _clean_query(message)
        if not query:
            return _FAILURE

        # Use DuckDuckGo to search PyPI
        search_query = f"site:pypi.org {query} python package"
        results = []

        with DDGS() as ddgs:
            raw = list(ddgs.text(search_query, max_results=8))

        # Filter only PyPI results and extract package names
        packages = []
        seen = set()

        for item in raw:
            url = item.get("href", "")
            title = item.get("title", "")
            body = item.get("body", "")

            # Only process pypi.org URLs
            if "pypi.org/project/" in url:
                # Extract package name from URL
                parts = url.split("pypi.org/project/")
                if len(parts) > 1:
                    pkg_name = parts[1].strip("/").split("/")[0]
                    if pkg_name and pkg_name not in seen:
                        seen.add(pkg_name)
                        packages.append({
                            "name": pkg_name,
                            "summary": body[:200] if body else title,
                            "url": f"https://pypi.org/project/{pkg_name}",
                        })

            if len(packages) >= 5:
                break

        if not packages:
            return f"Sorry I could not find any Python packages for '{query}'."

        lines = [f"Here are Python packages for '{query}':", ""]

        for i, pkg in enumerate(packages[:5], start=1):
            name = pkg["name"]
            summary = pkg["summary"][:200] if pkg["summary"] else "No description"
            url = pkg["url"]

            lines.append(f"{i}. {name}")
            lines.append(f"   Description: {summary}")
            lines.append(f"   Install: pip install {name}")
            lines.append(f"   Link: {url}")
            if i < min(5, len(packages)):
                lines.append("")

        return "\n".join(lines)

    except Exception:
        return _FAILURE


if __name__ == "__main__":
    print(run_python_packages("find python package for data augmentation"))
    print("\n---\n")
    print(run_python_packages("best library for image processing"))
