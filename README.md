---
title: ML Research Assistant
emoji: 🧠
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: true
license: mit
---

# 🧠 ML Research Assistant

An AI-powered research assistant built for ML engineers, researchers, and students. Ask about papers, compare models, explore benchmarks, and stay up to date with the latest AI news — all in one place.

**Live Demo → [HuggingFace Space](https://huggingface.co/spaces/Abdourakib/ml-research-assistant)**

---

## What It Does

Instead of jumping between ArXiv, Papers With Code, HuggingFace, and Google, you ask in plain English and the assistant routes your query to the right tool automatically.

| You ask | Tool fires |
|---|---|
| "compare GPT-4o vs Claude 3.5 on coding" | LLM Leaderboard |
| "best model for question answering" | Model Benchmarks |
| "find papers on RAG with code" | Papers w/ Code + ArXiv |
| "what is the weather in Philadelphia" | Weather |
| "latest AI news" | AI News Feed |
| "write a PyTorch training loop" | Code Generator |

---

## Features

- **17 specialized tools** auto-activated by a fuzzy keyword router
- **LLM Leaderboard** — compare 20 major LLMs across MMLU, GSM8K, HumanEval, HellaSwag with real scores from official release reports
- **Model Benchmarks** — task-specific leaderboards pulled from HuggingFace model cards with real F1/Accuracy scores
- **Papers w/ Code** — ArXiv paper search + GitHub implementation links with star counts
- **Live AI News Feed** — sidebar news tab that auto-refreshes, click any headline to ask the assistant about it
- **Chat History** — previous conversations saved in the sidebar
- **Tool toggles** — enable/disable any tool from the sidebar Tools tab
- **Warm parchment UI** — clean, editorial dark-white theme built with Gradio + CSS

---

## Tools

| Tool | Description |
|---|---|
| 📄 ArXiv | Search ML papers by topic |
| 💻 Papers w/ Code | Find papers + GitHub implementations |
| 🏆 LLM Leaderboard | Compare LLMs across MMLU, GSM8K, HumanEval, HellaSwag |
| 📊 Model Benchmarks | Task-specific model rankings with real scores |
| 🔍 Web Search | Live web search via DuckDuckGo |
| 🔬 Deep Search | Comprehensive multi-source search |
| 📝 Paper Summarizer | Summarize any ArXiv paper |
| 🤗 HF Models | Search HuggingFace models |
| 🗄 HF Datasets | Search HuggingFace datasets |
| 📰 AI News | Live AI/ML news feed |
| 📦 Python Packages | Find Python libraries for ML tasks |
| 🔥 GitHub Trending | Trending ML repositories |
| ⚙️ Code Generator | Generate PyTorch/TensorFlow code |
| 🌤 Weather | Current weather |
| 🧠 Memory | Remember user context across messages |
| 🧮 Calculator | Math expressions |
| 🐙 GitHub | GitHub repo search |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  HuggingFace Space                  │
│                                                     │
│  Gradio Native UI (gr.Blocks + CSS)                 │
│  Sidebar: History / Tools / AI News tabs            │
│          │                                          │
│          │  Gradio event handlers                   │
│          ▼                                          │
│  router.py  → keyword + fuzzy matching              │
│  17 tool modules                                    │
│  Qwen2.5-7B-Instruct (ZeroGPU)                     │
└─────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | Qwen2.5-7B-Instruct |
| Backend | Python, Gradio 5+ |
| GPU | HuggingFace ZeroGPU |
| UI | Gradio native components + CSS |
| News / Search | DuckDuckGo Search (ddgs) |
| Paper Search | ArXiv API |
| Code Links | GitHub Search API |
| Fuzzy Routing | thefuzz |

---

## Run Locally

```bash
git clone https://github.com/Abdrakib/ml-research-assistant.git
cd ml-research-assistant
pip install -r requirements.txt
python app.py
```

---

## Project Structure

```
ml-research-assistant/
├── app.py                  # Main Gradio app + UI
├── model.py                # Qwen2.5-7B inference + ZeroGPU
├── router.py               # Keyword + fuzzy tool router
├── prompt_builder.py       # Prompt construction
├── requirements.txt
├── tools/
│   ├── arxiv_search.py
│   ├── papers_with_code.py
│   ├── llm_leaderboard.py
│   ├── model_benchmarks.py
│   ├── ai_news.py
│   ├── huggingface_models.py
│   ├── huggingface_datasets.py
│   ├── code_generator.py
│   ├── paper_summarizer.py
│   ├── github_trending.py
│   ├── python_packages.py
│   ├── calculator.py
│   ├── deep_search.py
│   ├── github.py
│   ├── memory.py
│   ├── search.py
│   └── weather.py
```

---

## Author

**Rakib** — CS graduate, Community College of Philadelphia  
GitHub: [Abdrakib](https://github.com/Abdrakib)  
HuggingFace: [Abdourakib](https://huggingface.co/Abdourakib)  
Portfolio: [abdourakib.com](https://abdourakib.com)

---

## License

MIT
