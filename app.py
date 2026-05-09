"""
ML Research Assistant — app.py
Clean Gradio 6 native UI with professional dark theme.
"""

_memory_store = {}

import copy
import uuid
import gradio as gr

from model import generate_response
from prompt_builder import build_auto_enable_notice, build_prompt
from router import route
from tools.calculator           import run_calculator
from tools.deep_search          import run_deep_search
from tools.github               import run_github
from tools.memory               import get_memory_context, run_memory
from tools.search               import run_search
from tools.weather              import run_weather
from tools.arxiv_search         import run_arxiv_search
from tools.papers_with_code     import run_papers_with_code
from tools.huggingface_models   import run_huggingface_models
from tools.huggingface_datasets import run_huggingface_datasets
from tools.code_generator       import run_code_generator
from tools.ai_news              import run_ai_news, get_ai_news_feed
from tools.python_packages      import run_python_packages
from tools.paper_summarizer     import run_paper_summarizer
from tools.github_trending      import run_github_trending
from tools.model_benchmarks     import run_model_benchmarks
from tools.llm_leaderboard      import run_llm_leaderboard

import tools.memory as _mem_module
_mem_module._FORCE_OFFLINE_STORE = True
_mem_module._OFFLINE_STORE = _memory_store

# ---------------------------------------------------------------------------
# Tool state
# ---------------------------------------------------------------------------

_tool_state = {
    "search": True, "weather": True, "calc": True,
    "memory": True, "deep_search": True, "github": False,
    "arxiv": True, "papers_with_code": True, "llm_leaderboard": True,
    "model_benchmarks": True, "huggingface_models": True,
    "huggingface_datasets": True, "code_generator": True,
    "ai_news": True, "python_packages": True,
    "paper_summarizer": True, "github_trending": True,
}

_TOOL_LABELS = {
    "weather": "🌤 Weather", "search": "🔍 Web Search",
    "deep_search": "🔬 Deep Search", "calc": "🧮 Calculator",
    "memory": "🧠 Memory", "github": "🐙 GitHub",
    "arxiv": "📄 Arxiv", "papers_with_code": "💻 Papers w/ Code",
    "llm_leaderboard": "🏆 LLM Leaderboard",
    "model_benchmarks": "📊 Benchmarks",
    "huggingface_models": "🤗 HF Models",
    "huggingface_datasets": "🗄 HF Datasets",
    "code_generator": "⚙️ Code Generator",
    "ai_news": "📰 AI News",
    "python_packages": "📦 Python Packages",
    "paper_summarizer": "📝 Paper Summarizer",
    "github_trending": "🔥 GitHub Trending",
    "none": "No tool",
}

TOOLS_LIST = [
    ("arxiv",                "📄 Arxiv"),
    ("papers_with_code",     "💻 Papers w/ Code"),
    ("llm_leaderboard",      "🏆 LLM Leaderboard"),
    ("model_benchmarks",     "📊 Benchmarks"),
    ("search",               "🔍 Web Search"),
    ("paper_summarizer",     "📝 Paper Summarizer"),
    ("huggingface_models",   "🤗 HF Models"),
    ("huggingface_datasets", "🗄 HF Datasets"),
    ("ai_news",              "📰 AI News"),
    ("python_packages",      "📦 Python Packages"),
    ("github_trending",      "🔥 GitHub Trending"),
    ("code_generator",       "⚙️ Code Generator"),
    ("weather",              "🌤 Weather"),
    ("deep_search",          "🔬 Deep Search"),
    ("memory",               "🧠 Memory"),
    ("calc",                 "🧮 Calculator"),
    ("github",               "🐙 GitHub"),
]

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def _title_from(hist):
    if not hist:
        return "New chat"
    first = hist[0]
    text = str(first.get("content") or "").strip().replace("\n", " ")
    return (text[:45] + "…") if len(text) > 45 else (text or "New chat")


def chat(user_message, history):
    if not (user_message or "").strip():
        return history, ""

    active = dict(_tool_state)
    routed = route(user_message, active)

    if routed.get("auto_enabled"):
        t = routed.get("tool")
        if t and t != "none":
            _tool_state[t] = True

    tool_name   = routed["tool"]
    tool_result = ""

    if   tool_name == "weather":              tool_result = run_weather(user_message)
    elif tool_name == "search":               tool_result = run_search(user_message)
    elif tool_name == "deep_search":          tool_result = run_deep_search(user_message)
    elif tool_name == "calc":                 tool_result = run_calculator(user_message)
    elif tool_name == "memory":               tool_result = run_memory(user_message)
    elif tool_name == "github":               tool_result = run_github(user_message, "")
    elif tool_name == "arxiv":                tool_result = run_arxiv_search(user_message)
    elif tool_name == "papers_with_code":     tool_result = run_papers_with_code(user_message)
    elif tool_name == "llm_leaderboard":      tool_result = run_llm_leaderboard(user_message)
    elif tool_name == "model_benchmarks":     tool_result = run_model_benchmarks(user_message)
    elif tool_name == "huggingface_models":   tool_result = run_huggingface_models(user_message)
    elif tool_name == "huggingface_datasets": tool_result = run_huggingface_datasets(user_message)
    elif tool_name == "code_generator":       tool_result = run_code_generator(user_message)
    elif tool_name == "ai_news":              tool_result = run_ai_news(user_message)
    elif tool_name == "python_packages":      tool_result = run_python_packages(user_message)
    elif tool_name == "paper_summarizer":     tool_result = run_paper_summarizer(user_message)
    elif tool_name == "github_trending":      tool_result = run_github_trending(user_message)

    if tool_name == "memory" and tool_result == "Got it! I will remember that":
        history = list(history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "✅ Got it! I will remember that.\n\n*Tool: 🧠 Memory*"})
        return history, ""

    if tool_name == "memory" and tool_result.startswith("Here is what I know"):
        history = list(history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": tool_result + "\n\n*Tool: 🧠 Memory*"})
        return history, ""

    auto_notice = ""
    if routed.get("auto_enabled") and tool_name not in (None, "none"):
        auto_notice = build_auto_enable_notice(tool_name)

    full_prompt = build_prompt(user_message, tool_result, get_memory_context())
    reply = generate_response(full_prompt)

    if auto_notice:
        reply = f"{auto_notice}\n\n{reply}"

    label = _TOOL_LABELS.get(tool_name, "No tool")
    reply += f"\n\n*Tool: {label}*"

    history = list(history or [])
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def new_chat(history, archives):
    arch = list(archives or [])
    if history:
        arch.insert(0, {
            "id": uuid.uuid4().hex,
            "title": _title_from(history),
            "messages": copy.deepcopy(history),
        })
    return [], arch


def fetch_news():
    feed = get_ai_news_feed()
    if not feed:
        return "No news available right now. Try again later."
    lines = ["# 📰 Latest AI News\n"]
    for item in feed[:6]:
        lines.append(f"### {item['title']}")
        lines.append(f"*{item['tag']} · {item['source']} · {item['time']}*")
        lines.append(f"{item['summary']}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
body { background: #0f0f13 !important; }

.gradio-container {
    background: #0f0f13 !important;
    font-family: 'Inter', sans-serif !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

footer { display: none !important; }

/* ── Layout ──────────────────────────────────────────────────── */
.contain { padding: 0 !important; }
.gap     { gap: 4px !important; }

/* ── Sidebar ──────────────────────────────────────────────────── */
#sidebar {
    background: #16161f !important;
    border-right: 1px solid #2a2a3a !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* App title */
#app-title {
    background: #16161f !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
#app-title p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #f1f5f9 !important;
    padding: 14px 14px 10px !important;
    border-bottom: 1px solid #2a2a3a !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}

/* New chat button */
#new-chat-btn {
    margin: 8px 8px 4px !important;
    background: #0f0f13 !important;
    border: 1px solid #2a2a3a !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
    width: calc(100% - 16px) !important;
}
#new-chat-btn:hover {
    background: #1e1e2e !important;
    color: #e2e8f0 !important;
    border-color: #3a3a4a !important;
}

/* Sidebar tabs */
#sidebar-tabs {
    background: #16161f !important;
    border: none !important;
}
#sidebar-tabs .tab-nav {
    background: #16161f !important;
    border-bottom: 1px solid #2a2a3a !important;
    padding: 0 6px !important;
    gap: 0 !important;
}
#sidebar-tabs .tab-nav button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #64748b !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 8px 10px !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.15s !important;
}
#sidebar-tabs .tab-nav button.selected {
    color: #D97706 !important;
    border-bottom-color: #D97706 !important;
    background: transparent !important;
}
#sidebar-tabs .tabitem {
    background: #16161f !important;
    border: none !important;
    padding: 6px 0 !important;
}

/* History buttons */
.hist-btn button {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    text-align: left !important;
    font-size: 11px !important;
    padding: 6px 12px !important;
    border-radius: 6px !important;
    margin: 1px 6px !important;
    width: calc(100% - 12px) !important;
    font-family: 'Inter', sans-serif !important;
    justify-content: flex-start !important;
    transition: all 0.12s !important;
}
.hist-btn button:hover {
    background: #1e1e2e !important;
    color: #e2e8f0 !important;
}

/* Tool checkboxes */
#tools-container {
    padding: 4px 8px !important;
    background: #16161f !important;
}
#tools-container .wrap {
    gap: 2px !important;
}
#tools-container label {
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 5px 8px !important;
    border-radius: 6px !important;
    transition: background 0.12s !important;
}
#tools-container label:hover {
    background: #1e1e2e !important;
}
#tools-container input[type=checkbox] {
    accent-color: #a78bfa !important;
}

/* News */
#news-container {
    background: #16161f !important;
    padding: 0 6px !important;
}
#news-box {
    background: transparent !important;
    border: none !important;
}
#news-box textarea {
    background: #0f0f13 !important;
    border: 1px solid #2a2a3a !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
    line-height: 1.6 !important;
}
#news-refresh {
    background: #0f0f13 !important;
    border: 1px solid #2a2a3a !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
    margin-bottom: 6px !important;
    width: 100% !important;
}
#news-refresh:hover {
    border-color: #D97706 !important;
    color: #D97706 !important;
}

/* ── Main chat area ───────────────────────────────────────────── */
#main-col {
    background: #0f0f13 !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Chat header */
#chat-header {
    background: #0f0f13 !important;
    border: none !important;
    border-bottom: 1px solid #2a2a3a !important;
    padding: 0 !important;
    margin: 0 !important;
}
#chat-header p {
    padding: 12px 18px !important;
    margin: 0 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #f1f5f9 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
}

/* Chatbot */
#chatbot {
    background: #0f0f13 !important;
    border: none !important;
    flex: 1 !important;
}
#chatbot .bubble-wrap {
    background: #0f0f13 !important;
    padding: 16px !important;
}

/* User bubble */
#chatbot .message.user > div {
    background: #1e1e40 !important;
    color: #e2e8f0 !important;
    border-radius: 14px 14px 3px 14px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    padding: 10px 14px !important;
}

/* Bot bubble */
#chatbot .message.bot > div {
    background: #16161f !important;
    color: #e2e8f0 !important;
    border-radius: 3px 14px 14px 14px !important;
    border: 1px solid #2a2a3a !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
    padding: 10px 14px !important;
}

/* Bot avatar */
#chatbot .message.bot .avatar-container img,
#chatbot .message.bot .avatar-container {
    background: #1e1e40 !important;
    border-radius: 50% !important;
}

/* ── Input area ───────────────────────────────────────────────── */
#input-row {
    background: #0f0f13 !important;
    border-top: 1px solid #2a2a3a !important;
    padding: 10px 14px !important;
    gap: 8px !important;
    align-items: flex-end !important;
}

#msg-input {
    background: transparent !important;
    border: none !important;
    flex: 1 !important;
}
#msg-input textarea {
    background: #16161f !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    resize: none !important;
    line-height: 1.5 !important;
    transition: border-color 0.2s !important;
}
#msg-input textarea:focus {
    border-color: #D97706 !important;
    box-shadow: 0 0 0 2px rgba(217,119,6,0.15) !important;
    outline: none !important;
}
#msg-input textarea::placeholder { color: #3a3a4a !important; }
#msg-input label { display: none !important; }

/* Send button — GOLD */
#send-btn {
    background: #D97706 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    min-width: 42px !important;
    max-width: 42px !important;
    height: 42px !important;
    font-size: 18px !important;
    padding: 0 !important;
    transition: all 0.15s !important;
    flex-shrink: 0 !important;
}
#send-btn:hover {
    background: #B45309 !important;
    transform: scale(1.05) !important;
}

/* ── Scrollbar ───────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #D97706; }
"""

# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(title="ML Research Assistant", fill_height=True) as demo:

    archive_state = gr.State([])

    with gr.Row(equal_height=True):

        # ── SIDEBAR ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=230, elem_id="sidebar"):

            gr.Markdown("🧠 **ML Research** · *Assistant*", elem_id="app-title")

            new_chat_btn = gr.Button("➕  New chat", elem_id="new-chat-btn", size="sm")

            with gr.Tabs(elem_id="sidebar-tabs"):

                with gr.Tab("💬 History"):
                    @gr.render(inputs=[archive_state])
                    def render_history(archives):
                        for conv in (archives or []):
                            b = gr.Button(
                                "💬 " + (conv.get("title") or "Untitled")[:32],
                                elem_classes=["hist-btn"],
                                size="sm",
                            )
                            b.click(
                                lambda c=conv: c.get("messages") or [],
                                None, chatbot,
                            )

                with gr.Tab("⚡ Tools"):
                    with gr.Column(elem_id="tools-container"):
                        for tool_id, tool_label in TOOLS_LIST:
                            cb = gr.Checkbox(
                                label=tool_label,
                                value=_tool_state.get(tool_id, True),
                            )
                            def make_toggle(tid):
                                def toggle(val):
                                    _tool_state[tid] = val
                                return toggle
                            cb.change(make_toggle(tool_id), inputs=[cb], outputs=[])

                with gr.Tab("📰 AI News"):
                    with gr.Column(elem_id="news-container"):
                        news_refresh = gr.Button("🔄 Refresh News", elem_id="news-refresh", size="sm")
                        news_box = gr.Markdown(
                            value="*Click refresh to load the latest AI news.*",
                            elem_id="news-box",
                        )
                        news_refresh.click(fetch_news, inputs=[], outputs=[news_box])

        # ── MAIN CHAT ─────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="main-col"):

            gr.Markdown(
                "**ML Research Assistant** &nbsp;&nbsp; `Qwen2.5-7B`",
                elem_id="chat-header",
            )

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                show_label=False,
                height="80vh",
                type="messages",
                show_share_button=False,
                avatar_images=(None, "https://api.dicebear.com/7.x/bottts/svg?seed=mlresearch"),
            )

            with gr.Row(elem_id="input-row"):
                msg = gr.Textbox(
                    placeholder="Ask about papers, models, benchmarks...",
                    show_label=False,
                    scale=9,
                    container=False,
                    elem_id="msg-input",
                    lines=1,
                    max_lines=4,
                )
                send = gr.Button("↑", elem_id="send-btn", scale=1, min_width=42)

    # ── Events ────────────────────────────────────────────────────
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    send.click(chat, [msg, chatbot], [chatbot, msg])
    new_chat_btn.click(new_chat, [chatbot, archive_state], [chatbot, archive_state])

if __name__ == "__main__":
    demo.launch(share=True, css=CSS)
