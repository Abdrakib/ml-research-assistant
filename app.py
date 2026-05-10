"""
ML Research Assistant — app.py
Warm off-white / parchment theme. Paper-like, editorial, calm.
Gradio-native only. Backend logic unchanged.
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

# ── Tool state ────────────────────────────────────────────────────────────────

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

# ── Backend (unchanged) ───────────────────────────────────────────────────────

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
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant",  "content": "✅ Got it! I will remember that.\n\n*Tool: 🧠 Memory*"})
        return history, ""

    if tool_name == "memory" and tool_result.startswith("Here is what I know"):
        history = list(history or [])
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant",  "content": tool_result + "\n\n*Tool: 🧠 Memory*"})
        return history, ""

    auto_notice = ""
    if routed.get("auto_enabled") and tool_name not in (None, "none"):
        auto_notice = build_auto_enable_notice(tool_name)

    full_prompt = build_prompt(user_message, tool_result, get_memory_context())
    reply       = generate_response(full_prompt)

    if auto_notice:
        reply = f"{auto_notice}\n\n{reply}"

    reply += f"\n\n*Tool: {_TOOL_LABELS.get(tool_name, 'No tool')}*"

    history = list(history or [])
    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})
    return history, ""


def new_chat(history, archives):
    arch = list(archives or [])
    if history:
        arch.insert(0, {
            "id":       uuid.uuid4().hex,
            "title":    _title_from(history),
            "messages": copy.deepcopy(history),
        })
    return [], arch


def fetch_news():
    feed = get_ai_news_feed()
    if not feed:
        return "*No news right now — try again later.*"
    lines = ["#### Latest AI News\n"]
    for item in feed[:6]:
        lines.append(f"**{item['title']}**")
        lines.append(f"*{item['tag']} · {item['source']} · {item['time']}*\n")
        lines.append(f"{item['summary']}\n")
        lines.append("---")
    return "\n".join(lines)



# ── CSS — paper-like, flat, editorial ────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

/* Force Inter font everywhere */
body, .gradio-container, button, input, textarea, select {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Base colors replacing gr.themes ───────────────────────── */
body { background: #f5f0e8 !important; }
.gradio-container { background: #f5f0e8 !important; max-width: 100% !important; padding: 0 !important; }
.gradio-container * { color: #3d3530; }

/* Inputs */
input, textarea {
    background: #faf7f2 !important;
    border: 1px solid #ddd5c8 !important;
    color: #3d3530 !important;
}
input:focus, textarea:focus {
    border-color: #b45309 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(180,83,9,0.08) !important;
}

/* Primary buttons */
.gr-button-primary, button.primary {
    background: #b45309 !important;
    border-color: #b45309 !important;
    color: white !important;
}

/* Secondary buttons */
.gr-button-secondary, button.secondary {
    background: #ede8df !important;
    border-color: #ddd5c8 !important;
    color: #5c4f44 !important;
}

/* Panels */
.gr-panel { background: #f5f0e8 !important; border-color: #ddd5c8 !important; }

/* ── Base ───────────────────────────────────────────────────── */
footer { display: none !important; }
.gradio-container {
    max-width: 100% !important;
    padding: 0 !important;
    background: #f5f0e8 !important;
}
body { background: #f5f0e8 !important; }

/* ── Page row ────────────────────────────────────────────────── */
#page-row { gap: 0 !important; min-height: 100vh; }

/* ── Sidebar — quiet, receding ───────────────────────────────── */
#sidebar-col {
    background: #ede8df !important;
    border-right: 1px solid #e8e2d8 !important;
    padding: 0 !important;
    min-height: 100vh !important;
    max-width: 250px !important;
}

/* App name */
#app-name {
    padding: 0 !important;
    margin: 0 !important;
    border-bottom: 1px solid #ddd5c8 !important;
}
#app-name p {
    padding: 16px 16px 12px !important;
    margin: 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #3d3530 !important;
    letter-spacing: -0.01em !important;
}

/* Model badge in sidebar */
#model-badge {
    padding: 0 !important;
    margin: 0 !important;
}
#model-badge p {
    padding: 0 16px 10px !important;
    margin: 0 !important;
    font-size: 10px !important;
    color: #c2b8ae !important;
    font-weight: 400 !important;
    letter-spacing: 0.02em !important;
    border-bottom: 1px solid #ddd5c8 !important;
}

/* Sidebar top row */
#sidebar-top-row {
    padding: 8px 10px 4px !important;
    gap: 6px !important;
    margin: 0 !important;
}

/* Toggle button */
#toggle-btn {
    min-width: 32px !important;
    max-width: 32px !important;
    padding: 6px 0 !important;
    background: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    color: #b5a99e !important;
    border-radius: 7px !important;
    font-size: 11px !important;
    box-shadow: none !important;
}
#toggle-btn:hover {
    background: #ece6dc !important;
    color: #7a6e65 !important;
}

/* New chat button */
#new-chat-btn {
    margin: 0 !important;
    width: 100% !important;
    border-radius: 7px !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
    background: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    color: #7a6e65 !important;
    justify-content: flex-start !important;
    gap: 6px !important;
    box-shadow: none !important;
}
#new-chat-btn:hover {
    background: #ece6dc !important;
    color: #3d3530 !important;
    border-color: #cfc7bb !important;
}

/* Sidebar tabs */
#sidebar-tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #e8e2d8 !important;
    padding: 0 10px !important;
    gap: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}
#sidebar-tabs > .tab-nav > button {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    color: #b5a99e !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    padding: 8px 8px !important;
    margin: 0 !important;
    transition: color 0.15s !important;
    box-shadow: none !important;
}
#sidebar-tabs > .tab-nav > button.selected {
    color: #b45309 !important;
    border-bottom-color: #b45309 !important;
    background: transparent !important;
}
#sidebar-tabs .tabitem {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
    box-shadow: none !important;
}

/* History items */
.hist-btn > button {
    background: transparent !important;
    border: none !important;
    color: #8c7f74 !important;
    font-size: 11px !important;
    padding: 5px 14px !important;
    border-radius: 5px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    box-shadow: none !important;
    transition: all 0.12s !important;
}
.hist-btn > button:hover {
    background: #e8e2d8 !important;
    color: #3d3530 !important;
}

/* Tool checkboxes */
#tools-tab label {
    font-size: 11px !important;
    color: #7a6e65 !important;
    gap: 8px !important;
    padding: 4px 14px !important;
    border-radius: 5px !important;
}
#tools-tab label:hover { background: #e8e2d8 !important; }
#tools-tab input[type=checkbox] { accent-color: #b45309 !important; }

/* News */
#news-tab { padding: 0 10px !important; }
#news-refresh-btn {
    font-size: 11px !important;
    margin-bottom: 8px !important;
    width: 100% !important;
    background: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    color: #7a6e65 !important;
    box-shadow: none !important;
}
#news-refresh-btn:hover { color: #b45309 !important; border-color: #b45309 !important; }
#news-box { background: transparent !important; border: none !important; }
#news-box p, #news-box li {
    font-size: 11px !important;
    color: #8c7f74 !important;
    line-height: 1.6 !important;
}
#news-box strong { color: #3d3530 !important; font-size: 11px !important; }
#news-box hr { border-color: #ddd5c8 !important; margin: 8px 0 !important; }

/* ── Main area ───────────────────────────────────────────────── */
#main-col {
    background: #f5f0e8 !important;
    padding: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 100vh !important;
}

/* Header — just a thin line, minimal text */
#chat-header {
    padding: 0 !important;
    margin: 0 !important;
    background: #f5f0e8 !important;
    border-bottom: 1px solid #ddd5c8 !important;
}
#chat-header p {
    padding: 10px 24px !important;
    margin: 0 !important;
    font-size: 11px !important;
    color: #b5a99e !important;
    font-weight: 400 !important;
    letter-spacing: 0.03em !important;
}

/* Centered chat column */
#chat-center {
    flex: 1 !important;
    align-items: center !important;
    padding: 0 !important;
}

/* Chatbot */
#chatbot {
    width: 100% !important;
    max-width: 800px !important;
    margin: 0 auto !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    flex: 1 !important;
    overflow-y: auto !important;
}
#chatbot .bubble-wrap {
    background: transparent !important;
    padding: 32px 0 16px !important;
    gap: 8px !important;
}

/* User messages — warm parchment bubble */
#chatbot .message.user { justify-content: flex-end !important; }
#chatbot .message.user > div {
    background: #ede8df !important;
    color: #3d3530 !important;
    border-radius: 16px 16px 4px 16px !important;
    border: 1px solid #ddd5c8 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    padding: 10px 16px !important;
    max-width: 78% !important;
    box-shadow: none !important;
}

/* Assistant messages — pure text on page, no bubble */
#chatbot .message.bot > div {
    background: transparent !important;
    color: #3d3530 !important;
    border: none !important;
    border-radius: 0 !important;
    font-size: 14px !important;
    line-height: 1.85 !important;
    padding: 6px 0 !important;
    max-width: 100% !important;
    box-shadow: none !important;
}

/* Bot avatar — small, warm */
#chatbot .message.bot .avatar-container {
    width: 26px !important;
    height: 26px !important;
    min-width: 26px !important;
    border-radius: 50% !important;
    background: #ede8df !important;
    border: 1px solid #ddd5c8 !important;
    overflow: hidden !important;
}

/* Remove heavy avatar panel chrome */
#chatbot .message { padding: 4px 0 !important; }
#chatbot .message .message-buttons { opacity: 0 !important; transition: opacity 0.2s !important; }
#chatbot .message:hover .message-buttons { opacity: 1 !important; }

/* ── Composer — integrated rounded bar ───────────────────────── */
#composer-row {
    width: 100% !important;
    max-width: 800px !important;
    margin: 0 auto !important;
    padding: 6px 6px 6px 16px !important;
    margin-bottom: 20px !important;
    align-items: flex-end !important;
    gap: 0 !important;
    background: #faf7f2 !important;
    border: 1px solid #ddd5c8 !important;
    border-radius: 14px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: none !important;
    position: relative !important;
    pointer-events: auto !important;
}
#composer-row:focus-within {
    border-color: #b45309 !important;
    box-shadow: 0 0 0 3px rgba(180,83,9,0.08) !important;
}
/* Ensure textbox inside is fully clickable */
#composer-row > div {
    position: relative !important;
    z-index: 1 !important;
}

/* Textbox inside composer */
#msg-input {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
#msg-input textarea {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    color: #3d3530 !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 4px 0 !important;
    resize: none !important;
    outline: none !important;
    min-height: 28px !important;
    font-family: 'Inter', sans-serif !important;
    cursor: text !important;
    pointer-events: auto !important;
    position: relative !important;
    z-index: 20 !important;
}
#msg-input textarea::placeholder { color: #b5a99e !important; }
#msg-input textarea:focus { outline: none !important; box-shadow: none !important; }
#msg-input label { display: none !important; }
#msg-input .wrap {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    cursor: text !important;
    pointer-events: auto !important;
    z-index: 15 !important;
    position: relative !important;
}
#msg-input {
    pointer-events: auto !important;
    position: relative !important;
    z-index: 15 !important;
}

/* Send button — terracotta, inside the composer bar */
#send-btn {
    background: #b45309 !important;
    border: none !important;
    border-radius: 9px !important;
    color: #fff !important;
    min-width: 34px !important;
    max-width: 34px !important;
    height: 34px !important;
    font-size: 16px !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
    align-self: flex-end !important;
    box-shadow: none !important;
    transition: background 0.15s, transform 0.1s !important;
    line-height: 1 !important;
}
#send-btn:hover {
    background: #92400e !important;
    transform: scale(1.05) !important;
}

/* ── Scrollbar — minimal ─────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #ddd5c8; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #b45309; }
"""

# ── Gradio layout ─────────────────────────────────────────────────────────────

with gr.Blocks(title="ML Research Assistant", fill_height=True, css=CSS) as demo:

    archive_state = gr.State([])

    with gr.Row(elem_id="page-row", equal_height=False):

        # ── SIDEBAR ──────────────────────────────────────────────
        with gr.Column(scale=0, min_width=240, elem_id="sidebar-col") as sidebar_col:

            gr.HTML('''
            <div style="padding:16px 16px 10px;border-bottom:1px solid #e8e2d8;display:flex;align-items:center;gap:10px;">
              <div style="width:30px;height:30px;border-radius:8px;background:#92600A;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <span style="font-size:17px;line-height:1;">🧠</span>
              </div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#3d3530;line-height:1.2;">ML Research Assistant</div>
              </div>
            </div>
            ''', elem_id="app-name")
            gr.Markdown("Qwen2.5-7B · 17 tools", elem_id="model-badge")

            with gr.Row(elem_id="sidebar-top-row"):
                new_chat_btn = gr.Button(
                    "＋  New chat",
                    elem_id="new-chat-btn",
                    size="sm",
                    scale=4,
                )
                toggle_btn = gr.Button(
                    "◀",
                    elem_id="toggle-btn",
                    size="sm",
                    scale=1,
                )

            with gr.Tabs(elem_id="sidebar-tabs"):

                with gr.Tab("History"):
                    @gr.render(inputs=[archive_state])
                    def render_history(archives):
                        if not archives:
                            gr.Markdown("*No chats yet*")
                            return
                        for conv in archives:
                            b = gr.Button(
                                "  " + (conv.get("title") or "Untitled")[:32],
                                elem_classes=["hist-btn"],
                                size="sm",
                            )
                            b.click(
                                lambda c=conv: c.get("messages") or [],
                                inputs=None,
                                outputs=chatbot,
                            )

                with gr.Tab("Tools", elem_id="tools-tab"):
                    for tool_id, tool_label in TOOLS_LIST:
                        cb = gr.Checkbox(
                            label=tool_label,
                            value=_tool_state.get(tool_id, True),
                            container=False,
                        )
                        def _make_toggle(tid):
                            def _toggle(v): _tool_state[tid] = v
                            return _toggle
                        cb.change(_make_toggle(tool_id), inputs=cb, outputs=[])

                with gr.Tab("AI News", elem_id="news-tab"):
                    news_refresh = gr.Button(
                        "↻  Refresh",
                        size="sm",
                        elem_id="news-refresh-btn",
                    )
                    news_box = gr.Markdown(
                        "*Click Refresh to load the latest AI news.*",
                        elem_id="news-box",
                    )
                    news_refresh.click(fetch_news, outputs=news_box)

        # ── MAIN CHAT ─────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="main-col"):

            # Minimal header — just model info, very quiet
            gr.Markdown(
                "ML Research Assistant",
                elem_id="chat-header",
            )

            with gr.Column(elem_id="chat-center"):

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="",
                    show_label=False,
                    height=None,
                    type="messages",
                    show_copy_button=False,
                    allow_tags=False,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts-neutral/svg?seed=ml&backgroundColor=ede8df",
                    ),
                )

                with gr.Row(elem_id="composer-row"):
                    msg = gr.Textbox(
                        placeholder="Ask about papers, models, benchmarks, or news…",
                        show_label=False,
                        scale=1,
                        container=False,
                        elem_id="msg-input",
                        lines=1,
                        max_lines=6,
                    )
                    send = gr.Button(
                        "↑",
                        elem_id="send-btn",
                        scale=0,
                        min_width=34,
                    )

    # ── Events (unchanged) ────────────────────────────────────────
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    send.click(chat, [msg, chatbot], [chatbot, msg])
    new_chat_btn.click(new_chat, [chatbot, archive_state], [chatbot, archive_state])

    # Toggle sidebar visibility
    sidebar_visible = gr.State(True)

    def toggle_sidebar(visible):
        new_visible = not visible
        label = "▶" if not new_visible else "◀"
        return (
            gr.update(visible=new_visible),
            gr.update(value=label),
            new_visible,
        )

    toggle_btn.click(
        toggle_sidebar,
        inputs=[sidebar_visible],
        outputs=[sidebar_col, toggle_btn, sidebar_visible],
    )

if __name__ == "__main__":
    demo.launch(share=True)
