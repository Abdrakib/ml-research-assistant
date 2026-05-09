"""
ML Research Assistant — app.py
Architecture:
  - Gradio 6 native components for all interaction
  - Custom CSS for dark theme styling
  - gr.HTML() for static decorative elements only (no JS)
  - All tool routing and logic in Python backend
"""

_memory_store = {}

import copy
import json
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
    auto_notice = ""

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

    # Memory short-circuit
    if tool_name == "memory" and tool_result == "Got it! I will remember that":
        reply = "✅ Got it! I will remember that.\n\n*Tool: 🧠 Memory*"
        history = list(history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        return history, ""

    if tool_name == "memory" and tool_result.startswith("Here is what I know"):
        history = list(history or [])
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": tool_result + "\n\n*Tool: 🧠 Memory*"})
        return history, ""

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
        return "No news available right now."
    lines = []
    for item in feed[:6]:
        lines.append(f"**[{item['tag']}]** {item['title']}")
        lines.append(f"*{item['source']} · {item['time']}*")
        lines.append(f"{item['summary']}")
        lines.append("---")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSS — dark theme
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #13131a !important;
    font-family: 'Inter', sans-serif !important;
    color: #e2e8f0 !important;
}

footer { display: none !important; }

/* Hide default gradio padding */
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
.main { padding: 0 !important; }
.contain { padding: 0 !important; }
.gap { gap: 0 !important; }

/* Sidebar */
#sidebar-col {
    background: #1a1a24 !important;
    border-right: 0.5px solid #32323f !important;
    padding: 0 !important;
    min-height: 100vh !important;
}

/* Logo area */
#logo-html { padding: 0 !important; }

/* New chat button */
#new-chat-btn {
    background: #13131a !important;
    border: 0.5px solid #32323f !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    margin: 8px !important;
    font-size: 12px !important;
    font-family: 'Inter', sans-serif !important;
}
#new-chat-btn:hover {
    background: #1e1e28 !important;
    color: #e2e8f0 !important;
    border-color: #44445a !important;
}

/* Tabs */
#sidebar-tabs > .tab-nav {
    background: transparent !important;
    border-bottom: 0.5px solid #32323f !important;
    padding: 0 6px !important;
}
#sidebar-tabs > .tab-nav button {
    font-size: 11px !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 7px 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
#sidebar-tabs > .tab-nav button.selected {
    color: #D97706 !important;
    border-bottom-color: #D97706 !important;
    background: transparent !important;
}

/* History buttons */
.hist-btn {
    background: transparent !important;
    border: none !important;
    color: #64748b !important;
    text-align: left !important;
    font-size: 11px !important;
    padding: 6px 10px !important;
    border-radius: 6px !important;
    margin: 1px 6px !important;
    width: calc(100% - 12px) !important;
    font-family: 'Inter', sans-serif !important;
}
.hist-btn:hover {
    background: #1e1e28 !important;
    color: #e2e8f0 !important;
}

/* Chatbot */
#chatbot {
    background: #13131a !important;
    border: none !important;
    flex: 1 !important;
}
#chatbot .bubble-wrap { background: #13131a !important; }

/* User bubble */
#chatbot .message.user {
    background: #1e2040 !important;
    color: #e2e8f0 !important;
    border-radius: 14px 14px 3px 14px !important;
    border: none !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

/* Bot bubble */
#chatbot .message.bot {
    background: #1a1a24 !important;
    color: #e2e8f0 !important;
    border-radius: 3px 14px 14px 14px !important;
    border: 0.5px solid #32323f !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}

/* Input textbox */
#msg-input textarea {
    background: #1a1a24 !important;
    border: 0.5px solid #32323f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
#msg-input textarea:focus {
    border-color: #D97706 !important;
    box-shadow: 0 0 0 2px #D9770620 !important;
}
#msg-input textarea::placeholder { color: #44445a !important; }
#msg-input { background: transparent !important; border: none !important; }

/* Send button — GOLD */
#send-btn {
    background: #D97706 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    min-width: 40px !important;
    max-width: 40px !important;
    height: 40px !important;
    font-size: 16px !important;
    padding: 0 !important;
}
#send-btn:hover {
    background: #B45309 !important;
    transform: scale(1.04) !important;
}

/* News textbox */
#news-box textarea {
    background: #1a1a24 !important;
    border: 0.5px solid #32323f !important;
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
    border-radius: 8px !important;
}
#news-box { background: transparent !important; border: none !important; }

/* News refresh button */
#news-refresh-btn {
    background: #1a1a24 !important;
    border: 0.5px solid #32323f !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    font-size: 11px !important;
}
#news-refresh-btn:hover {
    border-color: #D97706 !important;
    color: #D97706 !important;
}

/* Tool checkboxes */
.tool-check label {
    color: #94a3b8 !important;
    font-size: 11px !important;
    font-family: 'Inter', sans-serif !important;
}
.tool-check input[type=checkbox]:checked { accent-color: #a78bfa !important; }

/* Main column */
#main-col { background: #13131a !important; padding: 0 !important; }

/* Header */
#header-html { border-bottom: 0.5px solid #32323f !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #32323f; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #D97706; }
"""

# ---------------------------------------------------------------------------
# Gradio UI — native components
# ---------------------------------------------------------------------------

LOGO_HTML = """
<div style="padding:14px 12px 10px;border-bottom:0.5px solid #32323f;display:flex;align-items:center;gap:9px;">
  <div style="width:28px;height:28px;border-radius:7px;background:#92600A;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
    <span style="font-size:16px;">🧠</span>
  </div>
  <div>
    <div style="font-size:12px;font-weight:600;color:#f1f5f9;line-height:1.2;">ML Research</div>
    <div style="font-size:9px;color:#92600A;font-weight:500;letter-spacing:.04em;">Assistant</div>
  </div>
</div>
"""

HEADER_HTML = """
<div style="padding:12px 18px;display:flex;align-items:center;justify-content:space-between;">
  <span style="font-size:13px;font-weight:500;color:#f1f5f9;">ML Research Assistant</span>
  <span style="font-size:10px;color:#475569;background:#1a1a24;padding:2px 9px;border-radius:20px;border:0.5px solid #32323f;">Qwen2.5-7B</span>
</div>
"""

TOOLS_LIST = [
    ("arxiv",               "📄 Arxiv"),
    ("papers_with_code",    "💻 Papers w/ Code"),
    ("llm_leaderboard",     "🏆 LLM Leaderboard"),
    ("model_benchmarks",    "📊 Benchmarks"),
    ("search",              "🔍 Web Search"),
    ("paper_summarizer",    "📝 Paper Summarizer"),
    ("huggingface_models",  "🤗 HF Models"),
    ("huggingface_datasets","🗄 HF Datasets"),
    ("ai_news",             "📰 AI News"),
    ("python_packages",     "📦 Python Packages"),
    ("github_trending",     "🔥 GitHub Trending"),
    ("code_generator",      "⚙️ Code Generator"),
    ("weather",             "🌤 Weather"),
    ("deep_search",         "🔬 Deep Search"),
    ("memory",              "🧠 Memory"),
    ("calc",                "🧮 Calculator"),
    ("github",              "🐙 GitHub"),
]

with gr.Blocks(css=CSS, title="ML Research Assistant", fill_height=True) as demo:

    archive_state = gr.State([])

    with gr.Row(equal_height=True):

        # ── SIDEBAR ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220, elem_id="sidebar-col"):

            gr.HTML(LOGO_HTML, elem_id="logo-html")

            new_chat_btn = gr.Button("➕  New chat", elem_id="new-chat-btn", size="sm")

            with gr.Tabs(elem_id="sidebar-tabs"):

                # History tab
                with gr.Tab("💬 History"):
                    @gr.render(inputs=[archive_state])
                    def render_history(archives):
                        for conv in (archives or []):
                            b = gr.Button(
                                (conv.get("title") or "Untitled")[:36],
                                elem_classes=["hist-btn"],
                                size="sm",
                            )
                            b.click(
                                lambda c=conv: c.get("messages") or [],
                                None, chatbot
                            )

                # Tools tab
                with gr.Tab("⚡ Tools"):
                    tool_checkboxes = {}
                    for tool_id, tool_label in TOOLS_LIST:
                        cb = gr.Checkbox(
                            label=tool_label,
                            value=_tool_state.get(tool_id, True),
                            elem_classes=["tool-check"],
                        )
                        tool_checkboxes[tool_id] = cb

                        def make_toggle(tid):
                            def toggle(val):
                                _tool_state[tid] = val
                            return toggle

                        cb.change(make_toggle(tool_id), inputs=[cb], outputs=[])

                # AI News tab
                with gr.Tab("📰 AI News"):
                    news_box = gr.Textbox(
                        label="",
                        value="Click refresh to load news",
                        lines=20,
                        interactive=False,
                        elem_id="news-box",
                        show_label=False,
                    )
                    news_refresh_btn = gr.Button(
                        "🔄 Refresh News",
                        elem_id="news-refresh-btn",
                        size="sm",
                    )
                    news_refresh_btn.click(fetch_news, inputs=[], outputs=[news_box])

        # ── MAIN CHAT ─────────────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="main-col"):

            gr.HTML(HEADER_HTML, elem_id="header-html")

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                show_label=False,
                height="78vh",
                bubble_full_width=False,
                type="messages",
                placeholder="""
                <div style='text-align:center;padding:60px 20px;'>
                  <div style='font-size:40px;margin-bottom:12px;'>🧠</div>
                  <div style='font-size:18px;font-weight:600;color:#f1f5f9;margin-bottom:8px;'>ML Research Assistant</div>
                  <div style='font-size:13px;color:#475569;line-height:1.6;'>Ask about papers, models, benchmarks, code,<br>or fetch the latest AI news.</div>
                </div>
                """,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about papers, models, benchmarks...",
                    show_label=False,
                    scale=9,
                    container=False,
                    elem_id="msg-input",
                    lines=1,
                )
                send = gr.Button("↑", elem_id="send-btn", scale=1)

    # ── Events ───────────────────────────────────────────────────────────
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    send.click(chat, [msg, chatbot], [chatbot, msg])
    new_chat_btn.click(new_chat, [chatbot, archive_state], [chatbot, archive_state])

if __name__ == "__main__":
    demo.launch(share=True)
