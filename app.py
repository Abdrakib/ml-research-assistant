"""
ML Research Assistant — app.py
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
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant",  "content": "✅ Got it!\n\n*Tool: 🧠 Memory*"})
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
    """Returns (HTML cards, headline choices for dropdown)."""
    feed = get_ai_news_feed()
    if not feed:
        return "<p style='color:#b5a99e;font-size:12px;padding:12px;'>No news right now — try again later.</p>", []

    cards = []
    for item in feed[:6]:
        title   = item.get('title', '')
        summary = item.get('summary', '')
        tag     = item.get('tag', '')
        source  = item.get('source', '')
        time    = item.get('time', '')
        url     = item.get('url', '')
        color   = item.get('color', '#b45309')

        # Article link — opens in new tab
        link_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="font-size:10px;color:{color};text-decoration:none;border:0.5px solid {color}44;padding:2px 7px;border-radius:10px;background:{color}11;">Read article ↗</a>' if url else ''

        card = f"""
        <div style="padding:10px 0;border-bottom:1px solid #e8e2d8;">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
            <span style="font-size:9px;padding:1px 6px;border-radius:10px;font-weight:600;
                         background:{color}18;color:{color};border:0.5px solid {color}33;">
              {tag}
            </span>
            <span style="font-size:9px;color:#b5a99e;">{source} · {time}</span>
          </div>
          <div style="font-size:12px;font-weight:500;color:#3d3530;line-height:1.4;margin-bottom:4px;">
            {title}
          </div>
          <div style="font-size:11px;color:#8c7f74;line-height:1.5;margin-bottom:6px;">
            {summary}
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            {link_html}
          </div>
        </div>
        """
        cards.append(card)

    html = '<div style="padding:4px 12px;">' + ''.join(cards) + '</div>'
    headlines = [item.get('title', '') for item in feed[:6] if item.get('title')]
    return html, gr.update(choices=headlines, value=None, visible=True)



# ---------------------------------------------------------------------------
# CSS — colors and sidebar ONLY. Nothing on textarea/input/button layout.
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

footer { display: none !important; }

body, .gradio-container {
    background: #f5f0e8 !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }

#sidebar-col {
    background: #ede8df !important;
    border-right: 1px solid #ddd5c8 !important;
    padding: 0 !important;
}

#model-badge p {
    font-size: 10px !important;
    color: #b5a99e !important;
    padding: 4px 14px 10px !important;
    margin: 0 !important;
}

#new-chat-btn {
    margin: 8px 10px !important;
    background: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    color: #7a6e65 !important;
    border-radius: 7px !important;
    width: calc(100% - 20px) !important;
}
#new-chat-btn:hover {
    background: #ece6dc !important;
    color: #3d3530 !important;
}

#sidebar-tabs .tab-nav {
    background: transparent !important;
    border-bottom: 1px solid #e8e2d8 !important;
}
#sidebar-tabs .tab-nav button {
    color: #b5a99e !important;
    font-size: 11px !important;
    background: transparent !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
}
#sidebar-tabs .tab-nav button.selected {
    color: #b45309 !important;
    border-bottom-color: #b45309 !important;
}
#sidebar-tabs .tabitem {
    background: transparent !important;
    border: none !important;
}

.hist-btn button {
    background: transparent !important;
    border: none !important;
    color: #8c7f74 !important;
    font-size: 11px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 5px !important;
    padding: 5px 12px !important;
    width: 100% !important;
}
.hist-btn button:hover {
    background: #e5dfd5 !important;
    color: #3d3530 !important;
}

#tools-tab label { font-size: 11px !important; color: #7a6e65 !important; }
#tools-tab input[type=checkbox] { accent-color: #b45309 !important; }

#news-box p { font-size: 11px !important; color: #8c7f74 !important; }
#news-box strong { color: #3d3530 !important; }
#news-box hr { border-color: #e8e2d8 !important; }

#main-col { background: #f5f0e8 !important; }

#chat-header p {
    font-size: 11px !important;
    color: #b5a99e !important;
    padding: 10px 20px !important;
    margin: 0 !important;
    border-bottom: 1px solid #e8e2d8 !important;
}

#chatbot { background: #f5f0e8 !important; border: none !important; }
#chatbot .bubble-wrap { background: #f5f0e8 !important; }

#chatbot .message.user > div {
    background: #ede8df !important;
    border: 1px solid #ddd5c8 !important;
    color: #3d3530 !important;
    border-radius: 16px 16px 4px 16px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
}

#chatbot .message.bot > div {
    background: transparent !important;
    border: none !important;
    color: #3d3530 !important;
    font-size: 14px !important;
    line-height: 1.8 !important;
}

#send-btn {
    background: #b45309 !important;
    border-color: #b45309 !important;
    color: white !important;
    border-radius: 8px !important;
    min-width: 50px !important;
}
#send-btn:hover { background: #92400e !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #ddd5c8; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #b45309; }

/* News dropdown */
#news-dropdown { font-size: 11px !important; }
#news-dropdown label { font-size: 11px !important; color: #7a6e65 !important; }
#news-dropdown select {
    font-size: 11px !important;
    background: #f5f0e8 !important;
    border-color: #ddd5c8 !important;
    color: #3d3530 !important;
}

/* Ask button */
#news-ask-btn {
    background: #b45309 !important;
    border-color: #b45309 !important;
    color: white !important;
    border-radius: 7px !important;
    font-size: 11px !important;
    width: 100% !important;
    margin-top: 4px !important;
}
#news-ask-btn:hover { background: #92400e !important; }

/* Hide Gradio processing status text */
.eta-bar { display: none !important; }
.progress-text { display: none !important; }
.progress-bar-wrap { display: none !important; }
.generating { display: none !important; }
"""

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="ML Research Assistant") as demo:

    archive_state = gr.State([])

    with gr.Row(equal_height=False):

        # ── SIDEBAR ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=240, elem_id="sidebar-col"):

            gr.HTML("""
            <div style="padding:14px 14px 8px;display:flex;align-items:center;
                        gap:10px;border-bottom:1px solid #e8e2d8;">
              <div style="width:30px;height:30px;border-radius:8px;
                          background:#92600A;display:flex;align-items:center;
                          justify-content:center;flex-shrink:0;font-size:16px;">
                🧠
              </div>
              <div>
                <div style="font-size:13px;font-weight:600;color:#3d3530;
                            font-family:Inter,sans-serif;line-height:1.2;">
                  ML Research
                </div>
                <div style="font-size:10px;color:#b5a99e;
                            font-family:Inter,sans-serif;">
                  Assistant
                </div>
              </div>
            </div>
            """)

            gr.Markdown("Qwen2.5-7B · 17 tools", elem_id="model-badge")

            new_chat_btn = gr.Button("＋  New chat", elem_id="new-chat-btn", size="sm")

            with gr.Tabs(elem_id="sidebar-tabs"):

                with gr.Tab("💬 History"):
                    @gr.render(inputs=[archive_state])
                    def render_history(archives):
                        if not archives:
                            gr.Markdown("*No chats yet*")
                            return
                        for conv in archives:
                            b = gr.Button(
                                "  " + (conv.get("title") or "Untitled")[:30],
                                elem_classes=["hist-btn"],
                                size="sm",
                            )
                            b.click(
                                lambda c=conv: c.get("messages") or [],
                                inputs=None,
                                outputs=chatbot,
                            )

                with gr.Tab("⚡ Tools", elem_id="tools-tab"):
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

                with gr.Tab("📰 AI News"):
                    news_refresh = gr.Button("🔄 Refresh", size="sm")

                    news_box = gr.HTML(
                        value="<p style='color:#b5a99e;font-size:12px;padding:12px;'>Click Refresh to load the latest AI news.</p>",
                        elem_id="news-box",
                    )

                    # Dropdown to pick a headline and ask the assistant
                    news_dropdown = gr.Dropdown(
                        choices=[],
                        label="Ask assistant about:",
                        interactive=True,
                        visible=False,
                        elem_id="news-dropdown",
                    )
                    news_ask_btn = gr.Button(
                        "💬 Ask assistant",
                        size="sm",
                        visible=False,
                        elem_id="news-ask-btn",
                    )

                    # Refresh: update HTML cards AND populate dropdown
                    news_refresh.click(
                        fetch_news,
                        outputs=[news_box, news_dropdown],
                    )
                    # Show ask button when dropdown is visible after refresh
                    news_refresh.click(
                        lambda: gr.update(visible=True),
                        outputs=news_ask_btn,
                    )
                    # news_ask_btn wired below after msg is defined

        # ── MAIN CHAT ─────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="main-col"):

            gr.Markdown(
                "ML Research Assistant · Qwen2.5-7B",
                elem_id="chat-header",
            )

            chatbot = gr.Chatbot(
                elem_id="chatbot",
                label="",
                show_label=False,
                height=600,
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about papers, models, benchmarks…",
                    show_label=False,
                    scale=9,
                    container=False,
                    lines=1,
                    max_lines=4,
                )
                send = gr.Button("↑", elem_id="send-btn", scale=1, min_width=50)

    # ── Events ────────────────────────────────────────────────────
    msg.submit(chat, [msg, chatbot], [chatbot, msg])
    send.click(chat, [msg, chatbot], [chatbot, msg])
    new_chat_btn.click(new_chat, [chatbot, archive_state], [chatbot, archive_state])

    # News ask button — wired here because msg is defined in main col
    news_ask_btn.click(
        lambda h: f'Tell me more about this AI news: "{h}"' if h else "",
        inputs=news_dropdown,
        outputs=msg,
    )

if __name__ == "__main__":
    demo.launch(share=True, css=CSS)
