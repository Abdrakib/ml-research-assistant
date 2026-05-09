"""
ML Research Assistant — app.py
Architecture:
  Backend : Gradio gr.Blocks (ZeroGPU compatible)
  Frontend: Custom HTML/CSS/JS injected via gr.HTML()
  Bridge  : Hidden Gradio components + JS fetch to /gradio_api/run/<api_name>
"""

_memory_store = {}

import copy
import json
import uuid
import gradio as gr

from model import generate_response
from prompt_builder import build_auto_enable_notice, build_prompt
from router import route
from tools.calculator        import run_calculator
from tools.deep_search       import run_deep_search
from tools.github            import run_github
from tools.memory            import get_memory_context, run_memory
from tools.search            import run_search
from tools.weather           import run_weather
from tools.arxiv_search      import run_arxiv_search
from tools.papers_with_code  import run_papers_with_code
from tools.huggingface_models   import run_huggingface_models
from tools.huggingface_datasets import run_huggingface_datasets
from tools.code_generator    import run_code_generator
from tools.ai_news           import run_ai_news, get_ai_news_feed
from tools.python_packages   import run_python_packages
from tools.paper_summarizer  import run_paper_summarizer
from tools.github_trending   import run_github_trending
from tools.model_benchmarks  import run_model_benchmarks
from tools.llm_leaderboard   import run_llm_leaderboard

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
    "weather":              "🌤 Weather",
    "search":               "🔍 Web Search",
    "deep_search":          "🔬 Deep Search",
    "calc":                 "🧮 Calculator",
    "memory":               "🧠 Memory",
    "github":               "🐙 GitHub",
    "arxiv":                "📄 Arxiv",
    "papers_with_code":     "💻 Papers w/ Code",
    "llm_leaderboard":      "🏆 LLM Leaderboard",
    "model_benchmarks":     "📊 Benchmarks",
    "huggingface_models":   "🤗 HF Models",
    "huggingface_datasets": "🗄 HF Datasets",
    "code_generator":       "⚙️ Code Generator",
    "ai_news":              "📰 AI News",
    "python_packages":      "📦 Python Packages",
    "paper_summarizer":     "📝 Paper Summarizer",
    "github_trending":      "🔥 GitHub Trending",
    "none":                 "No tool",
}

# ---------------------------------------------------------------------------
# Backend functions (called by hidden Gradio components)
# ---------------------------------------------------------------------------

def _title_from(hist):
    if not hist:
        return "New chat"
    first = hist[0]
    text = str(first.get("content") or "").strip().replace("\n", " ")
    return (text[:45] + "…") if len(text) > 45 else (text or "New chat")


def chat_fn(user_message, history_json, tool_state_json):
    """
    Main chat function called by hidden Gradio submit button.
    Returns: (history_json, bot_reply, tool_label, auto_activated_flag)
    """
    if not (user_message or "").strip():
        return history_json, "", "", "false"

    history   = json.loads(history_json  or "[]")
    tool_state = json.loads(tool_state_json or "{}")

    # merge with server state for tools not in client state
    active = {**_tool_state, **tool_state}
    routed = route(user_message, active)

    if routed.get("auto_enabled"):
        t = routed.get("tool")
        if t and t != "none":
            _tool_state[t] = True

    tool_name   = routed["tool"]
    tool_result = ""
    auto_flag   = "true" if routed.get("auto_enabled") else "false"

    # Run the matched tool
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
        reply = "✅ Got it! I will remember that."
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": reply})
        return json.dumps(history), reply, _TOOL_LABELS["memory"], auto_flag

    if tool_name == "memory" and tool_result.startswith("Here is what I know"):
        history.append({"role": "user",      "content": user_message})
        history.append({"role": "assistant", "content": tool_result})
        return json.dumps(history), tool_result, _TOOL_LABELS["memory"], auto_flag

    full_prompt = build_prompt(user_message, tool_result, get_memory_context())
    reply       = generate_response(full_prompt)
    tool_label  = _TOOL_LABELS.get(tool_name, "No tool")

    history.append({"role": "user",      "content": user_message})
    history.append({"role": "assistant", "content": reply})

    return json.dumps(history), reply, tool_label, auto_flag


def new_chat_fn(history_json, archives_json):
    """Save current chat to archives and reset."""
    history  = json.loads(history_json  or "[]")
    archives = json.loads(archives_json or "[]")
    if history:
        archives.insert(0, {
            "id":       uuid.uuid4().hex,
            "title":    _title_from(history),
            "messages": history,
        })
    return "[]", json.dumps(archives[:30])  # keep max 30 chats


def load_archive_fn(archives_json, chat_id):
    """Load a specific archived chat by id."""
    archives = json.loads(archives_json or "[]")
    for conv in archives:
        if conv.get("id") == chat_id:
            return json.dumps(conv.get("messages", []))
    return "[]"


def fetch_news_fn():
    """Fetch structured news feed for the sidebar news tab."""
    feed = get_ai_news_feed()
    return json.dumps(feed)


def toggle_tool_fn(tool_name, enabled, tool_state_json):
    """Toggle a tool on/off and return updated state JSON."""
    state = json.loads(tool_state_json or "{}")
    state[tool_name] = bool(enabled)
    _tool_state[tool_name] = bool(enabled)
    return json.dumps(state)


# ---------------------------------------------------------------------------
# Custom UI — full HTML/CSS/JS
# ---------------------------------------------------------------------------

CUSTOM_UI = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/dist/tabler-icons.min.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

#ml-app {
  display: flex; height: 100vh;
  background: #13131a; color: #e2e8f0;
  font-family: 'Inter', sans-serif; font-size: 13px; overflow: hidden;
}

/* Sidebar */
#sidebar {
  width: 220px; min-width: 220px;
  background: #1a1a24;
  border-right: 0.5px solid #32323f;
  display: flex; flex-direction: column; overflow: hidden;
}
#logo {
  padding: 14px 12px 10px;
  border-bottom: 0.5px solid #32323f;
  display: flex; align-items: center; gap: 9px;
}
#logo-badge {
  width: 28px; height: 28px; border-radius: 7px;
  background: #92600A;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
#logo-badge i { font-size: 15px; color: #a78bfa; }
#logo-title { font-size: 12px; font-weight: 600; color: #f1f5f9; }
#logo-sub   { font-size: 9px; color: #92600A; font-weight: 500; letter-spacing:.04em; }

#new-chat-btn {
  margin: 8px; padding: 7px 10px; border-radius: 8px;
  border: 0.5px solid #32323f; background: #13131a; color: #94a3b8;
  font-size: 12px; cursor: pointer;
  display: flex; align-items: center; gap: 6px; transition: all .15s;
  font-family: 'Inter', sans-serif; width: calc(100% - 16px);
}
#new-chat-btn:hover { background: #1e1e28; color: #e2e8f0; border-color: #44445a; }
#new-chat-btn i { font-size: 13px; }

/* Tabs */
#sidebar-tabs { display: flex; border-bottom: 0.5px solid #32323f; padding: 0 6px; }
.s-tab {
  flex: 1; padding: 7px 3px; font-size: 11px; font-weight: 500;
  border: none; background: transparent; cursor: pointer;
  border-bottom: 2px solid transparent; color: #64748b;
  transition: all .15s; font-family: 'Inter', sans-serif;
}
.s-tab.active { color: #D97706; border-bottom-color: #D97706; }
.s-tab i { font-size: 11px; vertical-align: -2px; margin-right: 3px; }

.s-panel { flex: 1; overflow-y: auto; overflow-x: hidden; display: none; flex-direction: column; padding: 4px 0; }
.s-panel.active { display: flex; }
.s-panel::-webkit-scrollbar { width: 3px; }
.s-panel::-webkit-scrollbar-thumb { background: #32323f; border-radius: 2px; }

/* History */
.hist-item {
  padding: 6px 10px; margin: 1px 6px; border-radius: 6px; cursor: pointer;
  font-size: 11px; color: #64748b;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; transition: all .12s;
}
.hist-item:hover  { background: #1e1e28; color: #e2e8f0; }
.hist-item.active { background: #1e1e28; color: #f1f5f9; font-weight: 500; }

/* Tool chips */
.tool-chip {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 8px; margin: 1px 6px; border-radius: 7px;
  cursor: pointer; border: 0.5px solid #32323f; transition: all .15s;
}
.tool-chip.on  { background: #1e2040; border-color: #1e2040; }
.tool-chip.off { background: #13131a; }
.tool-chip i   { font-size: 13px; }
.tool-chip.on  i { color: #a78bfa; }
.tool-chip.off i { color: #475569; }
.chip-label { font-size: 11px; font-weight: 500; flex: 1; }
.tool-chip.on  .chip-label { color: #e2e8f0; }
.tool-chip.off .chip-label { color: #475569; }
.toggle-track {
  width: 26px; height: 14px; border-radius: 7px;
  position: relative; flex-shrink: 0; transition: background .2s;
}
.tool-chip.on  .toggle-track { background: #a78bfa; }
.tool-chip.off .toggle-track { background: #32323f; }
.toggle-track::after {
  content: ''; position: absolute; top: 2px;
  width: 10px; height: 10px; border-radius: 50%;
  background: #fff; transition: left .2s;
}
.tool-chip.on  .toggle-track::after { left: 14px; }
.tool-chip.off .toggle-track::after { left: 2px; }
.tools-hint {
  margin: 8px; padding: 7px 9px; border-radius: 7px;
  background: #13131a; border: 0.5px solid #32323f;
  font-size: 10px; color: #475569; line-height: 1.5;
}
.tools-hint i { color: #a78bfa; vertical-align: -1px; }

/* News */
.news-card {
  margin: 3px 6px; padding: 9px 10px; border-radius: 8px;
  border: 0.5px solid #32323f; background: #13131a;
  cursor: pointer; transition: border-color .15s;
}
.news-card:hover { border-color: #D97706; }
.news-meta { display: flex; align-items: center; gap: 6px; margin-bottom: 5px; }
.news-tag  { font-size: 9px; padding: 1px 6px; border-radius: 10px; font-weight: 600; }
.news-time { font-size: 9px; color: #475569; }
.news-title   { font-size: 11px; font-weight: 500; color: #f1f5f9; line-height: 1.4; margin-bottom: 3px; }
.news-summary { font-size: 10px; color: #64748b; line-height: 1.4; margin-bottom: 5px; }
.news-cta     { font-size: 9px; color: #475569; }
.news-refresh {
  display: flex; align-items: center; justify-content: space-between;
  padding: 6px 10px; font-size: 10px; color: #475569;
  border-bottom: 0.5px solid #32323f; flex-shrink: 0;
}
#refresh-btn {
  width: 24px; height: 24px; border-radius: 5px;
  background: #13131a; border: 0.5px solid #32323f;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: all .15s;
}
#refresh-btn:hover { border-color: #D97706; }
#refresh-btn i { font-size: 12px; color: #94a3b8; }
#refresh-btn.spinning i { animation: spin .6s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }

/* Main */
#main { flex: 1; display: flex; flex-direction: column; min-width: 0; background: #13131a; }
#chat-header {
  padding: 12px 18px; border-bottom: 0.5px solid #32323f;
  display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
}
#chat-header-title { font-size: 13px; font-weight: 500; color: #f1f5f9; }
#model-badge {
  font-size: 10px; color: #475569; background: #1a1a24;
  padding: 2px 9px; border-radius: 20px; border: 0.5px solid #32323f;
}

/* Messages */
#messages {
  flex: 1; overflow-y: auto; padding: 16px;
  display: flex; flex-direction: column; gap: 12px;
}
#messages::-webkit-scrollbar { width: 3px; }
#messages::-webkit-scrollbar-thumb { background: #32323f; border-radius: 2px; }

#welcome {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  flex: 1; gap: 12px; padding: 40px 20px; text-align: center;
}
#welcome-icon {
  width: 52px; height: 52px; border-radius: 14px; background: #92600A;
  display: flex; align-items: center; justify-content: center;
}
#welcome-icon i { font-size: 26px; color: #a78bfa; }
#welcome h2 { font-size: 18px; font-weight: 600; color: #f1f5f9; }
#welcome p  { font-size: 13px; color: #475569; line-height: 1.6; }

.msg-user { display: flex; justify-content: flex-end; }
.msg-user .bubble {
  max-width: 72%; background: #1e2040; color: #e2e8f0;
  border-radius: 14px 14px 3px 14px; padding: 10px 14px;
  font-size: 13px; line-height: 1.6;
}
.msg-bot { display: flex; gap: 9px; align-items: flex-start; }
.bot-avatar {
  width: 26px; height: 26px; min-width: 26px; border-radius: 50%;
  background: #1e2040; display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.bot-avatar i { font-size: 13px; color: #a78bfa; }
.msg-bot .bubble {
  flex: 1; min-width: 0; background: #1a1a24;
  border: 0.5px solid #32323f; border-radius: 3px 14px 14px 14px;
  padding: 10px 14px; font-size: 13px; line-height: 1.6;
  color: #e2e8f0; white-space: pre-wrap; word-break: break-word;
}
.tool-badge { display: inline-block; margin-top: 6px; font-size: 10px; color: #475569; }
.auto-notice { display: flex; justify-content: center; }
.auto-notice span {
  font-size: 10px; color: #a78bfa; background: #a78bfa11;
  padding: 2px 12px; border-radius: 20px; border: 0.5px solid #a78bfa33;
}

/* Typing */
#typing { display: none; gap: 9px; align-items: flex-start; }
#typing .bot-avatar { width:26px;height:26px;min-width:26px;border-radius:50%;background:#1e2040;display:flex;align-items:center;justify-content:center; }
#typing .bot-avatar i { font-size:13px;color:#a78bfa; }
.typing-dots {
  background: #1a1a24; border: 0.5px solid #32323f;
  border-radius: 3px 14px 14px 14px; padding: 12px 16px;
  display: flex; gap: 4px; align-items: center;
}
.dot { width:6px;height:6px;border-radius:50%;background:#475569;animation:pulse 1.2s ease-in-out infinite; }
.dot:nth-child(2){animation-delay:.2s;} .dot:nth-child(3){animation-delay:.4s;}
@keyframes pulse{0%,100%{opacity:.3;transform:scale(.8)}50%{opacity:1;transform:scale(1)}}

/* Input */
#input-area  { border-top: 0.5px solid #32323f; padding: 10px 14px; flex-shrink: 0; }
#input-row   { display: flex; gap: 8px; align-items: flex-end; }
#msg-input {
  flex: 1; background: #1a1a24; border: 0.5px solid #32323f;
  border-radius: 10px; padding: 10px 14px;
  font-size: 13px; color: #e2e8f0; font-family: 'Inter', sans-serif;
  resize: none; outline: none; min-height: 42px; max-height: 120px;
  line-height: 1.5; transition: border-color .2s;
}
#msg-input:focus { border-color: #D97706; }
#msg-input::placeholder { color: #44445a; }
#send-btn {
  width: 36px; height: 36px; min-width: 36px; border-radius: 8px;
  background: #D97706; border: none; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .15s; flex-shrink: 0;
}
#send-btn:hover   { background: #B45309; transform: scale(1.04); }
#send-btn:disabled{ background: #32323f; cursor: not-allowed; transform: none; }
#send-btn i { font-size: 16px; color: #fff; }
#input-hint { margin-top: 5px; font-size: 10px; color: #32323f; text-align: center; }
</style>

<!-- HTML Structure -->
<div id="ml-app">
  <div id="sidebar">
    <div id="logo">
      <div id="logo-badge"><i class="ti ti-brain"></i></div>
      <div>
        <div id="logo-title">ML Research</div>
        <div id="logo-sub">Assistant</div>
      </div>
    </div>
    <button id="new-chat-btn"><i class="ti ti-plus"></i> New chat</button>
    <div id="sidebar-tabs">
      <button class="s-tab active" id="tab-hist"><i class="ti ti-history"></i>History</button>
      <button class="s-tab" id="tab-tools"><i class="ti ti-bolt"></i>Tools <span id="tool-count" style="font-size:9px;padding:1px 4px;border-radius:8px;background:#a78bfa22;color:#a78bfa;margin-left:2px;"></span></button>
      <button class="s-tab" id="tab-news"><i class="ti ti-news"></i>AI News <span style="font-size:9px;padding:1px 4px;border-radius:8px;background:#ef444422;color:#ef4444;margin-left:2px;">live</span></button>
    </div>
    <div class="s-panel active" id="panel-hist">
      <div id="hist-list"><div style="padding:20px 12px;font-size:11px;color:#32323f;text-align:center;">No chats yet</div></div>
    </div>
    <div class="s-panel" id="panel-tools">
      <div id="tools-list"></div>
      <div class="tools-hint"><i class="ti ti-bolt"></i> Active tools fire automatically. Toggle off to disable.</div>
    </div>
    <div class="s-panel" id="panel-news">
      <div class="news-refresh">
        <span id="news-updated">Loading...</span>
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="font-size:9px;color:#32323f;">auto·30m</span>
          <div id="refresh-btn"><i class="ti ti-refresh"></i></div>
        </div>
      </div>
      <div id="news-list" style="flex:1;overflow-y:auto;padding:4px 0;">
        <div style="padding:20px 12px;font-size:11px;color:#32323f;text-align:center;">Loading news...</div>
      </div>
    </div>
  </div>

  <div id="main">
    <div id="chat-header">
      <span id="chat-header-title">ML Research Assistant</span>
      <span id="model-badge">Qwen2.5-7B</span>
    </div>
    <div id="messages">
      <div id="welcome">
        <div id="welcome-icon"><i class="ti ti-brain"></i></div>
        <h2>ML Research Assistant</h2>
        <p>Ask about papers, models, benchmarks, code,<br>or let me fetch the latest AI news for you.</p>
      </div>
    </div>
    <div id="typing">
      <div class="bot-avatar"><i class="ti ti-brain"></i></div>
      <div class="typing-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
    </div>
    <div id="input-area">
      <div id="input-row">
        <textarea id="msg-input" placeholder="Ask about papers, models, benchmarks..." rows="1"></textarea>
        <button id="send-btn"><i class="ti ti-arrow-up"></i></button>
      </div>
      <div id="input-hint">Tools auto-activate · Click news cards to ask about them</div>
    </div>
  </div>
</div>

<script>
// ── All JS in one block, runs after DOM is ready ──────────────
(function() {

var _history = [], _archives = [], _toolState = {}, _busy = false, _currentChatId = null;

var TOOLS = [
  {id:'arxiv',               icon:'ti-file-text',      label:'Arxiv',           on:true},
  {id:'papers_with_code',    icon:'ti-code',            label:'Papers w/ Code',  on:true},
  {id:'llm_leaderboard',     icon:'ti-trophy',          label:'LLM Leaderboard', on:true},
  {id:'model_benchmarks',    icon:'ti-chart-bar',       label:'Benchmarks',      on:true},
  {id:'search',              icon:'ti-search',          label:'Web Search',      on:true},
  {id:'paper_summarizer',    icon:'ti-file-description',label:'Paper Summarizer',on:true},
  {id:'huggingface_models',  icon:'ti-robot',           label:'HF Models',       on:true},
  {id:'huggingface_datasets',icon:'ti-database',        label:'HF Datasets',     on:true},
  {id:'ai_news',             icon:'ti-news',            label:'AI News',         on:true},
  {id:'python_packages',     icon:'ti-package',         label:'Python Packages', on:true},
  {id:'github_trending',     icon:'ti-trending-up',     label:'GitHub Trending', on:true},
  {id:'code_generator',      icon:'ti-terminal',        label:'Code Generator',  on:true},
  {id:'weather',             icon:'ti-cloud',           label:'Weather',         on:true},
  {id:'deep_search',         icon:'ti-zoom-in',         label:'Deep Search',     on:true},
  {id:'memory',              icon:'ti-brain',           label:'Memory',          on:true},
  {id:'calc',                icon:'ti-calculator',      label:'Calculator',      on:true},
  {id:'github',              icon:'ti-brand-github',    label:'GitHub',          on:false},
];

function getSessionHash() {
  if (!window._mlSessionHash) window._mlSessionHash = Math.random().toString(36).slice(2);
  return window._mlSessionHash;
}

function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Tab switching ─────────────────────────────────────────────
function switchTab(tab) {
  ['hist','tools','news'].forEach(function(t) {
    document.getElementById('tab-'+t).classList.toggle('active', t===tab);
    document.getElementById('panel-'+t).classList.toggle('active', t===tab);
  });
}

// ── Tools ─────────────────────────────────────────────────────
function renderTools() {
  var list = document.getElementById('tools-list');
  list.innerHTML = '';
  TOOLS.forEach(function(tool) {
    var on = _toolState[tool.id] !== false;
    var div = document.createElement('div');
    div.className = 'tool-chip ' + (on?'on':'off');
    div.dataset.tool = tool.id;
    div.innerHTML = '<i class="ti '+tool.icon+'"></i><span class="chip-label">'+tool.label+'</span><div class="toggle-track"></div>';
    div.addEventListener('click', function() { toggleTool(tool.id, div); });
    list.appendChild(div);
  });
}

function toggleTool(id, el) {
  var isOn = el.classList.contains('on');
  el.classList.toggle('on', !isOn);
  el.classList.toggle('off', isOn);
  _toolState[id] = !isOn;
  updateToolCount();
}

function updateToolCount() {
  var c = Object.values(_toolState).filter(Boolean).length;
  document.getElementById('tool-count').textContent = c;
}

// ── Chat ──────────────────────────────────────────────────────
function scrollToBottom() {
  var m = document.getElementById('messages');
  m.scrollTop = m.scrollHeight;
}

function setBusy(busy) {
  _busy = busy;
  document.getElementById('send-btn').disabled = busy;
  document.getElementById('typing').style.display = busy ? 'flex' : 'none';
  if (busy) scrollToBottom();
}

function appendBubble(role, text, toolLabel) {
  var msgs   = document.getElementById('messages');
  var typing = document.getElementById('typing');
  var div    = document.createElement('div');
  if (role === 'user') {
    div.className = 'msg-user';
    div.innerHTML = '<div class="bubble">'+escHtml(text)+'</div>';
  } else {
    var badge = toolLabel ? '<div class="tool-badge"><i class="ti ti-tool"></i> '+escHtml(toolLabel)+'</div>' : '';
    div.className = 'msg-bot';
    div.innerHTML = '<div class="bot-avatar"><i class="ti ti-brain"></i></div><div class="bubble">'+escHtml(text)+badge+'</div>';
  }
  msgs.insertBefore(div, typing);
  scrollToBottom();
}

function appendAutoNotice(label) {
  var msgs   = document.getElementById('messages');
  var typing = document.getElementById('typing');
  var div    = document.createElement('div');
  div.className = 'auto-notice';
  div.innerHTML = '<span><i class="ti ti-bolt" style="font-size:10px;vertical-align:-1px;"></i> '+escHtml(label)+' auto-activated</span>';
  msgs.insertBefore(div, typing);
}

function appendError(msg) {
  var msgs   = document.getElementById('messages');
  var typing = document.getElementById('typing');
  var div    = document.createElement('div');
  div.className = 'msg-bot';
  div.innerHTML = '<div class="bot-avatar"><i class="ti ti-brain"></i></div><div class="bubble" style="color:#ef4444;">'+escHtml(msg)+'</div>';
  msgs.insertBefore(div, typing);
}

function sendMessage() {
  if (_busy) return;
  var input = document.getElementById('msg-input');
  var msg   = (input.value||'').trim();
  if (!msg) return;

  var welcome = document.getElementById('welcome');
  if (welcome) welcome.style.display = 'none';

  appendBubble('user', msg);
  input.value = '';
  input.style.height = 'auto';
  setBusy(true);

  var histJson  = JSON.stringify(_history);
  var stateJson = JSON.stringify(_toolState);

  fetch('/gradio_api/run/chat', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      data: [msg, histJson, stateJson],
      session_hash: getSessionHash(),
    })
  })
  .then(function(r){return r.json();})
  .then(function(res){
    setBusy(false);
    if (res.error) { appendError(res.error); return; }
    var data      = res.data;
    var histJson  = data[0];
    var botReply  = data[1];
    var toolLabel = data[2];
    var autoFlag  = data[3];
    _history = JSON.parse(histJson||'[]');
    if (autoFlag==='true') appendAutoNotice(toolLabel);
    appendBubble('bot', botReply, toolLabel);
    scrollToBottom();
  })
  .catch(function(err){
    setBusy(false);
    appendError('Connection error — please try again.');
    console.error(err);
  });
}

// ── New chat ──────────────────────────────────────────────────
function newChat() {
  if (_history.length > 0) {
    _archives.unshift({
      id: Math.random().toString(36).slice(2),
      title: (_history[0]&&_history[0].content||'Chat').slice(0,45),
      messages: JSON.parse(JSON.stringify(_history)),
    });
    if (_archives.length > 30) _archives.pop();
    renderHistory();
  }
  _history = [];
  _currentChatId = null;
  var msgs = document.getElementById('messages');
  msgs.innerHTML =
    '<div id="welcome" style="display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:12px;padding:40px 20px;text-align:center;">' +
    '<div id="welcome-icon" style="width:52px;height:52px;border-radius:14px;background:#92600A;display:flex;align-items:center;justify-content:center;"><i class="ti ti-brain" style="font-size:26px;color:#a78bfa;"></i></div>' +
    '<h2 style="font-size:18px;font-weight:600;color:#f1f5f9;">ML Research Assistant</h2>' +
    '<p style="font-size:13px;color:#475569;line-height:1.6;">Ask about papers, models, benchmarks, code,<br>or let me fetch the latest AI news for you.</p>' +
    '</div>';
  msgs.appendChild(document.getElementById('typing'));
}

// ── History ───────────────────────────────────────────────────
function renderHistory() {
  var list = document.getElementById('hist-list');
  if (!_archives.length) {
    list.innerHTML = '<div style="padding:20px 12px;font-size:11px;color:#32323f;text-align:center;">No chats yet</div>';
    return;
  }
  list.innerHTML = '';
  _archives.forEach(function(conv) {
    var div = document.createElement('div');
    div.className = 'hist-item'+(conv.id===_currentChatId?' active':'');
    div.textContent = conv.title||'Untitled';
    div.addEventListener('click', function(){ loadChat(conv.id); });
    list.appendChild(div);
  });
}

function loadChat(chatId) {
  var conv = _archives.find(function(c){return c.id===chatId;});
  if (!conv) return;
  _currentChatId = chatId;
  _history = JSON.parse(JSON.stringify(conv.messages||[]));
  var msgs   = document.getElementById('messages');
  var typing = document.getElementById('typing');
  msgs.innerHTML = '';
  _history.forEach(function(m){ appendBubble(m.role==='user'?'user':'bot', m.content); });
  msgs.appendChild(typing);
  renderHistory();
  scrollToBottom();
  switchTab('hist');
}

// ── News ──────────────────────────────────────────────────────
function fetchNews() {
  var btn = document.getElementById('refresh-btn');
  btn.classList.add('spinning');
  fetch('/gradio_api/run/fetch_news', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      data: [],
      session_hash: getSessionHash(),
    })
  })
  .then(function(r){return r.json();})
  .then(function(res){
    btn.classList.remove('spinning');
    if (res.error) return;
    var feed = JSON.parse(res.data[0]||'[]');
    renderNews(feed);
    document.getElementById('news-updated').textContent = 'updated just now';
  })
  .catch(function(){
    btn.classList.remove('spinning');
    document.getElementById('news-updated').textContent = 'update failed';
  });
}

function renderNews(feed) {
  var list = document.getElementById('news-list');
  if (!feed||!feed.length) {
    list.innerHTML = '<div style="padding:20px 12px;font-size:11px;color:#32323f;text-align:center;">No news available</div>';
    return;
  }
  list.innerHTML = '';
  feed.forEach(function(card) {
    var div = document.createElement('div');
    div.className = 'news-card';
    div.innerHTML =
      '<div class="news-meta">' +
        '<span class="news-tag" style="background:'+card.color+'22;color:'+card.color+';border:0.5px solid '+card.color+'44;">'+escHtml(card.tag)+'</span>' +
        '<span class="news-time">'+escHtml(card.time)+' · '+escHtml(card.source)+'</span>' +
      '</div>' +
      '<div class="news-title">'+escHtml(card.title)+'</div>' +
      '<div class="news-summary">'+escHtml(card.summary)+'</div>' +
      '<div class="news-cta"><i class="ti ti-message"></i> click to ask assistant</div>';
    div.addEventListener('click', function(){ pasteNews(card.title); });
    list.appendChild(div);
  });
}

function pasteNews(headline) {
  var input = document.getElementById('msg-input');
  input.value = '"'+headline+'" — can you tell me more about this?';
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight,120)+'px';
  input.focus();
  switchTab('hist');
  input.style.borderColor = '#D97706';
  setTimeout(function(){ input.style.borderColor=''; }, 2000);
}

// ── Init — attach all event listeners here ────────────────────
function init() {
  // Populate tool state
  TOOLS.forEach(function(t){ _toolState[t.id]=t.on; });

  // Render tools
  renderTools();
  updateToolCount();

  // Tab buttons
  document.getElementById('tab-hist').addEventListener('click',  function(){ switchTab('hist');  });
  document.getElementById('tab-tools').addEventListener('click', function(){ switchTab('tools'); });
  document.getElementById('tab-news').addEventListener('click',  function(){ switchTab('news');  });

  // New chat
  document.getElementById('new-chat-btn').addEventListener('click', newChat);

  // Send button
  document.getElementById('send-btn').addEventListener('click', sendMessage);

  // Enter key on textarea
  document.getElementById('msg-input').addEventListener('keydown', function(e){
    if (e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); }
  });

  // Auto-resize textarea
  document.getElementById('msg-input').addEventListener('input', function(){
    this.style.height='auto';
    this.style.height=Math.min(this.scrollHeight,120)+'px';
  });

  // Refresh news button
  document.getElementById('refresh-btn').addEventListener('click', fetchNews);

  // Fetch news on load
  fetchNews();

  // Auto-refresh every 30 min
  setInterval(fetchNews, 30*60*1000);
}

// Gradio 6 renders HTML via Svelte asynchronously
// DOM elements may not exist when this script first runs
// Poll until #send-btn exists, then init
function waitAndInit() {
  if (document.getElementById('send-btn')) {
    init();
  } else {
    setTimeout(waitAndInit, 100);
  }
}
waitAndInit();

})(); // end IIFE
</script>
"""

# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

# Minimal CSS to hide default Gradio UI and make container full height
_GRADIO_CSS = """
footer { display: none !important; }
.gradio-container { padding: 0 !important; margin: 0 !important; max-width: 100% !important; background: #13131a !important; }
.main { padding: 0 !important; }
#component-0 { height: 100vh !important; }
.block { padding: 0 !important; border: none !important; background: transparent !important; }
"""

with gr.Blocks(css=_GRADIO_CSS, title="ML Research Assistant", fill_height=True) as demo:

    # ── Custom UI ──
    gr.HTML(CUSTOM_UI)

    # ── Hidden backend components ──
    # Invisible; JS calls api_name="chat" / "fetch_news" via /gradio_api/run/...

    # Main chat
    _hist_in   = gr.Textbox(value="[]",  visible=False, elem_id="history-input")
    _state_in  = gr.Textbox(value="{}",  visible=False, elem_id="tool-state-input")
    _user_in   = gr.Textbox(value="",    visible=False, elem_id="user-input")
    _hist_out  = gr.Textbox(visible=False)
    _reply_out = gr.Textbox(visible=False)
    _tool_out  = gr.Textbox(visible=False)
    _auto_out  = gr.Textbox(visible=False)

    _user_in.submit(
        chat_fn,
        inputs=[_user_in, _hist_in, _state_in],
        outputs=[_hist_out, _reply_out, _tool_out, _auto_out],
        api_name="chat",
    )

    # News feed fetch
    _news_out = gr.Textbox(visible=False)
    _news_btn = gr.Button(visible=False)
    _news_btn.click(fetch_news_fn, inputs=[], outputs=[_news_out], api_name="fetch_news")

if __name__ == "__main__":
    demo.launch(share=True)
