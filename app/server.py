from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from chain import chain

app = FastAPI(
    title="Real-Time Wikipedia Assistant",
    version="5.0",
)

_context_cache: dict[str, str] = {}


class Message(BaseModel):
    role: str
    content: str

class Question(BaseModel):
    question: str
    history: list[Message] = []
    session_id: str = "default"


@app.post("/ask")
async def ask_question(q: Question):
    try:
        history = [{"role": m.role, "content": m.content} for m in q.history]
        cached_context = _context_cache.get(q.session_id, None)

        result = chain(q.question, history=history, cached_context=cached_context)

        # Safely unpack tuple (answer, context)
        if isinstance(result, tuple):
            answer, new_context = result
        else:
            answer = result
            new_context = cached_context

        if new_context:
            _context_cache[q.session_id] = new_context

        return JSONResponse(content={"answer": str(answer)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/clear")
async def clear_session(body: dict):
    session_id = body.get("session_id", "default")
    _context_cache.pop(session_id, None)
    return JSONResponse(content={"status": "cleared"})


@app.get("/")
async def root():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WikiRAG — Live Wikipedia Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0d0f14;
      --surface: #13161d;
      --border: #1e2330;
      --accent: #4ade80;
      --accent2: #38bdf8;
      --text: #e8eaf0;
      --muted: #6b7280;
      --user-bg: #1a2a1a;
      --user-border: #2d5a2d;
      --ai-bg: #111827;
      --ai-border: #1e2d40;
      --radius: 14px;
      --font-display: 'Syne', sans-serif;
      --font-mono: 'DM Mono', monospace;
    }
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-display);
      height: 100dvh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      overflow: hidden;
    }
    header {
      padding: 18px 28px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 14px;
      background: var(--surface);
      position: relative;
      overflow: hidden;
    }
    header::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, rgba(74,222,128,.06) 0%, transparent 60%);
      pointer-events: none;
    }
    .logo {
      width: 36px; height: 36px;
      background: var(--accent);
      border-radius: 8px;
      display: grid;
      place-items: center;
      font-size: 18px;
      flex-shrink: 0;
    }
    .header-text h1 { font-size: 17px; font-weight: 800; letter-spacing: -.3px; }
    .header-text p { font-size: 11px; color: var(--muted); font-family: var(--font-mono); margin-top: 2px; }
    .header-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
    .pill {
      background: rgba(74,222,128,.12);
      border: 1px solid rgba(74,222,128,.25);
      color: var(--accent);
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 500;
      padding: 4px 10px;
      border-radius: 99px;
      letter-spacing: .5px;
    }
    #clear-btn {
      background: rgba(255,255,255,.04);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 11px;
      font-family: var(--font-mono);
      padding: 4px 12px;
      border-radius: 99px;
      cursor: pointer;
      transition: border-color .2s, color .2s;
    }
    #clear-btn:hover { border-color: #ef4444; color: #ef4444; }
    #chat {
      overflow-y: auto;
      padding: 24px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      scrollbar-width: thin;
      scrollbar-color: var(--border) transparent;
    }
    .welcome {
      margin: auto;
      text-align: center;
      max-width: 480px;
      animation: fadeUp .6s ease forwards;
    }
    .welcome .icon { font-size: 48px; margin-bottom: 16px; }
    .welcome h2 { font-size: 26px; font-weight: 800; letter-spacing: -.5px; }
    .welcome p { color: var(--muted); font-size: 13px; margin-top: 10px; line-height: 1.7; }
    .example-chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 20px; }
    .chip {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 99px;
      padding: 7px 14px;
      font-size: 12px;
      cursor: pointer;
      transition: border-color .2s, color .2s;
      color: var(--muted);
    }
    .chip:hover { border-color: var(--accent); color: var(--accent); }
    .msg {
      display: flex;
      gap: 10px;
      max-width: 780px;
      width: 100%;
      animation: fadeUp .3s ease forwards;
    }
    .msg.user { align-self: flex-end; flex-direction: row-reverse; }
    .msg.ai   { align-self: flex-start; }
    .avatar {
      width: 30px; height: 30px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      font-size: 14px;
      flex-shrink: 0;
      margin-top: 2px;
    }
    .msg.user .avatar { background: var(--user-bg); border: 1px solid var(--user-border); }
    .msg.ai   .avatar { background: var(--ai-bg);   border: 1px solid var(--ai-border); }
    .bubble {
      padding: 12px 16px;
      border-radius: var(--radius);
      font-size: 14px;
      line-height: 1.7;
      max-width: calc(100% - 44px);
    }
    .msg.user .bubble {
      background: var(--user-bg);
      border: 1px solid var(--user-border);
      border-top-right-radius: 4px;
      color: #d1fae5;
    }
    .msg.ai .bubble {
      background: var(--ai-bg);
      border: 1px solid var(--ai-border);
      border-top-left-radius: 4px;
      color: var(--text);
    }
    .typing { display: flex; gap: 5px; padding: 4px 0; }
    .typing span {
      width: 7px; height: 7px;
      background: var(--muted);
      border-radius: 50%;
      animation: bounce 1.2s infinite ease-in-out;
    }
    .typing span:nth-child(2) { animation-delay: .2s; }
    .typing span:nth-child(3) { animation-delay: .4s; }
    .bubble b, .bubble strong { color: #60a5fa; font-weight: 700; }
    .bubble ul, .bubble ol {
      margin: 12px 0 12px 32px !important;
      padding-left: 12px !important;
      display: flex !important;
      flex-direction: column !important;
      gap: 10px !important;
      border-left: 2px solid #334155 !important;
      list-style: none !important;
    }
    .bubble li {
      line-height: 1.75;
      color: var(--text);
      padding-left: 8px;
      position: relative;
    }
    .bubble li::before {
      content: "•";
      color: #60a5fa;
      font-weight: 700;
      position: absolute;
      left: -14px;
    }
    .bubble p { margin: 16px 0; line-height: 1.9; }
    .bubble p:first-child { margin-top: 0; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble ul + p, .bubble ol + p { margin-top: 18px; }
    .bubble p + ul, .bubble p + ol { margin-top: 18px; }
    footer {
      padding: 16px 20px;
      border-top: 1px solid var(--border);
      background: var(--surface);
    }
    .input-row {
      display: flex;
      gap: 10px;
      max-width: 780px;
      margin: 0 auto;
      align-items: flex-end;
    }
    #question {
      flex: 1;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 12px 16px;
      font-size: 14px;
      font-family: var(--font-display);
      color: var(--text);
      outline: none;
      resize: none;
      min-height: 46px;
      max-height: 140px;
      transition: border-color .2s;
      line-height: 1.5;
    }
    #question:focus { border-color: rgba(74,222,128,.4); }
    #question::placeholder { color: var(--muted); }
    #send {
      width: 46px; height: 46px;
      background: var(--accent);
      border: none;
      border-radius: var(--radius);
      cursor: pointer;
      display: grid;
      place-items: center;
      flex-shrink: 0;
      transition: background .2s, transform .1s;
    }
    #send:hover:not(:disabled) { background: #22c55e; }
    #send:active:not(:disabled) { transform: scale(.95); }
    #send:disabled { background: #1a3a1a; cursor: not-allowed; }
    #send svg { width: 18px; height: 18px; fill: #0d0f14; }
    #send:disabled svg { fill: var(--muted); }
    .hint { text-align: center; font-size: 10px; color: var(--muted); margin-top: 8px; font-family: var(--font-mono); }
    @keyframes fadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes bounce {
      0%, 60%, 100% { transform: translateY(0); }
      30% { transform: translateY(-8px); }
    }
  </style>
</head>
<body>
  <header>
    <div class="logo">📖</div>
    <div class="header-text">
      <h1>WikiRAG</h1>
      <p>Real-time Wikipedia · Groq LLaMA 3.3 · Pinecone</p>
    </div>
    <div class="header-right">
      <button id="clear-btn">Clear chat</button>
      <div class="pill">LIVE</div>
    </div>
  </header>

  <div id="chat">
    <div class="welcome" id="welcome">
      <div class="icon">🌍</div>
      <h2>Ask me anything.</h2>
      <p>I search Wikipedia via Pinecone and answer using only that context.<br/>
      Ask follow-up questions freely — I remember the conversation.</p>
      <div class="example-chips">
        <div class="chip" onclick="ask(this)">What is Artificial Intelligence?</div>
        <div class="chip" onclick="ask(this)">History of the Roman Empire</div>
        <div class="chip" onclick="ask(this)">How does CRISPR work?</div>
        <div class="chip" onclick="ask(this)">Who was Ada Lovelace?</div>
        <div class="chip" onclick="ask(this)">What caused World War I?</div>
      </div>
    </div>
  </div>

  <footer>
    <div class="input-row">
      <textarea id="question" placeholder="Ask a question…" rows="1"></textarea>
      <button id="send" title="Send">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M2.01 21 23 12 2.01 3 2 10l15 2-15 2z"/>
        </svg>
      </button>
    </div>
    <p class="hint">Enter ↵ to send · Shift+Enter for new line</p>
  </footer>

  <script>
    const chatEl   = document.getElementById('chat');
    const inputEl  = document.getElementById('question');
    const sendBtn  = document.getElementById('send');
    const clearBtn = document.getElementById('clear-btn');

    // Unique session ID per browser tab
    const sessionId = Math.random().toString(36).slice(2);
    let history = [];

    inputEl.addEventListener('input', () => {
      inputEl.style.height = 'auto';
      inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + 'px';
    });

    function fmt(text) {
      if (!text || typeof text !== 'string') return String(text || '');

      // Step 1: bold
      text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

      // Step 2: split into lines and process
      const lines = text.split('\n');
      const result = [];
      let i = 0;

      while (i < lines.length) {
        const line = lines[i].trim();

        // Bullet line
        if (/^[-•*] (.+)/.test(line)) {
          const items = [];
          while (i < lines.length && /^[-•*] (.+)/.test(lines[i].trim())) {
            items.push('<li>' + lines[i].trim().replace(/^[-•*] /, '') + '</li>');
            i++;
          }
          result.push('<ul>' + items.join('') + '</ul>');
          continue;
        }

        // Numbered line
        if (/^\d+\. (.+)/.test(line)) {
          const items = [];
          while (i < lines.length && /^\d+\. (.+)/.test(lines[i].trim())) {
            items.push('<li>' + lines[i].trim().replace(/^\d+\. /, '') + '</li>');
            i++;
          }
          result.push('<ol>' + items.join('') + '</ol>');
          continue;
        }

        // Empty line = paragraph break (skip it, handled by <p> wrapping)
        if (line === '') {
          i++;
          continue;
        }

        // Regular text line — collect consecutive non-empty, non-list lines
        const para = [];
        while (i < lines.length) {
          const l = lines[i].trim();
          if (l === '' || /^[-•*] /.test(l) || /^\d+\. /.test(l)) break;
          para.push(l);
          i++;
        }
        if (para.length > 0) {
          result.push('<p>' + para.join(' ') + '</p>');
        }
      }

      return result.join('');
    }

    function addMsg(role, content) {
      const isUser   = role === 'user';
      const isTyping = content === '__typing__';
      // Ensure content is always a string
      content = String(content || '');
      document.getElementById('welcome')?.remove();

      const wrap = document.createElement('div');
      wrap.className = `msg ${role}`;

      const av = document.createElement('div');
      av.className = 'avatar';
      av.textContent = isUser ? '🧑' : '🤖';

      const bub = document.createElement('div');
      bub.className = 'bubble';

      if (isTyping) {
        bub.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
      } else if (isUser) {
        bub.textContent = content;
      } else {
        bub.innerHTML = fmt(content);
      }

      wrap.appendChild(av);
      wrap.appendChild(bub);
      chatEl.appendChild(wrap);
      chatEl.scrollTop = chatEl.scrollHeight;
      return bub;
    }

    async function sendQuestion(q) {
      q = q.trim();
      if (!q) return;

      addMsg('user', q);
      inputEl.value = '';
      inputEl.style.height = 'auto';
      sendBtn.disabled = true;

      const typingBub = addMsg('ai', '__typing__');

      try {
        const res = await fetch('/ask', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: q, history, session_id: sessionId }),
        });
        const data = await res.json();
        typingBub.parentElement.remove();

        if (data.error) {
          addMsg('ai', `⚠️ Error: ${data.error}`);
        } else {
          addMsg('ai', data.answer);
          history.push({ role: 'user',      content: q });
          history.push({ role: 'assistant', content: data.answer });
          if (history.length > 20) history = history.slice(-20);
        }
      } catch (err) {
        typingBub.parentElement.remove();
        addMsg('ai', `⚠️ Network error: ${err.message}`);
      }

      sendBtn.disabled = false;
      inputEl.focus();
    }

    function ask(chip) { sendQuestion(chip.textContent); }

    function showWelcome() {
      const welcome = document.createElement('div');
      welcome.className = 'welcome';
      welcome.id = 'welcome';
      welcome.innerHTML = `
        <div class="icon">🌍</div>
        <h2>Ask me anything.</h2>
        <p>I search Wikipedia via Pinecone and answer using only that context.<br/>
        Ask follow-up questions freely — I remember the conversation.</p>
        <div class="example-chips">
          <div class="chip" onclick="ask(this)">What is Artificial Intelligence?</div>
          <div class="chip" onclick="ask(this)">History of the Roman Empire</div>
          <div class="chip" onclick="ask(this)">How does CRISPR work?</div>
          <div class="chip" onclick="ask(this)">Who was Ada Lovelace?</div>
          <div class="chip" onclick="ask(this)">What caused World War I?</div>
        </div>`;
      chatEl.appendChild(welcome);
    }

    clearBtn.addEventListener('click', () => {
      history = [];
      fetch('/clear', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId })
      });
      chatEl.innerHTML = '';
      showWelcome();
    });

    sendBtn.addEventListener('click', () => sendQuestion(inputEl.value));
    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(inputEl.value); }
    });

    inputEl.focus();
  </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)