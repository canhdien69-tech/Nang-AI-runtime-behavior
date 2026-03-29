# ==============================================================================
# PROJECT: NẮNG AI - v47.4
# FILE: main.py
# MỤC ĐÍCH: FastAPI backend + WebSocket streaming + React Cyberpunk UI
#   [G4] Thay Tkinter bằng FastAPI + WebSocket → không còn thread-unsafe UI
#   Chạy: python main.py  →  mở http://localhost:8000
# ==============================================================================

import asyncio, json, re, threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config import NangConfig, logger
from utils import SystemUtils
from audio import AudioMouth, AudioEar
from tools import Toolbox, ToolGuard
from brain import NangBrain


# ==============================================================================
# LIFESPAN — khởi động backend một lần khi server start
# ==============================================================================
brain:         NangBrain   = None
tools:         Toolbox     = None
guard:         ToolGuard   = None
mouth:         AudioMouth  = None
executor_llm:  ThreadPoolExecutor = None
executor_tools:ThreadPoolExecutor = None
executor_audio:ThreadPoolExecutor = None   # riêng cho TTS tránh block tool threads
_tool_semaphore: asyncio.Semaphore = None
# [#3] Flag thay _work_queue private API — thread-safe, không phụ thuộc Python version
import threading as _threading
_audio_speaking = _threading.Event()   # set khi TTS đang chạy, clear khi xong


@asynccontextmanager
async def lifespan(app: FastAPI):
    global brain, tools, guard, mouth, executor_llm, executor_tools, executor_audio, _tool_semaphore
    logger.info("[Server] Khởi động backend...")
    brain          = NangBrain()
    guard          = ToolGuard()
    tools          = Toolbox(brain.tokenizer, guard=guard)
    mouth          = AudioMouth()
    executor_llm   = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm")
    # [#3] Giảm max_workers=2 — giới hạn zombie tool threads thực tế nhất cho desktop app
    executor_tools = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tool")
    executor_audio = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")
    # [#1] Semaphore = max_workers tool — acquire TRƯỚC submit để thực sự giới hạn
    _tool_semaphore = asyncio.Semaphore(2)
    logger.info(f"[Server] Sẵn sàng. Linh hồn Gen {brain.soul.active.generation} đã thức tỉnh.")
    yield
    executor_llm.shutdown(wait=False)
    executor_tools.shutdown(wait=False)
    executor_audio.shutdown(wait=False)
    if mouth is not None:
        mouth.close()   # [#1] Shutdown TTS event loop — tránh loop leak khi restart
    logger.info("[Server] Shutdown.")


app = FastAPI(lifespan=lifespan)


# ==============================================================================
# REACT CYBERPUNK UI — single HTML file served at root
# ==============================================================================
CYBERPUNK_HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>NẮNG AI — DEEPMIND RSSM</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');
  :root {
    --bg:       #050810;
    --panel:    rgba(0,255,255,0.04);
    --border:   rgba(0,255,255,0.18);
    --cyan:     #00f5ff;
    --pink:     #ff2d78;
    --yellow:   #f5e642;
    --green:    #39ff14;
    --dim:      rgba(255,255,255,0.35);
    --glow-c:   0 0 8px #00f5ff, 0 0 20px rgba(0,245,255,0.3);
    --glow-p:   0 0 8px #ff2d78, 0 0 20px rgba(255,45,120,0.3);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: #e0f7ff;
    font-family: 'Share Tech Mono', monospace;
    height: 100vh; display: flex; flex-direction: column;
    overflow: hidden;
  }
  /* ── HEADER ── */
  header {
    padding: 10px 20px;
    border-bottom: 1px solid var(--border);
    background: rgba(0,245,255,0.03);
    display: flex; align-items: center; gap: 16px;
  }
  header h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem; font-weight: 900;
    color: var(--cyan); text-shadow: var(--glow-c);
    letter-spacing: 3px;
  }
  #status-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: #333; transition: background .3s;
  }
  #status-dot.online  { background: var(--green); box-shadow: 0 0 8px var(--green); }
  #status-dot.typing  { background: var(--pink);  box-shadow: 0 0 8px var(--pink); animation: pulse 0.6s infinite alternate; }
  @keyframes pulse { to { opacity: 0.3; } }
  #soul-info {
    margin-left: auto; font-size: .72rem; color: var(--dim);
  }
  /* ── CHAT AREA ── */
  #chat {
    flex: 1; overflow-y: auto; padding: 16px 20px;
    display: flex; flex-direction: column; gap: 10px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  .msg {
    max-width: 78%; padding: 10px 14px;
    border-radius: 4px; line-height: 1.55; font-size: .88rem;
    position: relative; word-break: break-word;
  }
  .msg.user {
    align-self: flex-end;
    background: rgba(0,245,255,0.08);
    border: 1px solid var(--cyan);
    color: #c8f9ff;
    box-shadow: var(--glow-c);
  }
  .msg.ai {
    align-self: flex-start;
    background: rgba(255,45,120,0.07);
    border: 1px solid var(--pink);
    color: #ffd6e8;
    box-shadow: var(--glow-p);
  }
  .msg.sys {
    align-self: center;
    background: rgba(245,230,66,0.06);
    border: 1px solid rgba(245,230,66,0.3);
    color: var(--yellow); font-size: .78rem;
    max-width: 90%; text-align: center;
  }
  .msg-label {
    font-size: .68rem; font-family: 'Orbitron', sans-serif;
    margin-bottom: 4px; letter-spacing: 1px;
  }
  .msg.user .msg-label { color: var(--cyan); }
  .msg.ai   .msg-label { color: var(--pink); }
  /* ── TOOL BADGE ── */
  .tool-badge {
    display: inline-block; font-size: .65rem;
    background: rgba(57,255,20,0.12);
    border: 1px solid var(--green);
    color: var(--green); padding: 1px 6px;
    border-radius: 2px; margin-bottom: 5px;
  }
  /* ── INPUT AREA ── */
  #input-area {
    padding: 12px 20px;
    border-top: 1px solid var(--border);
    background: rgba(0,245,255,0.02);
    display: flex; gap: 8px; align-items: center;
  }
  #msg-input {
    flex: 1; background: rgba(0,0,0,0.4);
    border: 1px solid var(--border);
    color: #e0f7ff; font-family: 'Share Tech Mono', monospace;
    font-size: .9rem; padding: 9px 14px; border-radius: 3px;
    outline: none; transition: border .2s;
  }
  #msg-input:focus { border-color: var(--cyan); box-shadow: var(--glow-c); }
  .btn {
    font-family: 'Orbitron', sans-serif; font-size: .72rem;
    font-weight: 700; letter-spacing: 1px;
    padding: 9px 18px; border: none; border-radius: 3px;
    cursor: pointer; transition: all .15s;
  }
  #send-btn {
    background: var(--cyan); color: #000;
    box-shadow: var(--glow-c);
  }
  #send-btn:hover   { background: #00d4e0; }
  #send-btn:disabled { opacity: .4; cursor: not-allowed; }
  #mic-btn {
    background: var(--pink); color: #fff;
    box-shadow: var(--glow-p);
  }
  #mic-btn.listening { animation: pulse .5s infinite alternate; }
  /* ── SCROLLBAR ── */
  #chat::-webkit-scrollbar { width: 4px; }
  #chat::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
  /* ── SCANLINE OVERLAY ── */
  body::after {
    content: ''; position: fixed; inset: 0; pointer-events: none;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px
    );
    z-index: 9999;
  }
</style>
</head>
<body>
<header>
  <div id="status-dot"></div>
  <h1>NẮNG AI &nbsp;//&nbsp; DEEPMIND RSSM v47.4</h1>
  <span id="soul-info">kết nối...</span>
</header>
<div id="chat"></div>
<div id="input-area">
  <input id="msg-input" type="text" placeholder="Nhắn tin với Nắng..." autocomplete="off"/>
  <button class="btn" id="mic-btn" title="Mic">🎙</button>
  <button class="btn" id="send-btn">GỬI</button>
</div>

<script>
const chat     = document.getElementById('chat');
const input    = document.getElementById('msg-input');
const sendBtn  = document.getElementById('send-btn');
const micBtn   = document.getElementById('mic-btn');
const dot      = document.getElementById('status-dot');
const soulInfo = document.getElementById('soul-info');

let ws = null;
let currentAiDiv = null;
let currentAiText = null;

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    dot.className = 'online';
    addMsg('sys', '✦ KẾT NỐI THÀNH CÔNG ✦');
  };

  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);

    if (data.type === 'soul') {
      soulInfo.textContent = `Gen ${data.gen} | Entropy ${data.entropy} | Loss ${data.loss}`;
      return;
    }
    if (data.type === 'tool') {
      if (currentAiDiv) {
        const badge = document.createElement('div');
        badge.className = 'tool-badge';
        badge.textContent = '⚙ ' + data.name;
        currentAiDiv.querySelector('.msg-content').prepend(badge);
      }
      return;
    }
    if (data.type === 'start') {
      dot.className = 'typing';
      sendBtn.disabled = true;
      currentAiDiv  = createMsgDiv('ai', '');
      currentAiText = currentAiDiv.querySelector('.msg-content');
      return;
    }
    if (data.type === 'token') {
      if (currentAiText) {
        currentAiText.textContent += data.token;
        chat.scrollTop = chat.scrollHeight;
      }
      return;
    }
    if (data.type === 'end') {
      dot.className = 'online';
      sendBtn.disabled = false;
      currentAiDiv  = null;
      currentAiText = null;
      return;
    }
    if (data.type === 'error') {
      addMsg('sys', '⚠ ' + data.msg);
      dot.className = 'online';
      sendBtn.disabled = false;
      return;
    }
    if (data.type === 'sys') {
      addMsg('sys', data.msg);
      return;
    }
  };

  ws.onclose = () => {
    dot.className = '';
    addMsg('sys', '⚡ Mất kết nối — thử lại sau 3s...');
    setTimeout(connect, 3000);
  };

  ws.onerror = () => ws.close();
}

function createMsgDiv(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const label = document.createElement('div');
  label.className = 'msg-label';
  label.textContent = role === 'user' ? 'ANH' : 'NẮNG';
  const content = document.createElement('div');
  content.className = 'msg-content';
  content.textContent = text;
  wrap.appendChild(label);
  wrap.appendChild(content);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
  return wrap;
}

function addMsg(role, text) {
  createMsgDiv(role, text);
}

function send() {
  const msg = input.value.trim();
  if (!msg || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMsg('user', msg);
  ws.send(JSON.stringify({type: 'chat', msg}));
  input.value = '';
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

// Mic button — Web Speech API fallback
micBtn.addEventListener('click', () => {
  if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
    addMsg('sys', '⚠ Trình duyệt không hỗ trợ mic. Dùng Chrome.');
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recog = new SR();
  recog.lang = 'vi-VN'; recog.interimResults = false;
  micBtn.classList.add('listening');
  recog.onresult = e => {
    input.value = e.results[0][0].transcript;
    micBtn.classList.remove('listening');
    send();
  };
  recog.onerror = () => micBtn.classList.remove('listening');
  recog.onend   = () => micBtn.classList.remove('listening');
  recog.start();
});

connect();
</script>
</body>
</html>"""


# ==============================================================================
# ROUTES
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("[WS] Client connected")

    # [#6] Guard: brain có thể chưa init nếu WS connect trước lifespan xong
    if brain is None or _tool_semaphore is None:
        await ws.send_text(json.dumps({"type": "error", "msg": "Server chưa sẵn sàng, thử lại sau."}))
        await ws.close()
        return

    async def send(data: dict) -> bool:
        """Gửi data về client. Trả về False nếu client đã disconnect."""
        try:
            await ws.send_text(json.dumps(data, ensure_ascii=False))
            return True
        except Exception as e:
            logger.debug(f"[WS SEND FAIL]: {e}")
            return False

    # Gửi soul info ban đầu
    await send({
        "type":    "soul",
        "gen":     brain.soul.active.generation,
        "entropy": round(brain.soul.avg_entropy, 3),
        "loss":    "—",
    })
    await send({"type": "sys", "msg": f"✦ Nắng đã thức tỉnh. Linh hồn Gen {brain.soul.active.generation}. ✦"})

    loop          = asyncio.get_running_loop()
    _req_count    = 0
    DANGEROUS     = {"delete_junk", "control_phone"}
    _session_lock = asyncio.Lock()
    import time as _time
    _last_req_ts  = 0.0
    _RATE_LIMIT_S = 1.0

    # [#1] Ping task riêng — receive_text() blocking nên ping phải chạy song song
    # Đặt ping sau receive_text() là dead code vì bị block khi client không gửi gì
    _ping_task: asyncio.Task = None

    async def _ping_loop():
        """Gửi ping mỗi 30s để giữ connection qua proxy/nginx."""
        try:
            while True:
                await asyncio.sleep(30)
                if not await send({"type": "ping"}):
                    break   # send fail → client disconnect → dừng ping
        except asyncio.CancelledError:
            pass   # bình thường khi session kết thúc

    _ping_task = asyncio.create_task(_ping_loop())

    try:
        while True:
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=300)
            except asyncio.TimeoutError:
                logger.info("[WS] Client không hoạt động 5 phút — đóng kết nối.")
                await ws.close()
                break

            # JSON decode error
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("[WS] JSON decode error — bỏ qua.")
                continue

            msg_type = data.get("type", "")

            # Update config — bật/tắt Research Mode từ UI
            if msg_type == "update_config":
                key = data.get("key")
                value = data.get("value")
                if key == "RESEARCH_MODE":
                    NangConfig.RESEARCH_MODE = value
                    NangConfig.BASELINE_MODE = value
                    await send({"type": "sys", "msg": f"⚙️ Research Mode: {'BẬT' if value else 'TẮT'}"})
                continue

            # Switch model — xử lý ngay, không vào session lock
            if msg_type == "switch_model":
                size = data.get("size", "3B")
                await send({"type": "sys", "msg": f"⚙️ Đang chuyển sang {size}B... vui lòng chờ"})
                ok = await loop.run_in_executor(
                    executor_llm, lambda s=size: brain.switch_model(s)
                )
                if ok:
                    await send({"type": "sys", "msg": f"✅ Đã chuyển sang {size}B thành công"})
                else:
                    await send({"type": "sys", "msg": f"⚠️ Chuyển model thất bại hoặc đang dùng {size}B rồi"})
                continue

            # Accept both "chat" (legacy) and "message" (frontend)
            if msg_type not in ("chat", "message"):
                continue

            user_msg = (data.get("msg") or data.get("content") or "").strip()
            if not user_msg:
                continue

            if len(user_msg) > 2000:
                user_msg = user_msg[:2000]
                await send({"type": "sys", "msg": "⚠️ Tin nhắn quá dài, đã tự động cắt."})

            user_msg = Toolbox.sanitize_prompt(user_msg)

            # Rate limit — KHÔNG update timestamp khi reject để tránh bypass bằng spam
            now = _time.monotonic()
            if now - _last_req_ts < _RATE_LIMIT_S:
                await send({"type": "sys", "msg": "⏳ Anh nhắn nhanh quá, chờ tí nhé..."})
                continue
            _last_req_ts = now

            # _session_lock.locked() = đang xử lý → báo busy
            if _session_lock.locked():
                await send({"type": "sys", "msg": "⏳ Em đang trả lời, anh chờ tí nhé..."})
                continue

            async with _session_lock:
                _req_count += 1

                # ── Tool detection & execution ──────────────────────────────
                # [#8] Dùng list + join thay string concat O(n²)
                tool_res_parts = []
                tool_conf      = False
                _tool_names    = []
                _tool_name     = ""
                _action_name   = "calm"   # default — overwrite sau think()
                _h_np          = None
                _z_np          = None
                _reflection    = None
                _run_id        = ""
                is_panicking = brain.soul.is_frozen or brain.soul.active.e_weights[1] > 0.6

                # [#6] Router timeout — tránh WS treo nếu router hang
                try:
                    tool_calls = await asyncio.wait_for(
                        loop.run_in_executor(executor_llm, brain.router.detect, user_msg),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("[Router] Timeout — fallback no_tool")
                    tool_calls = [{"name": "no_tool", "arguments": {}}]

                # [#10] Validate tool_calls structure — LLM có thể trả None/string/malformed
                if not isinstance(tool_calls, list):
                    tool_calls = []

                for call in tool_calls[:3]:
                    # [#4] Guard — LLM có thể trả ["string", 123, None] thay vì list of dicts
                    if not isinstance(call, dict):
                        continue
                    name = call.get("name", "no_tool")
                    if name != "no_tool": _tool_names.append(name)
                    # [#8] Validate args type — LLM có thể trả args = string thay vì dict
                    args = call.get("arguments", {})
                    if not isinstance(args, dict):
                        args = {}

                    if name == "no_tool":
                        continue

                    name = name.lower()

                    # [#11] DANGEROUS double check — không chỉ rely vào guard
                    if name in DANGEROUS and is_panicking:
                        tool_res_parts.append(f"\n[GUARD]: Tool {name} bị khóa khi Nắng đang hoảng loạn.")
                        continue

                    ok, reason = guard.is_allowed(name, is_panicking)
                    if not ok:
                        tool_res_parts.append(f"\n[GUARD]: {reason}")
                        continue

                    if name == "visit_url":
                        url = args.get("url", "")
                        if url.startswith("file://") or re.search(
                            r'https?://(?:localhost|127\.|10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)',
                            url
                        ):
                            tool_res_parts.append("\n[GUARD]: URL bị chặn (file:// hoặc internal IP)")
                            continue

                    # [SECURITY] Defense-in-depth path validation
                    if name == "read_file":
                        from pathlib import Path as _Path
                        raw_path = args.get("path", "")
                        _blocked = False
                        try:
                            p = _Path(raw_path)
                            # [#3] Chặn symlink — .resolve() follow symlink nên check trước
                            if p.is_symlink():
                                tool_res_parts.append("\n[GUARD]: Symlink bị chặn vì lý do bảo mật")
                                _blocked = True
                            if not _blocked:
                                resolved = str(p.resolve())
                                _SYS_PREFIXES = [
                                    "/etc", "/proc", "/sys", "/dev", "/boot", "/root",
                                    "C:\\Windows", "C:\\System32", "C:\\Program Files",
                                ]
                                if any(resolved.lower().startswith(px.lower()) for px in _SYS_PREFIXES):
                                    tool_res_parts.append("\n[GUARD]: Path bị chặn (system directory)")
                                    _blocked = True
                            if not _blocked and ".." in raw_path:
                                import os as _os
                                resolved2 = str(_Path(raw_path).resolve())
                                cwd  = _os.getcwd()
                                home = str(_Path.home())
                                if not (resolved2.startswith(cwd) or resolved2.startswith(home)):
                                    tool_res_parts.append("\n[GUARD]: Path traversal bị chặn")
                                    _blocked = True
                        except Exception:
                            tool_res_parts.append("\n[GUARD]: Path không hợp lệ")
                            _blocked = True
                        if _blocked:
                            continue

                    await send({"type": "tool", "name": name})

                    # [#1] Acquire semaphore TRƯỚC khi submit — không tạo thread khi đã đủ
                    # Submit bên ngoài semaphore là sai bản chất: thread được tạo trước, sau đó mới chờ
                    _tool_fn = None
                    _tool_args = ()
                    if   name == "search_web":   _tool_fn, _tool_args = tools.search_web,    (str(args.get("query",   user_msg))[:500],)
                    elif name == "visit_url":    _tool_fn, _tool_args = tools.visit_website, (str(args.get("url",     ""))[:500],)
                    elif name == "read_file":    _tool_fn, _tool_args = tools.read_file,     (str(args.get("path",    ""))[:500],)
                    elif name == "scan_junk":    _tool_fn, _tool_args = tools.scan_junk,     ()
                    elif name == "delete_junk":  _tool_fn, _tool_args = tools.delete_junk,   ()
                    elif name == "control_phone":_tool_fn, _tool_args = tools.control_phone, (str(args.get("command", user_msg))[:200],)
                    elif name == "read_diary":   _tool_fn, _tool_args = tools.read_diary,    ()

                    if _tool_fn is not None:
                        fut = None   # [#2] Init trước — tránh UnboundLocalError nếu timeout trước assign
                        try:
                            async with _tool_semaphore:
                                fut = executor_tools.submit(_tool_fn, *_tool_args)
                                r   = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=10.0)
                            if isinstance(r, tuple):
                                tool_res_parts.append(r[0])
                                tool_conf  = tool_conf or r[1]
                                if not r[1]: guard.set_cooldown(name)
                            else:
                                tool_res_parts.append(str(r))
                                tool_conf  = True
                        except asyncio.TimeoutError:
                            tool_res_parts.append(f"\n[TOOL TIMEOUT]: {name} không phản hồi sau 10s.")
                            guard.set_cooldown(name)
                            if fut is not None:   # [#2] Guard — fut có thể chưa được assign
                                fut.cancel()
                        except Exception as e:
                            tool_res_parts.append(f"\n[TOOL ERROR]: {e}")
                            guard.set_cooldown(name)
                            await loop.run_in_executor(
                                executor_tools,
                                lambda n=name, err=e: brain.ltm.save_failure("tool_fail", f"{n}: {err}")
                            )

                # [#8] Join tool_res một lần từ list — O(n) thay O(n²) string concat
                tool_res = "".join(tool_res_parts)
                if len(tool_res) > 4000:
                    tool_res = tool_res[:4000] + "\n[TOOL RES TRUNCATED]"

                # Lưu tool_name thật để pass vào orchestrate() — không dùng _action_name
                _tool_name = ",".join(dict.fromkeys(_tool_names)) if _tool_names else ""   # dedup, giữ order

                # ── LLM Generation ──────────────────────────────────────────
                if not await send({"type": "start"}):
                    continue

                try:
                    streamer, gen_exception, _action_name, _h_np, _z_np, _reflection, _run_id = await asyncio.wait_for(
                        loop.run_in_executor(
                            executor_llm,
                            lambda: brain.think(user_msg, tool_result=tool_res, tool_confidence=tool_conf)
                        ),
                        timeout=NangConfig.GENERATE_TIMEOUT + 10
                    )
                except asyncio.TimeoutError:
                    await send({"type": "error", "msg": "LLM timeout — model không phản hồi."})
                    continue

                # [#5] Dùng list + join cho full_response — tránh O(n²) string concat
                # Chỉ giữ để save_interaction và sentiment — không cần giữ toàn bộ trong RAM
                _resp_parts   = []
                _stream_iter  = iter(streamer)
                _send_failed  = 0
                _total_len    = 0   # [#3] Counter thay sum() — O(1) mỗi token thay O(n)

                # [#9] Wrap toàn bộ stream loop với timeout — tránh stream treo vô hạn
                async def _stream_loop():
                    nonlocal _resp_parts, _send_failed, _total_len

                    def _get_next():
                        try:
                            return next(_stream_iter)
                        except StopIteration:
                            return None

                    while True:
                        try:
                            tok = await asyncio.to_thread(_get_next)
                            if tok is None:
                                break
                            if not tok:
                                await asyncio.sleep(0)
                                continue
                            _resp_parts.append(tok)
                            _total_len += len(tok)   # [#3] O(1) thay sum() O(n)
                            if _total_len > 10000:
                                logger.warning("[Stream] Response quá dài — dừng stream.")
                                await send({"type": "sys", "msg": "⚠️ Câu trả lời quá dài, đã bị cắt."})
                                break
                            await send({"type": "token", "token": tok})
                            _send_failed = 0
                        except StopIteration:
                            break
                        except Exception as e:
                            _send_failed += 1
                            logger.warning(f"[Stream] token error: {e}")
                            if _send_failed >= 3:
                                logger.info("[Stream] Client có thể đã disconnect — dừng stream.")
                                break

                try:
                    await asyncio.wait_for(_stream_loop(), timeout=NangConfig.GENERATE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("[Stream] Stream timeout — dừng.")

                # Join một lần sau khi stream xong
                full_response = "".join(_resp_parts)

                # [RESEARCH] Orchestrator
                if NangConfig.RESEARCH_MODE and full_response:
                    _orch = await loop.run_in_executor(
                        executor_llm,
                        lambda: brain.orchestrate(
                            response      = full_response,
                            run_id        = _run_id,
                            tool_name     = _tool_name,
                            tool_result   = tool_res,
                            rag_context   = brain._last_rag_context,
                            action_name   = _action_name,
                            h_np          = _h_np,
                            z_np          = _z_np,
                            stress_factor = brain._prev_stress,
                            user_msg      = user_msg,
                        )
                    )
                    full_response = _orch["final_response"]
                    _decision = _orch["pipeline_log"].get("decision", "ok")
                    if _decision == "reject":
                        await send({"type": "sys", "msg": "⚠️ [Nắng không chắc về câu trả lời này]"})
                    elif _decision == "improved":
                        logger.info("[Orchestrator] Response improved by reflection")
                    if _req_count % 10 == 0:
                        logger.info(f"[Orchestrator] decision={_decision} log={_orch['pipeline_log']}")

                from queue import Empty as _QueueEmpty
                while True:
                    try:
                        err = gen_exception.get_nowait()
                        await send({"type": "error", "msg": str(err)})
                        await loop.run_in_executor(
                            executor_llm,
                            lambda e=err: brain.ltm.save_failure(
                                "timeout" if "treo" in str(e) else "generate_error", str(e)[:200]
                            )
                        )
                    except _QueueEmpty:
                        break

                await loop.run_in_executor(executor_llm, lambda: brain.save_interaction(user_msg, full_response, tool_conf))
                # [M10] Update memory usefulness — đo semantic overlap memory vs response
                # Gọi trước update_reward_signal để usefulness có thể ảnh hưởng reward sau này
                if full_response:
                    await loop.run_in_executor(
                        executor_llm,
                        lambda: brain.ltm.update_usefulness(full_response, brain._last_turn_id)
                    )
                _sent_score = brain.sentiment.score(full_response[:200]) if full_response else 0.5
                _sent_norm  = float(max(0.0, min(1.0, 0.5 + _sent_score * 0.33)))
                _mem_useful = brain.ltm.get_avg_usefulness()
                await loop.run_in_executor(
                    executor_llm,
                    lambda s=_sent_norm, tc=tool_conf, mu=_mem_useful:
                        brain.update_reward_signal(s, tc, True, mu)
                )

                # [RESEARCH] Log turn metrics sau mỗi turn hoàn thành
                if NangConfig.RESEARCH_MODE:
                    _dreamer_loss = float(brain.soul.dreamer.loss_history[-1]) \
                        if brain.soul.dreamer.loss_history else 0.0
                    # [GPT #1] Dùng brain.last_reward — reward thật từ env.step()
                    # KHÔNG dùng sentiment_shift — đó chỉ là 1 thành phần của reward
                    # [GPT #2] _action_name đã được unpack từ think() — đúng timing
                    await loop.run_in_executor(
                        executor_llm,
                        lambda: brain.log_research_turn(
                            user_input      = user_msg,
                            response        = full_response,
                            stress_factor   = brain._prev_stress,
                            action_name     = _action_name,
                            sentiment_score = _sent_norm,
                            reward          = brain.last_reward,
                            dreamer_loss    = _dreamer_loss,
                            h_np            = _h_np,
                            z_np            = _z_np,
                        )
                    )

                if full_response.strip():
                    # [#3] Dùng threading.Event thay _work_queue private API
                    # _work_queue.qsize() không có guarantee ổn định giữa Python versions
                    if not _audio_speaking.is_set():
                        def _speak_with_flag(text):
                            _audio_speaking.set()
                            try:
                                mouth.speak(text)
                            finally:
                                _audio_speaking.clear()
                        loop.run_in_executor(executor_audio, _speak_with_flag, full_response)

                await send({
                    "type":    "soul",
                    "gen":     brain.soul.active.generation,
                    "entropy": round(brain.soul.avg_entropy, 3),
                    "loss":    f"{brain.soul.dreamer.loss_history[-1]:.4f}" if brain.soul.dreamer.loss_history else "—",
                })

                if NangConfig.RESEARCH_MODE:
                    actual_reward = getattr(brain, "last_reward", 0.0)
                    actual_stress = getattr(brain, "_prev_stress", 0.0)
                    await send({
                        "type":   "metrics",
                        "reward": actual_reward,
                        "stress": actual_stress,
                        "action": _action_name,
                        "eval":   getattr(brain, "last_eval", "N/A"),
                    })
                    print(f"[DEBUG] reward={actual_reward:.3f} action={_action_name}")

                _pipeline = _orch.get("pipeline_log", {}) if "_orch" in dir() and isinstance(_orch, dict) else {}
                await send({
                    "type":          "end",
                    "reward":        getattr(brain, "last_reward", 0.0),
                    "stress":        getattr(brain, "_prev_stress", 0.0),
                    "action":        _action_name,
                    "entropy":       round(brain.soul.avg_entropy, 3) if hasattr(brain.soul, "avg_entropy") else 0.0,
                    "eval_score":    _pipeline.get("self_eval", {}).get("score", 0.0),
                    "dreamer_loss":  float(brain.soul.dreamer.loss_history[-1]) if brain.soul.dreamer.loss_history else 0.0,
                    "consistency":   _pipeline.get("reflection", {}).get("consistency", 0.0),
                    "decision":      _pipeline.get("decision", "ok"),
                    "hallucination": _pipeline.get("hallucination", {}).get("confidence", 0.0),
                    "mode":          "latent" if getattr(brain, "_latent_gen", None) else "baseline",
                })
                print(f"[DEBUG] reward={getattr(brain, 'last_reward', 0.0):.3f} action={_action_name}")

                if _req_count % 5 == 0:
                    await loop.run_in_executor(
                        executor_llm,
                        lambda: SystemUtils.clean_memory(brain.soul.dreamer.device)
                    )

                if NangConfig.RESEARCH_MODE and _req_count % 10 == 0:
                    logger.info(
                        f"[Research] turn={_req_count} "
                        f"action={_action_name} "
                        f"reward={brain.last_reward:.3f} "
                        f"stress={brain._prev_stress:.3f} "
                        f"mode={'baseline' if NangConfig.BASELINE_MODE else 'latent'}"
                    )
                # NOTE: executor restart đã bị xóa vì gây race condition:
                # request A đang dùng executor_tools, request B replace nó →
                # old semaphore và new semaphore không đồng bộ → concurrency limit bypass.
                # Fix thực sự: internal timeout trong từng tool (đã có trong tools.py).

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
        try:
            await send({"type": "error", "msg": str(e)})
        except Exception:
            pass
    finally:
        # [RESEARCH] Save session khi disconnect
        if NangConfig.RESEARCH_MODE:
            try:
                brain.save_research_session()
            except Exception as e:
                logger.warning(f"[Research] save_research_session failed: {e}")
        # [#7] Cancel ping task
        if _ping_task is not None and not _ping_task.done():
            _ping_task.cancel()
            try:
                await _ping_task
            except asyncio.CancelledError:
                pass


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=NangConfig.SERVER_HOST,
        port=NangConfig.SERVER_PORT,
        reload=False,
        log_level="info"
    )
