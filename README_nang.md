# 🌤️ Nắng AI v47

> Lightweight autonomous agent with latent-conditioned behavior control

**Nắng AI không phải chatbot.**

Nó là một hệ thống kết hợp:
- LLM generation (Qwen 2.5 3B / Hermes 3 8B, 4-bit quantized)
- World Model (Dreamer / RSSM)
- Behavior control qua latent state
- Tool system + safety layers
- Self-evaluation + reflection engine

---

## 💡 Core Idea

```
LLM không tự quyết định hoàn toàn.

Behavior bị điều khiển bởi:
  - latent state (h, z) từ RSSM
  - reward signal từ environment
  - self-evaluation từng turn
```

Thay vì prompt thêm "hãy bình tĩnh" — hệ thống điều khiển behavior trực tiếp qua latent state injection.

---

## 🏗️ Architecture

```
User Input
    ↓
Embedding + Sentiment Scoring
    ↓
ConversationEnv (state vector)
    ↓
RSSM World Model (h, z latent)
    ↓
LatentAdapter → soft prefix tokens → LLM
    ↓
Response (streaming)
    ↓
Self-Eval + Reflection + Reward
    ↓
Memory (ChromaDB) + Metrics logging
```

---

## 🧠 Core Components

### 1. NangBrain (`brain.py`) — Orchestrator
- Load LLM 4-bit quantized
- Shared embedding model (không load 2 lần)
- Thread-safe generation với RLock
- Hybrid tool routing (regex fast-path + LLM fallback)
- Streaming token output

### 2. World Model — RSSM Dreamer (`soul.py`)
- Latent state: `h` (deterministic) + `z` (stochastic)
- Encode emotional dynamics qua turns
- Drive behavior thông qua latent conditioning

### 3. Latent → LLM Bridge (`latent_adapter.py`)
- `(h, z)` → adapter → soft prefix tokens → LLM
- Không dùng prompt text, không cần finetune LLM
- Điều khiển behavior trực tiếp ở token level

### 4. Conversation Environment (`env.py`)
- State = embedding vector + meta signals
- Action space: `[calm, warm, concerned, tool_use, tool_skip, memory_deep]`
- Reward: sentiment shift + task success + memory usefulness
- Không phải toy env — dùng trực tiếp trên conversation thật

### 5. Long-term Memory (`memory.py`)
- ChromaDB storage với async embedding (non-blocking)
- Deduplication (cosine > 0.95)
- Importance filtering + recency decay
- Failure memory riêng cho tool errors
- Memory → reward shaping

### 6. Sentiment System (`sentiment.py`)
- Embedding-based (không keyword matching)
- Top-k cosine mean
- Context-aware scoring

### 7. Tool System + Safety (`tools.py`, `tool_verifier.py`)
- `ToolGuard`: cooldown, permission check
- `Toolbox`: web search, read file, visit URL, ADB phone control
- `ToolVerifier`: LLM tự verify output (VALID / INVALID / UNCERTAIN)

### 8. Self Evaluation (`self_evaluator.py`)
- LLM tự chấm: relevance, tone, safety
- Rule-based fallback: generic detection, repetition, persona break

### 9. Hallucination Detection (`hallucination_detector.py`)
- Không dùng LLM — dùng embedding similarity
- Semantic similarity + contradiction detection + length anomaly

### 10. Reflection Engine (`reflection.py`)
- Dùng latent state (h, z) để check consistency
- Nếu lệch → regenerate response

### 11. Research Metrics (`metrics.py`)
- Track: personality consistency, stress coherence, latent dynamics, reward/loss

### 12. Backend (`main.py`)
- FastAPI + WebSocket streaming
- Thread pool riêng: LLM / tools / audio
- Semaphore giới hạn tool execution

### 13. Audio System (`audio.py`)
- TTS: `edge-tts`
- STT: `speech_recognition`
- Async loop (không lag UI)

---

## ✅ Key Features

**Latent-conditioned behavior**
Không phải `prompt = "be calm"` mà là `latent state → control LLM`

**Deferred reward system**
Reward không tính ngay — inject vào turn sau, giống RL thực

**Non-blocking architecture**
Embedding async, tool async, audio async, LLM lock-safe

**Safety layers**
Tool Guard → Tool Verifier → Hallucination Detector → Self-Evaluator

**Research-ready**
Metrics logging, reflection experiments, latent conditioning modes

---

## ⚠️ Limitations

- RSSM chưa fully trained
- Evaluation signal còn noisy
- Chưa có long-term learning pipeline
- Không phải AGI

---

## 🧪 Research Mode

```python
# config.py
NangConfig.RESEARCH_MODE = True   # bật logging + metrics
NangConfig.BASELINE_MODE = True   # chỉ log latent, không inject
NangConfig.BASELINE_MODE = False  # inject latent vào LLM (experimental)
```

Khi bật Research Mode:
- Latent adapter hoạt động
- Reflection engine chạy
- Metrics logging ghi đầy đủ
- UI hiển thị: Reward, Stress, Consistency, Decision, Hallucination

---

## 🚀 Run

```bash
pip install -r requirements.txt
python main.py
```

Mở trình duyệt: `http://localhost:8000`

---

## 📁 File Structure

```
NắngAI/
├── main.py                    # FastAPI server + WebSocket
├── brain.py                   # LLM orchestrator + generation
├── soul.py                    # RSSM World Model (Dreamer)
├── env.py                     # Conversation environment
├── memory.py                  # Long-term memory (ChromaDB)
├── latent_adapter.py          # Latent → LLM bridge
├── sentiment.py               # Embedding-based sentiment
├── tools.py                   # Tool system + ToolGuard
├── tool_verifier.py           # LLM-based tool output verification
├── self_evaluator.py          # Self-evaluation engine
├── hallucination_detector.py  # Embedding-based hallucination check
├── reflection.py              # Reflection + regeneration
├── metrics.py                 # Research metrics tracking
├── audio.py                   # TTS + STT
├── config.py                  # Configuration
├── utils.py                   # Utilities
└── index.html                 # Web UI
```

---

## 📊 Stats

- **~5.8k lines of core logic (~7–8k including UI & infrastructure)**
- **17 modules**
- **Hardware:** RTX 3050 6GB
- **Models:** Qwen 2.5 3B / Hermes 3 8B (4-bit quantized)

---

## 🧠 Philosophy

```
Không cần train model lớn hơn.

Chỉ cần:
  - control behavior tốt hơn
  - hiểu state tốt hơn
  - phản hồi thông minh hơn
```

> *Reusing ideas is fine. Reusing code without attribution is not.*

---

## 📜 License

**CC BY-NC 4.0** — Free cho cá nhân và nghiên cứu, cấm thương mại.

> Bạn được phép: sử dụng, chỉnh sửa, chia sẻ với điều kiện ghi nguồn.  
> Bạn không được phép: sử dụng cho mục đích thương mại dưới bất kỳ hình thức nào.

---

## 👤 Author

**Long** — Vũng Tàu, Việt Nam

*Tự học, không có background IT chính quy. Build bằng cách làm thật, fail thật, fix thật.*

---

> ⚠️ **Note:** This is a research-oriented system focused on behavior control, not a full-scale training framework. RSSM conditioning is experimental.
