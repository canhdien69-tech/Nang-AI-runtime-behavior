# ==============================================================================
# PROJECT: NẮNG AI - v47.5
# FILE: memory.py
# MỤC ĐÍCH: Long-term memory với ChromaDB + SentenceTransformer (RAG)
# UPDATES v47.5:
#   [M1]  Cache _cached_count → tránh gọi .count() O(N) liên tục
#   [M2]  Eviction dùng ts_unix filter thay full scan O(N log N)
#   [M3]  Embedding async trong background thread → không block main thread
#   [M4]  Write filter: chỉ lưu khi importance đủ cao (confidence + length)
#   [M5]  Dedup: skip nếu cosine similarity > 0.95 với entry gần nhất
#   [M6]  Composite score = similarity * importance * recency
#   [M7]  Failure memory riêng: lưu tool fail / panic events
#   [M8]  RAG inject kèm uncertainty tag [ĐỘ TIN CẬY: X.XX]
#   [M9]  Memory → Dreamer: trả về failure_penalty để reward shaping
# ==============================================================================

import json, datetime, uuid, os, threading, time, math
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import NangConfig, CONF, logger
from utils import SystemUtils


class LongTermMemory:
    """
    Bộ nhớ dài hạn dùng ChromaDB + multilingual embedding.

    Hai collection:
      - episodic: lịch sử hội thoại (có write filter + dedup)
      - failure:  tool fail, panic, hallucination events

    Memory → Dreamer: get_failure_penalty() để reward shaping.
    """

    # Ngưỡng importance tối thiểu để lưu vào long-term memory
    # 0 = luôn lưu, 1 = chỉ lưu khi rất quan trọng
    IMPORTANCE_THRESHOLD = 0.2
    # Cosine sim > ngưỡng này → coi là duplicate, skip
    DEDUP_THRESHOLD      = 0.95
    # Failure penalty inject vào Dreamer reward
    FAILURE_PENALTY      = -0.5

    def __init__(self, embed_model=None, embed_lock=None):
        self._client = chromadb.PersistentClient(
            path=NangConfig.CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        # Episodic memory collection
        self._col = self._client.get_or_create_collection(
            name=NangConfig.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        # [M7] Failure memory collection riêng
        self._fail_col = self._client.get_or_create_collection(
            name=f"{NangConfig.CHROMA_COLLECTION}_failures",
            metadata={"hnsw:space": "cosine"}
        )

        # [1.1] Nhận model từ ngoài nếu có, tránh load 2 lần
        if embed_model is not None:
            self._embed = embed_model
            logger.info("[Memory] Using injected embed model.")
        else:
            logger.info(f"[Memory] Loading embed model: {NangConfig.EMBED_MODEL}")
            self._embed = SentenceTransformer(NangConfig.EMBED_MODEL, device="cpu")

        # [M1] Cache count để tránh gọi .count() O(N) liên tục
        self._cached_count = self._col.count()
        logger.info(f"[Memory] ChromaDB ready — {self._cached_count} episodic entries")

        # [2.2] Lock cho JSON write
        self._json_lock = threading.Lock()

        # [M3] Queue + background thread cho async embedding
        import queue as _queue
        self._embed_queue: _queue.Queue = _queue.Queue(maxsize=50)
        self._embed_thread = threading.Thread(
            target=self._background_embed_worker, daemon=True
        )
        self._embed_thread.start()
        # [#2] Lock cho SentenceTransformer — không thread-safe khi gọi đồng thời
        # Dùng shared lock từ brain.py nếu có — đảm bảo toàn bộ hệ thống dùng 1 lock
        self._embed_lock  = embed_lock if embed_lock is not None else threading.Lock()
        # [#1] Lock cho _cached_count — tránh race condition giữa background và main thread
        self._count_lock  = threading.Lock()

        # Short-term buffer
        self._recent: list[dict] = self._load_recent_from_json()

        # [M9] Failure penalty buffer
        self._recent_failures: list[dict] = []

        # [M10] Memory usefulness tracking — paper-level metric
        # Buffer keyed by turn_id — tránh lệch khi multi-request async
        self._usefulness_buffer: dict = {}   # {turn_id: {doc_id: emb}}
        self._useful_lock = threading.Lock()

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------
    def _load_recent_from_json(self) -> list:
        try:
            with open(CONF["FILES"]["MEMORY"], 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
        except Exception as e:
            logger.warning(f"[Memory] load_recent_json failed: {e}")
            return []

    def _save_recent_json(self):
        """[2.2] Thread-safe JSON persist."""
        with self._json_lock:
            try:
                with open(CONF["FILES"]["MEMORY"], 'w', encoding='utf-8') as f:
                    json.dump(self._recent, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"[Memory] save_recent_json failed: {e}")

    def _embed_text(self, text: str) -> list[float]:
        # [#2] Lock để tránh race condition khi main thread và background thread
        # cùng gọi encode() — SentenceTransformer không guarantee thread-safe
        with self._embed_lock:
            return self._embed.encode(text, normalize_embeddings=True).tolist()

    def _compute_importance(self, user_msg: str, ai_response: str, tool_confidence: bool) -> float:
        """
        [M4] Tính importance score để quyết định có lưu long-term không.
        Score [0, 1] dựa trên:
          - độ dài message (dài = thường có thông tin hơn)
          - tool_confidence (có data thực = quan trọng hơn)
          - không phải câu quá ngắn / filler
        """
        # Filler phrases thường không cần nhớ
        FILLER = ["ok", "oke", "okay", "được", "vâng", "ừ", "uhm", "hmm", "hi", "hello"]
        msg_lower = user_msg.lower().strip()
        if msg_lower in FILLER or len(msg_lower) < 5:
            return 0.0

        # Base score từ độ dài
        length_score = min(1.0, (len(user_msg) + len(ai_response)) / 500)
        # Tool confidence bonus
        confidence_bonus = 0.3 if tool_confidence else 0.0
        # Question / instruction bonus
        question_bonus = 0.2 if any(c in user_msg for c in "?!") else 0.0

        return min(1.0, length_score + confidence_bonus + question_bonus)

    def _is_duplicate(self, emb: list[float]) -> bool:
        """
        [M5] Kiểm tra xem embedding có quá giống entry gần nhất không.
        Dùng cosine similarity với top-1 query.
        """
        with self._count_lock:
            cached = self._cached_count
        if cached == 0:
            return False
        try:
            results = self._col.query(
                query_embeddings=[emb],
                n_results=1,
                include=["distances"]
            )
            if results["distances"] and results["distances"][0]:
                sim = 1.0 - results["distances"][0][0]
                if sim >= self.DEDUP_THRESHOLD:
                    logger.debug(f"[Memory] Dedup skip: similarity={sim:.3f}")
                    return True
        except Exception:
            pass
        return False

    def _composite_score(self, similarity: float, ts_iso: str, importance: float = 0.5) -> float:
        """
        [M6] Composite memory score = similarity * importance * recency_decay.
        recency_decay = exp(-days_old / 30) — ký ức 30 ngày tuổi còn ~37%
        """
        try:
            ts   = datetime.datetime.fromisoformat(ts_iso)
            days = (datetime.datetime.now() - ts).total_seconds() / 86400
            recency = math.exp(-days / 30.0)
        except Exception:
            recency = 0.5
        return round(similarity * importance * recency, 4)

    def _background_embed_worker(self):
        """
        [M3] Background thread xử lý ChromaDB add.
        Embedding + DB write được offload ra đây để không block main thread.
        """
        import queue as _queue
        while True:
            task = None
            try:
                task = self._embed_queue.get(timeout=5)
                if task is None:
                    self._embed_queue.task_done()
                    break
                doc_id, combined, user_msg, ai_response, ts, importance = task
                emb = self._embed_text(combined)

                if self._is_duplicate(emb):
                    continue

                self._col.add(
                    ids=[doc_id],
                    embeddings=[emb],
                    documents=[combined],
                    metadatas=[{
                        "user":       user_msg,
                        "ai":         ai_response,
                        "ts":         ts,
                        "ts_unix":    datetime.datetime.fromisoformat(ts).timestamp(),
                        "importance": importance,
                    }]
                )
                with self._count_lock:
                    self._cached_count += 1
                self._evict_if_needed()
            except _queue.Empty:
                continue   # queue trống sau 5s — bình thường, không cần log
            except Exception as e:
                logger.warning(f"[Memory] background_embed_worker error: {e}")
            finally:
                # [#7] Chỉ gọi task_done() khi get() thành công — tránh counter desync
                if task is not None:
                    try: self._embed_queue.task_done()
                    except Exception: pass

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def save_interaction(
        self,
        user_msg: str,
        ai_response: str,
        tool_confidence: bool = False
    ):
        """
        Lưu lượt hội thoại.
        [M4] Write filter: chỉ lưu long-term nếu importance đủ cao.
        [M3] Embedding + DB write async trong background thread.
        """
        ts = datetime.datetime.now().isoformat()
        entry = {"user": user_msg, "ai": ai_response, "ts": ts}

        # Short-term buffer — LUÔN lưu (cần cho context ngắn hạn)
        self._recent.append(entry)
        if len(self._recent) > NangConfig.MEMORY_LIMIT:
            self._recent = self._recent[-NangConfig.MEMORY_LIMIT:]
        self._save_recent_json()

        # [M4] Long-term: chỉ lưu khi importance đủ cao
        importance = self._compute_importance(user_msg, ai_response, tool_confidence)
        if importance >= self.IMPORTANCE_THRESHOLD:
            combined = f"Người dùng: {user_msg}\nNắng: {ai_response}"
            doc_id   = f"turn_{uuid.uuid4().hex[:12]}"
            try:
                # [M3] Đẩy vào queue — background thread xử lý
                self._embed_queue.put_nowait(
                    (doc_id, combined, user_msg, ai_response, ts, importance)
                )
            except Exception:
                logger.warning(f"[Memory] embed_queue full — DROP (importance={importance:.2f})")
        else:
            logger.debug(f"[Memory] Skip long-term (importance={importance:.2f} < threshold)")

        # Diary log
        try:
            SystemUtils.rotate_log()
            with open(CONF["FILES"]["DIARY"], 'a', encoding='utf-8') as f:
                f.write(f"[{ts}] U: {user_msg}\nA: {ai_response}\n---\n")
        except Exception as e:
            logger.warning(f"[Memory] diary write failed: {e}")

    def save_failure(self, event_type: str, detail: str):
        """
        [M7] Lưu failure event vào failure memory riêng.
        event_type: "tool_fail" | "panic" | "timeout" | "hallucination"
        Dùng cho: reward shaping (M9), failure pattern analysis.
        """
        ts = datetime.datetime.now().isoformat()
        combined = f"[{event_type}] {detail}"
        try:
            emb = self._embed_text(combined)
            self._fail_col.add(
                ids=[f"fail_{uuid.uuid4().hex[:10]}"],
                embeddings=[emb],
                documents=[combined],
                metadatas=[{
                    "event_type": event_type,
                    "detail":     detail[:500],
                    "ts":         ts,
                    "ts_unix":    datetime.datetime.fromisoformat(ts).timestamp(),
                }]
            )
            # Buffer gần nhất cho reward shaping — lưu ts_unix để tránh parse lại
            self._recent_failures.append({
                "type":    event_type,
                "ts":      ts,
                "ts_unix": datetime.datetime.fromisoformat(ts).timestamp(),
            })
            if len(self._recent_failures) > 20:
                self._recent_failures = self._recent_failures[-20:]
            logger.info(f"[Memory] Failure saved: {event_type} — {detail[:60]}")
        except Exception as e:
            logger.warning(f"[Memory] save_failure failed: {e}")

    def get_failure_penalty(self, query_text: str, window_minutes: int = 10) -> float:
        """
        [M9] Trả về penalty score để inject vào Dreamer reward.
        [#5] Dùng ts_unix thay parse ISO datetime mỗi lần — O(1) thay O(N) parse.
        """
        if not self._recent_failures:
            return 0.0
        cutoff_unix = time.time() - window_minutes * 60
        recent = [
            f for f in self._recent_failures
            if f.get("ts_unix", 0.0) > cutoff_unix
        ]
        if not recent:
            return 0.0
        return self.FAILURE_PENALTY * min(len(recent) / 3.0, 1.0)

    def query(self, text: str, top_k: int = NangConfig.RAG_TOP_K) -> tuple:
        """
        RAG query với composite score = similarity * importance * recency.
        [M10] Trả về (memories, turn_id) — turn_id dùng để match với update_usefulness().
        Keyed buffer tránh lệch khi multi-request async.
        """
        with self._count_lock:
            cached = self._cached_count
        if cached == 0:
            return [], None
        try:
            emb     = self._embed_text(text)
            turn_id = uuid.uuid4().hex[:8]   # unique key cho turn này
            results = self._col.query(
                query_embeddings=[emb],
                n_results=min(top_k, cached),
                include=["metadatas", "distances", "embeddings"]
            )
            out          = []
            turn_buffer  = {}

            for i, (meta, dist) in enumerate(zip(results["metadatas"][0], results["distances"][0])):
                sim = 1.0 - dist
                if sim < NangConfig.RAG_MIN_SCORE:
                    continue
                importance  = meta.get("importance", 0.5)
                score       = self._composite_score(sim, meta.get("ts", ""), importance)
                doc_id      = results["ids"][0][i] if results.get("ids") else None
                usefulness  = meta.get("usefulness", 0.5)

                if doc_id and results.get("embeddings") and results["embeddings"][0]:
                    turn_buffer[doc_id] = results["embeddings"][0][i]

                out.append({
                    "doc_id":     doc_id,
                    "user":       meta.get("user", ""),
                    "ai":         meta.get("ai", ""),
                    "ts":         meta.get("ts", ""),
                    "score":      score,
                    "similarity": round(sim, 3),
                    "importance": round(importance, 3),
                    "usefulness": round(usefulness, 3),
                })

            # Lưu buffer theo turn_id — không clear buffer cũ của turn khác
            with self._useful_lock:
                # Guard size + TTL cleanup — tránh leak khi turn bị cancel/timeout
                now = time.time()
                if len(self._usefulness_buffer) > 100:
                    self._usefulness_buffer.clear()
                    logger.debug("[Memory] usefulness_buffer overflow — cleared")
                else:
                    # Xóa entries cũ hơn 60s — turn đã bị drop
                    self._usefulness_buffer = {
                        k: v for k, v in self._usefulness_buffer.items()
                        if now - v[0] < 60
                    }
                self._usefulness_buffer[turn_id] = (time.time(), turn_buffer)
            return out, turn_id
        except Exception as e:
            logger.warning(f"[Memory] query failed: {e}")
            return [], None

    def update_usefulness(self, response: str, turn_id: str = None):
        """
        [M10] Đo và update memory usefulness sau mỗi turn.
        [#1] turn_id — pop đúng buffer của turn này, không lệch async
        [#2] Chỉ update khi usefulness > 0.7 — giảm DB write load
        [#3] max(0, dot) thay normalize về [0,1] — giữ signal mạnh
        """
        if turn_id is None:
            return
        with self._useful_lock:
            entry = self._usefulness_buffer.pop(turn_id, None)
        if not entry:
            return
        _, buffer = entry   # unpack (timestamp, turn_buffer)
        try:
            import numpy as _np
            resp_emb = _np.array(self._embed_text(response[:500]))

            for doc_id, mem_emb in buffer.items():
                mem_arr    = _np.array(mem_emb)
                # [#3] max(0, dot) — giữ signal mạnh, không flatten distribution
                usefulness = float(max(0.0, _np.dot(resp_emb, mem_arr)))
                # [#2] Chỉ update khi đủ high — tránh spam DB write
                if usefulness < 0.7:
                    continue
                try:
                    existing = self._col.get(ids=[doc_id], include=["metadatas"])
                    if existing and existing["metadatas"]:
                        meta           = existing["metadatas"][0]
                        old_usefulness = meta.get("usefulness", 0.5)
                        meta["usefulness"] = round(0.7 * old_usefulness + 0.3 * usefulness, 4)
                        self._col.update(ids=[doc_id], metadatas=[meta])
                        logger.debug(
                            f"[Memory] Usefulness: {doc_id[:8]} "
                            f"{old_usefulness:.3f} → {meta['usefulness']:.3f}"
                        )
                except Exception as e:
                    logger.debug(f"[Memory] usefulness update failed {doc_id}: {e}")

        except Exception as e:
            logger.warning(f"[Memory] update_usefulness failed: {e}")

    def get_avg_usefulness(self) -> float:
        """
        [M10] Trả về mean usefulness của session hiện tại.
        Dùng trong research/metrics.py — "Memory Utilization Rate".
        """
        try:
            with self._count_lock:
                cached = self._cached_count
            if cached == 0:
                return 0.0
            sample = self._col.get(
                limit=min(100, cached),
                include=["metadatas"]
            )
            scores = [
                m.get("usefulness", 0.5)
                for m in sample["metadatas"]
                if "usefulness" in m
            ]
            return round(float(__import__('numpy').mean(scores)), 4) if scores else 0.5
        except Exception:
            return 0.5

    def get_recent(self, n: Optional[int] = None) -> list[dict]:
        if n is None:
            n = NangConfig.MEMORY_LIMIT
        return self._recent[-n:]

    def _evict_if_needed(self):
        """
        [M2] Eviction dùng ts_unix thay full scan O(N log N).
        Lấy entries cũ nhất bằng filter range query nếu ChromaDB support,
        fallback về full scan chỉ khi cần.
        """
        try:
            if self._cached_count <= NangConfig.MAX_CHROMA_ENTRIES:
                return
            n_delete = max(1, self._cached_count // 10)
            # Lấy n_delete entries cũ nhất — ChromaDB không có native ORDER BY
            # nhưng có thể lấy với limit rồi sort phía Python
            # Lấy gấp đôi để có buffer tốt hơn khi sort
            sample = self._col.get(
                limit=min(n_delete * 2, self._cached_count),
                include=["metadatas"]
            )
            ids_ts = [
                (doc_id, meta.get("ts_unix", 0.0))
                for doc_id, meta in zip(sample["ids"], sample["metadatas"])
            ]
            ids_ts.sort(key=lambda x: x[1])   # sort theo ts_unix tăng dần
            ids_to_delete = [x[0] for x in ids_ts[:n_delete]]
            self._col.delete(ids=ids_to_delete)
            with self._count_lock:
                self._cached_count -= len(ids_to_delete)
            logger.info(f"[Memory] Evicted {len(ids_to_delete)} entries. Cached count: {self._cached_count}")
        except Exception as e:
            logger.warning(f"[Memory] eviction failed: {e}")

    def format_rag_context(self, query_text: str, top_k: Optional[int] = None) -> tuple:
        """
        [M8] RAG context kèm uncertainty + usefulness tags.
        Trả về (context_str, turn_id) — turn_id dùng cho update_usefulness().
        """
        effective_top_k = top_k if top_k is not None else NangConfig.RAG_TOP_K
        memories, turn_id = self.query(query_text, top_k=effective_top_k)
        if not memories:
            return "", turn_id
        memories = sorted(memories, key=lambda m: m.get("score", 0.0), reverse=True)
        lines = ["[KÝ ỨC DÀI HẠN — những lần em đã được dạy / trò chuyện liên quan]:"]
        for m in memories:
            similarity = m.get("similarity", 0.0)
            usefulness = m.get("usefulness", 0.5)
            confidence = f"{similarity:.2f}"
            useful_tag = f"[ĐỘ HỮU ÍCH: {usefulness:.2f}]" if usefulness != 0.5 else ""
            lines.append(
                f"  • [{m['ts'][:10]}][ĐỘ TIN CẬY: {confidence}]{useful_tag} "
                f"Anh: {m['user'][:120]}"
            )
            lines.append(f"    Em: {m['ai'][:120]}")
        return "\n".join(lines), turn_id
