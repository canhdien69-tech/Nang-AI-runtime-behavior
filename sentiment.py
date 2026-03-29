# ==============================================================================
# PROJECT: NẮNG AI - v47.6
# FILE: sentiment.py
# MỤC ĐÍCH: Sentiment Analysis dựa trên embedding cosine similarity
# UPDATES v47.6:
#   [S1] top-k mean thay max cosine — tránh spike từ 1 anchor match yếu
#   [S2] normalize formula (neg-pos)/(|neg|+|pos|+eps) — stable hơn naive subtraction
#   [S3] SENTIMENT_SCALE từ config thay hardcode 1.5
#   [S4] Simple LRU cache cho encode — tránh re-encode cùng text
#   [S5] Guard empty anchors → return 0.0
#   [S6] Nonlinear blend — embedding là base, keyword là spike
#   [S7] Context-aware score() — pass thêm recent context để detect escalation
# ==============================================================================

import numpy as np
import threading
from sentence_transformers import SentenceTransformer

from config import NangConfig, logger


class SentimentScorer:
    """
    Score cảm xúc câu nói dựa trên cosine similarity với anchor phrases.
    Trả về stress_score: float trong [0.0, STRESS_CAP]

    Logic v47.6:
      neg_score = top-k mean cosine sim với NEG anchors  [S1]
      pos_score = top-k mean cosine sim với POS anchors  [S1]
      raw_stress = (neg-pos) / (|neg|+|pos|+eps)         [S2]
      stress = clip(raw * SENTIMENT_SCALE, 0.0, STRESS_CAP) [S3]

    blend():
      embedding là base, keyword là spike              [S6]
      final = max(emb, keyword*(1-w)) + w*emb (clipped)
    """

    # Top-k anchors dùng cho mean — không dùng toàn bộ tránh noise từ anchors xa
    _TOPK = 3

    def __init__(self, embed_model: SentenceTransformer, embed_lock=None):
        self._embed = embed_model
        # [#3] Dùng shared lock từ brain.py — đảm bảo chung lock với memory + brain
        self._cache_lock = embed_lock if embed_lock is not None else threading.Lock()
        self._encode_cache: dict = {}
        self._cache_maxsize = 512

        # Pre-encode anchors một lần khi khởi động
        self._neg_embs = self._encode_anchors(NangConfig.SENTIMENT_NEG_ANCHORS)
        self._pos_embs = self._encode_anchors(NangConfig.SENTIMENT_POS_ANCHORS)
        logger.info(
            f"[Sentiment] Anchors encoded: "
            f"{len(NangConfig.SENTIMENT_NEG_ANCHORS)} neg, "
            f"{len(NangConfig.SENTIMENT_POS_ANCHORS)} pos."
        )

    def _encode_anchors(self, phrases: list) -> np.ndarray:
        """Encode list anchor phrases thành matrix (N, dim)."""
        if not phrases:
            return np.zeros((0, 1), dtype=np.float32)
        return self._embed.encode(phrases, normalize_embeddings=True)

    def _encode_text(self, text: str) -> np.ndarray:
        """
        [S4] Encode text với simple dict cache.
        lru_cache không hoạt động trên method có self — dùng dict cache thay.
        Evict khi vượt maxsize (drop oldest half).
        """
        with self._cache_lock:
            if text in self._encode_cache:
                # [LRU] Move to end — most recently used
                emb = self._encode_cache.pop(text)
                self._encode_cache[text] = emb
                return emb
        emb = self._embed.encode(text, normalize_embeddings=True)
        with self._cache_lock:
            while len(self._encode_cache) >= self._cache_maxsize:
                # [LRU] Pop oldest (first inserted = least recently used)
                self._encode_cache.pop(next(iter(self._encode_cache)))
            self._encode_cache[text] = emb
        return emb

    def _cosine_topk_mean(self, text_emb: np.ndarray, anchor_embs: np.ndarray) -> float:
        """
        [S1] Top-k mean cosine similarity thay vì max.
        Stable hơn: cần nhiều anchors match mới kéo score lên,
        không bị spike từ 1 anchor match yếu.
        [S5] Guard empty anchors → return 0.0
        """
        if len(anchor_embs) == 0:
            return 0.0
        sims = anchor_embs @ text_emb   # (N,) — dot product = cosine vì đã normalize
        k    = min(self._TOPK, len(sims))
        topk = np.partition(sims, -k)[-k:]
        return float(np.mean(topk))

    def score(self, text: str, context: str = "") -> float:
        """
        [S7] Context-aware score — pass recent context để detect escalation/sarcasm.
        Trả về embedding_stress: float [0.0, STRESS_CAP].
        Gọi blend() để kết hợp với keyword stress.
        """
        try:
            # [S7] Nếu có context, combine để detect escalation và sarcasm tốt hơn
            combined = f"{context} {text}".strip() if context else text
            emb = self._encode_text(combined)

            neg = self._cosine_topk_mean(emb, self._neg_embs)
            pos = self._cosine_topk_mean(emb, self._pos_embs)

            # [S2] Normalized formula thay vì naive subtraction
            # (neg - pos) / (|neg| + |pos| + eps) → range [-1, 1] → map sang [0, 1]
            denom  = abs(neg) + abs(pos) + 1e-6
            raw    = (neg - pos) / denom   # [-1, 1]
            # [S2] max(0, raw) — chỉ penalize negative sentiment
            # (raw+1)/2 làm neutral=0.5 bias → system luôn có stress baseline
            # max(0, raw) → negative=stress, neutral/positive=0
            stress = max(0.0, raw)
            # [S3] Dùng SENTIMENT_SCALE từ config thay hardcode
            return float(np.clip(stress * NangConfig.SENTIMENT_SCALE, 0.0, NangConfig.STRESS_CAP))
        except Exception as e:
            logger.warning(f"[Sentiment] score() failed: {e} — fallback 0.0")
            return 0.0

    def blend(self, text: str, keyword_stress: float, context: str = "") -> float:
        """
        [S6] Nonlinear blend — embedding là base, keyword là spike.
        Thay vì linear: w*emb + (1-w)*keyword
        Dùng: max(emb, keyword*(1-w)) + w*emb — clamped
        → keyword mạnh bất ngờ vẫn được capture
        → embedding là floor, không bị override hoàn toàn
        [S7] Pass context xuống score() để detect sarcasm/escalation.
        """
        w          = NangConfig.SENTIMENT_WEIGHT
        emb_stress = self.score(text, context=context)
        # [S6] Nonlinear blend
        blended    = max(emb_stress, keyword_stress * (1.0 - w)) + w * emb_stress
        return float(np.clip(blended, 0.0, NangConfig.STRESS_CAP))
