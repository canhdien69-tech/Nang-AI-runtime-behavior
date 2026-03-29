# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/hallucination_detector.py
# MỤC ĐÍCH: Hallucination Detection
#
# Detect hallucination dựa trên:
#   1. Semantic contradiction — response embedding mâu thuẫn với tool/RAG context
#   2. Confidence score — cosine sim giữa response và source context
#   3. Length anomaly — response quá dài so với context = potential fabrication
# ==============================================================================

import numpy as np
from dataclasses import dataclass
from config import NangConfig, logger


@dataclass
class HallucinationResult:
    is_hallucination:   bool
    confidence:         float
    semantic_sim:       float
    semantic_sim_raw:   float   # raw value (có thể âm) — dùng để detect contradiction
    length_ratio:       float
    reason:             str = ""


class HallucinationDetector:
    """
    Detect hallucination không cần thêm LLM call — dùng embedding similarity.

    Approach:
      - Embed response + source context (tool result + RAG)
      - Cosine similarity thấp → response không dựa trên context → potential hallucination
      - Length ratio cao → response dài hơn nhiều so với source → fabrication risk
    """

    def __init__(self, embed_model, embed_lock):
        self._embed      = embed_model
        self._embed_lock = embed_lock
        logger.info("[HallucinationDetector] Init OK")

    def _embed_text(self, text: str) -> np.ndarray:
        with self._embed_lock:
            return self._embed.encode(text[:500], normalize_embeddings=True)

    def detect(
        self,
        response:    str,
        tool_result: str = "",
        rag_context: str = "",
    ) -> HallucinationResult:
        """
        Detect hallucination trong response.

        Args:
            response:    LLM response cần check
            tool_result: Tool output đã dùng (nếu có)
            rag_context: RAG context đã inject (nếu có)
        """
        if not NangConfig.HALLUCINATION_CHECK:
            return HallucinationResult(False, 1.0, 1.0, 1.0, "disabled")

        if not response:
            return HallucinationResult(False, 1.0, 1.0, 1.0, "empty response")

        # Nếu không có source context → không thể verify
        source = " ".join(filter(None, [tool_result, rag_context])).strip()
        if not source or len(source) < 20:
            return HallucinationResult(False, 0.5, 0.5, 1.0, "no source context")

        try:
            resp_emb   = self._embed_text(response)
            source_emb = self._embed_text(source)

            # 1. Semantic similarity
            semantic_sim = float(np.dot(resp_emb, source_emb))
            # [#1] Giữ raw value — không clamp âm để detect contradiction
            sim_pos = max(0.0, semantic_sim)   # dùng cho confidence

            # 2. Length ratio
            length_ratio = len(response) / max(len(source), 1)

            # [#3] Confidence có penalty theo length anomaly
            length_penalty = min(1.0, (length_ratio - 1.0) / 3.0)
            length_penalty = min(1.0, (length_ratio - 1.0) / 3.0)
            confidence = sim_pos / (1.0 + 0.2 * max(0.0, length_ratio - 1.0))
            confidence *= (1.0 - 0.3 * length_penalty)

            # 3. Hallucination verdict
            is_hallucination = False
            reason           = "ok"

            if semantic_sim < -0.2:
                # [#1] Semantic contradiction — response NGƯỢC nghĩa source
                is_hallucination = True
                reason           = f"semantic_contradiction={semantic_sim:.3f}"
            elif sim_pos < NangConfig.HALLUCINATION_THRESH:
                is_hallucination = True
                reason           = f"low_semantic_sim={sim_pos:.3f}"
            elif length_ratio > 2.5 and sim_pos < NangConfig.HALLUCINATION_THRESH + 0.1:
                # [#2] Length anomaly với threshold chặt hơn
                is_hallucination = True
                reason           = f"length_anomaly={length_ratio:.1f}x + low_sim={sim_pos:.3f}"
                confidence       = confidence * 0.5

            if is_hallucination:
                logger.warning(
                    f"[Hallucination] Detected: sim={sim_pos:.3f} raw={semantic_sim:.3f} "
                    f"len_ratio={length_ratio:.1f} reason={reason}"
                )

            return HallucinationResult(
                is_hallucination = is_hallucination,
                confidence       = round(confidence, 4),
                semantic_sim     = round(sim_pos, 4),
                semantic_sim_raw = round(semantic_sim, 4),
                length_ratio     = round(length_ratio, 2),
                reason           = reason,
            )

        except Exception as e:
            logger.warning(f"[HallucinationDetector] detect failed: {e}")
            return HallucinationResult(False, 0.5, 0.5, 1.0, f"error: {e}")
