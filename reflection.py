# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/reflection.py
# MỤC ĐÍCH: Latent-Guided Self-Reflection
#
# CONTRIBUTION:
#   Self-reflection trong LLM đã có (Constitutional AI, Self-RAG).
#   Nhưng chưa ai dùng WORLD MODEL LATENT STATE (h, z từ Dreamer)
#   làm signal để guide reflection — đây là contribution thứ 2 của Nắng AI.
#
#   Thay vì rule-based ("đừng nói X"), reflection ở đây là learned:
#   RSSM latent state encode emotional dynamics → nếu response không
#   consistent với latent state → reflect → improve.
#
# PIPELINE:
#   response → embed → compare với latent (h,z) projected → consistency_score
#   nếu score < threshold → reflection prompt → regenerate
#
# MODES:
#   REFLECT_ONLY  — chỉ score, không regenerate (data collection)
#   REFLECT_REGEN — score + regenerate nếu thấp (production use)
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from config import NangConfig, logger


@dataclass
class ReflectionResult:
    """Kết quả của 1 reflection step."""
    original_response:    str
    reflected_response:   str           # = original nếu không regenerate
    consistency_score:    float         # [0, 1] — cao = consistent với latent
    latent_valence:       float         # estimated emotional valence từ h
    response_valence:     float         # estimated valence từ response embedding
    did_reflect:          bool          # True nếu đã regenerate
    reflection_delta:     float         # consistency_score sau - trước (nếu reflect)


class LatentConsistencyScorer(nn.Module):
    """
    Đo consistency giữa RSSM latent state và response embedding.

    Input:  h (det_dim,), z (stoch_dim,), response_emb (embed_dim,)
    Output: consistency_score ∈ [0, 1]

    Architecture:
      latent (h+z) → MLP → latent_proj (proj_dim,)
      response_emb → Linear → resp_proj (proj_dim,)
      score = cosine_similarity(latent_proj, resp_proj)

    Training signal: reward từ env (high reward = high consistency).
    """

    def __init__(
        self,
        latent_dim:  int = NangConfig.CONV_DET_DIM + NangConfig.CONV_STOCH_DIM,
        embed_dim:   int = NangConfig.OBS_DIM,
        proj_dim:    int = 64,
        hidden_dim:  int = 128,
    ):
        super().__init__()
        self.proj_dim = proj_dim

        # Latent → projection space
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Response embedding → projection space
        self.resp_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        # Init nhỏ
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        h:            torch.Tensor,   # (batch, det_dim)
        z:            torch.Tensor,   # (batch, stoch_dim)
        response_emb: torch.Tensor,   # (batch, embed_dim)
    ) -> torch.Tensor:
        """Trả về consistency score (batch,) ∈ [0, 1]."""
        latent      = torch.cat([h, z], dim=-1)
        lat_proj    = self.latent_proj(latent)             # (batch, proj_dim)
        resp_proj   = self.resp_proj(response_emb)          # (batch, proj_dim)

        # Normalize rồi cosine similarity
        lat_norm    = nn.functional.normalize(lat_proj, dim=-1)
        resp_norm   = nn.functional.normalize(resp_proj, dim=-1)
        cos_sim     = (lat_norm * resp_norm).sum(dim=-1)   # (batch,) ∈ [-1, 1]

        # Map [-1, 1] → [0, 1]
        return (cos_sim + 1.0) / 2.0


class ReflectionEngine:
    """
    Latent-Guided Self-Reflection Engine.

    Workflow:
      1. Embed response bằng SentenceTransformer
      2. Score consistency với RSSM latent (h, z)
      3. Nếu score < threshold → tạo reflection prompt → regenerate
      4. Log ReflectionResult cho research metrics

    Mode:
      REFLECT_ONLY  — chỉ score (dùng khi collect baseline data)
      REFLECT_REGEN — score + regenerate (dùng khi test reflection)
    """

    MODE_ONLY  = "reflect_only"
    MODE_REGEN = "reflect_regen"

    # Threshold dưới ngưỡng này → trigger reflection
    CONSISTENCY_THRESHOLD = 0.45

    def __init__(
        self,
        embed_model,          # SentenceTransformer — shared với brain
        embed_lock,           # threading.Lock — shared với brain
        tokenizer,            # LLM tokenizer
        model,                # LLM model
        gen_lock,             # RLock — shared với brain.think()
        device,
        mode: str = MODE_ONLY,
    ):
        self._embed       = embed_model
        self._embed_lock  = embed_lock
        self._tokenizer   = tokenizer
        self._model       = model
        self._gen_lock    = gen_lock
        self._device      = device
        self.mode         = mode

        _embed_dim = embed_model.get_sentence_embedding_dimension()
        self._scorer = LatentConsistencyScorer(embed_dim=_embed_dim).to(device)
        logger.info(f"[Reflection] Init — mode: {mode}, threshold: {self.CONSISTENCY_THRESHOLD}")

    def _embed_text(self, text: str) -> np.ndarray:
        with self._embed_lock:
            return self._embed.encode(text, normalize_embeddings=True)

    def _score(
        self,
        h_np:         np.ndarray,
        z_np:         np.ndarray,
        response_emb: np.ndarray,
    ) -> float:
        """Tính consistency score — không train, chỉ inference."""
        with torch.no_grad():
            h   = torch.tensor(h_np, dtype=torch.float32).to(self._device)
            z   = torch.tensor(z_np, dtype=torch.float32).to(self._device)
            emb = torch.tensor(response_emb, dtype=torch.float32).unsqueeze(0).to(self._device)

            # Reshape nếu cần
            if h.dim() == 1: h = h.unsqueeze(0)
            if z.dim() == 1: z = z.unsqueeze(0)

            score = self._scorer(h, z, emb)
            return float(score.squeeze().cpu())

    def _build_reflection_prompt(
        self,
        original_response: str,
        consistency_score: float,
        action_name:       str,
        stress_factor:     float,
    ) -> str:
        """
        Tạo reflection prompt để LLM tự cải thiện response.
        Dựa trên emotional state (action_name, stress) chứ không phải rule cứng.
        """
        stress_desc = "căng thẳng cao" if stress_factor > 0.6 else \
                      "căng thẳng vừa" if stress_factor > 0.3 else "bình tĩnh"
        tone_desc   = {
            "calm":        "bình tĩnh và ổn định",
            "warm":        "ấm áp và quan tâm",
            "concerned":   "lo lắng và hỏi thêm",
            "tool_use":    "chính xác và thực tế",
            "tool_skip":   "giải thích rõ ràng",
            "memory_deep": "dựa trên ký ức",
        }.get(action_name, "tự nhiên")

        return (
            f"<|im_start|>system\n"
            f"Em là Nắng. Em vừa trả lời nhưng cảm thấy chưa đúng tone.\n"
            f"Trạng thái hiện tại: {stress_desc}, cần tone: {tone_desc}.\n"
            f"Hãy viết lại ngắn gọn hơn, đúng cảm xúc hơn.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Câu trả lời cũ của em: {original_response[:300]}\n"
            f"Viết lại cho phù hợp hơn với trạng thái cảm xúc hiện tại.\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _regenerate(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Regenerate response với reflection prompt.
        [#2] Non-blocking lock — nếu LLM đang busy thì skip reflection turn này
        thay vì block và risk deadlock với think() đang giữ gen_lock.
        """
        try:
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)
            # Non-blocking acquire — skip nếu gen_lock đang được giữ bởi think()
            acquired = self._gen_lock.acquire(timeout=0.05)
            if not acquired:
                logger.debug("[Reflection] gen_lock busy — skip regenerate this turn")
                return ""
            try:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens   = min(max_tokens, 100),   # hard cap
                    temperature      = 0.3,
                    do_sample        = True,
                    pad_token_id     = self._tokenizer.eos_token_id,
                    use_cache        = True,
                )
            finally:
                self._gen_lock.release()
            new_ids = out[0][inputs["input_ids"].shape[-1]:]
            return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        except Exception as e:
            logger.warning(f"[Reflection] regenerate failed: {e}")
            return ""

    def reflect(
        self,
        response:      str,
        h_np:          np.ndarray,
        z_np:          np.ndarray,
        action_name:   str   = "calm",
        stress_factor: float = 0.0,
    ) -> ReflectionResult:
        """
        Main reflection pipeline.

        Args:
            response:      Response gốc từ LLM
            h_np:          Deterministic latent state (det_dim,)
            z_np:          Stochastic latent state (stoch_dim,)
            action_name:   Action được chọn bởi RSSM agent
            stress_factor: Stress hiện tại của agent

        Returns:
            ReflectionResult với consistency_score và reflected_response
        """
        if not response or h_np is None or z_np is None:
            return ReflectionResult(
                original_response  = response or "",
                reflected_response = response or "",
                consistency_score  = 0.5,
                latent_valence     = 0.0,
                response_valence   = 0.0,
                did_reflect        = False,
                reflection_delta   = 0.0,
            )

        try:
            # 1. Embed response
            resp_emb     = self._embed_text(response[:500])

            # 2. Score consistency
            score_before = self._score(h_np, z_np, resp_emb)

            # 3. Estimate valence từ latent h và response
            h_flat           = h_np.flatten()
            latent_valence   = float(np.tanh(h_flat[:8].mean())) if len(h_flat) >= 8 else 0.0
            response_valence = float(np.tanh(resp_emb[:8].mean())) if len(resp_emb) >= 8 else 0.0

            logger.debug(
                f"[Reflection] score={score_before:.3f} "
                f"lat_val={latent_valence:.3f} resp_val={response_valence:.3f}"
            )

            # 4. Reflect nếu cần
            reflected_response = response
            did_reflect        = False
            reflection_delta   = 0.0

            adaptive_threshold = self.CONSISTENCY_THRESHOLD + 0.1 * stress_factor
            if self.mode == self.MODE_REGEN and score_before < adaptive_threshold:
                prompt   = self._build_reflection_prompt(
                    response, score_before, action_name, stress_factor
                )
                new_resp = self._regenerate(prompt)

                if new_resp and len(new_resp) > 10:
                    # Score lại response mới
                    new_emb      = self._embed_text(new_resp[:500])
                    score_after  = self._score(h_np, z_np, new_emb)
                    reflection_delta = score_after - score_before

                    # Chỉ dùng response mới nếu thực sự tốt hơn
                    if score_after > score_before:
                        reflected_response = new_resp
                        did_reflect        = True
                        logger.info(
                            f"[Reflection] Improved: {score_before:.3f} → {score_after:.3f} "
                            f"(+{reflection_delta:.3f})"
                        )
                    else:
                        logger.debug(
                            f"[Reflection] Regenerated worse ({score_after:.3f} < {score_before:.3f})"
                            f" — giữ original"
                        )

            return ReflectionResult(
                original_response  = response,
                reflected_response = reflected_response,
                consistency_score  = score_before,
                latent_valence     = latent_valence,
                response_valence   = response_valence,
                did_reflect        = did_reflect,
                reflection_delta   = reflection_delta,
            )

        except Exception as e:
            logger.warning(f"[Reflection] reflect() failed: {e}")
            return ReflectionResult(
                original_response  = response,
                reflected_response = response,
                consistency_score  = 0.5,
                latent_valence     = 0.0,
                response_valence   = 0.0,
                did_reflect        = False,
                reflection_delta   = 0.0,
            )

    def save(self, path: str):
        """Save scorer weights."""
        torch.save(self._scorer.state_dict(), path)
        logger.info(f"[Reflection] Scorer saved: {path}")

    def load(self, path: str):
        """Load scorer weights."""
        import os
        if os.path.exists(path):
            self._scorer.load_state_dict(torch.load(path, map_location=self._device))
            logger.info(f"[Reflection] Scorer loaded: {path}")
