# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/latent_adapter.py
# MỤC ĐÍCH: Latent-Conditioned Generation
#   Inject RSSM latent state (h, z) vào LLM như soft prefix tokens
#   thay vì text trong system prompt.
#
# CONTRIBUTION:
#   Đây là điểm kết nối thật sự giữa RSSM emotional state và LLM generation.
#   Khác với approach hiện tại (inject text "tone=calm" vào prompt),
#   latent conditioning cho phép LLM attend trực tiếp vào latent dynamics
#   của emotional state — không qua bottleneck ngôn ngữ.
#
# ARCHITECTURE:
#   h (CONV_DET_DIM=64) + z (CONV_STOCH_DIM=16) → concat (80,)
#   → LatentAdapter (MLP) → (N_PREFIX_TOKENS, LLM_HIDDEN_DIM)
#   → prepend vào input embeddings → LLM generate
#
# COMPATIBILITY:
#   - unsloth FastLanguageModel (4bit quantized)
#   - Standard HuggingFace CausalLM
#   - Fallback: nếu không access được embeddings, dùng text prefix
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np
from config import NangConfig, logger


class LatentAdapter(nn.Module):
    """
    Project RSSM latent (h, z) thành soft prefix tokens cho LLM.

    Input:  h (det_dim,) + z (stoch_dim,) → concat (latent_dim,)
    Output: (n_prefix_tokens, llm_hidden_dim) — prepend vào input embeddings

    Training: adapter được train riêng qua reconstruction loss
    (predict next token distribution conditioned on latent state).
    LLM weights KHÔNG được update — adapter là lightweight ~1M params.
    """

    def __init__(
        self,
        latent_dim:     int = NangConfig.CONV_DET_DIM + NangConfig.CONV_STOCH_DIM,
        llm_hidden_dim: int = NangConfig.LATENT_LLM_DIM,
        n_prefix_tokens: int = NangConfig.N_PREFIX_TOKENS,
        hidden_dim:     int = 256,
    ):
        super().__init__()
        self.n_prefix_tokens = n_prefix_tokens
        self.llm_hidden_dim  = llm_hidden_dim
        self.latent_dim      = latent_dim

        # MLP: latent → prefix token embeddings
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_prefix_tokens * llm_hidden_dim),
        )

        # Scale init nhỏ — tránh dominate input embeddings ban đầu
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: deterministic state  (batch, det_dim) hoặc (det_dim,)
            z: stochastic state     (batch, stoch_dim) hoặc (stoch_dim,)
        Returns:
            prefix_embeds: (batch, n_prefix_tokens, llm_hidden_dim)
        """
        # Handle unbatched input
        if h.dim() == 1:
            h = h.unsqueeze(0)
            z = z.unsqueeze(0)

        latent = torch.cat([h, z], dim=-1)   # (batch, latent_dim)
        out    = self.net(latent)             # (batch, n_prefix * hidden)
        return out.view(-1, self.n_prefix_tokens, self.llm_hidden_dim)


class LatentConditionedGenerator:
    """
    Wrapper inject latent prefix vào LLM generation.

    Hai mode:
      MODE_EMBED  — inject vào input_embeddings trực tiếp (requires model internals access)
      MODE_FALLBACK — dùng text representation nếu không access được embeddings

    Tự detect mode khi init.
    """

    MODE_EMBED    = "embed"
    MODE_FALLBACK = "fallback"

    def __init__(self, model, tokenizer, adapter: LatentAdapter, device):
        self.model     = model
        self.tokenizer = tokenizer
        self.adapter   = adapter.to(device)
        self.device    = device
        self.mode      = self._detect_mode()
        logger.info(f"[LatentGen] Mode: {self.mode}")

    def _detect_mode(self) -> str:
        """
        Kiểm tra xem có access được model.get_input_embeddings() không.
        unsloth wrap model nhưng vẫn expose phần lớn HF API.
        """
        try:
            emb = self.model.get_input_embeddings()
            if emb is not None and hasattr(emb, 'weight'):
                # Test forward pass nhỏ để xác nhận
                test = torch.zeros(1, 1, dtype=torch.long).to(self.device)
                _ = emb(test)
                return self.MODE_EMBED
        except Exception as e:
            logger.warning(f"[LatentGen] Embed access failed: {e} — fallback mode")
        return self.MODE_FALLBACK

    def _get_prefix_embeds(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Tính prefix embeddings từ latent state."""
        with torch.no_grad():
            return self.adapter(h, z)   # (1, n_prefix, hidden_dim)

    def _latent_to_text(self, h: torch.Tensor, z: torch.Tensor) -> str:
        """
        Fallback: convert latent thành text description.
        Dùng khi không access được model internals.
        """
        h_np = h.squeeze().cpu().numpy()
        z_np = z.squeeze().cpu().numpy()

        # Các chiều có semantic trong h (deterministic — ổn định hơn)
        energy    = float(np.linalg.norm(h_np))
        valence   = float(np.tanh(h_np[:8].mean()))   # 8 chiều đầu → valence
        arousal   = float(np.tanh(h_np[8:16].mean())) # 8 chiều tiếp → arousal
        # z (stochastic) → uncertainty
        uncertainty = float(np.std(z_np))

        # Map sang text description ngắn gọn
        v_word = "positive" if valence > 0.2 else "negative" if valence < -0.2 else "neutral"
        a_word = "high" if arousal > 0.2 else "low" if arousal < -0.2 else "moderate"

        return (
            f"[LATENT STATE] valence={v_word} arousal={a_word} "
            f"energy={energy:.2f} uncertainty={uncertainty:.2f}"
        )

    def prepare_inputs(
        self,
        input_ids: torch.Tensor,
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> dict:
        """
        Chuẩn bị inputs cho LLM generation với latent conditioning.

        Returns dict với keys phù hợp cho model.generate():
          MODE_EMBED:    inputs_embeds + attention_mask
          MODE_FALLBACK: input_ids + attention_mask (với latent text prefix)
        """
        batch_size = input_ids.shape[0]

        if self.mode == self.MODE_EMBED:
            try:
                embed_layer  = self.model.get_input_embeddings()
                text_embeds  = embed_layer(input_ids)          # (1, seq, hidden)
                prefix_embeds = self._get_prefix_embeds(h, z)  # (1, n_prefix, hidden)

                # Concat: [prefix | text]
                combined_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)

                # Attention mask: 1 cho tất cả (prefix + text)
                prefix_mask = torch.ones(batch_size, self.adapter.n_prefix_tokens).to(self.device)
                text_mask   = torch.ones_like(input_ids)
                combined_mask = torch.cat([prefix_mask, text_mask], dim=1)

                return {
                    "inputs_embeds":  combined_embeds,
                    "attention_mask": combined_mask,
                }
            except Exception as e:
                logger.warning(f"[LatentGen] Embed inject failed: {e} — fallback")
                self.mode = self.MODE_FALLBACK

        # Fallback: prepend latent text vào input_ids
        latent_text   = self._latent_to_text(h, z)
        latent_ids    = self.tokenizer(latent_text, return_tensors="pt",
                                        add_special_tokens=False).input_ids.to(self.device)
        combined_ids  = torch.cat([latent_ids, input_ids], dim=1)
        combined_mask = torch.ones_like(combined_ids)

        return {
            "input_ids":      combined_ids,
            "attention_mask": combined_mask,
        }

    def save(self, path: str):
        """Save adapter weights."""
        torch.save(self.adapter.state_dict(), path)
        logger.info(f"[LatentGen] Adapter saved: {path}")

    def load(self, path: str):
        """Load adapter weights."""
        import os
        if os.path.exists(path):
            self.adapter.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"[LatentGen] Adapter loaded: {path}")
        else:
            logger.warning(f"[LatentGen] Adapter file not found: {path}")
