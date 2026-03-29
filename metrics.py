# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/metrics.py
# MỤC ĐÍCH: Evaluation framework cho latent conditioning research
#
# METRICS:
#   1. PersonalityConsistency — embedding similarity responses qua nhiều turns
#   2. StressCoherence — stress response nhất quán với input sentiment
#   3. LatentDynamics — track h, z evolution qua conversation
#   4. BaselineComparison — so sánh latent-conditioned vs text-conditioned
#
# OUTPUT: JSON logs + summary statistics
# ==============================================================================

import json
import time
import datetime
import threading
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

from config import NangConfig, logger


@dataclass
class TurnMetrics:
    """Metrics cho một turn conversation."""
    turn_id:          int
    timestamp:        float
    user_input:       str
    response:         str
    # Latent state
    h_norm:           float          # ||h|| — energy của deterministic state
    z_std:            float          # std(z) — uncertainty của stochastic state
    latent_valence:   float          # estimated valence từ h
    latent_arousal:   float          # estimated arousal từ h
    # Emotion/stress
    stress_factor:    float
    action_name:      str
    soul_frozen:      bool
    avg_entropy:      float
    # Quality
    response_len:     int
    sentiment_score:  float
    # RL
    reward:           float
    dreamer_loss:     float
    # Conditioning mode
    conditioning_mode: str           # "latent_embed" | "latent_text" | "text_only"


@dataclass
class SessionMetrics:
    """Aggregate metrics cho toàn session."""
    session_id:             str
    start_time:             float
    conditioning_mode:      str
    turns:                  list = field(default_factory=list)

    # Computed statistics (filled by compute_stats)
    personality_consistency: Optional[float] = None   # mean cosine sim responses
    stress_coherence:        Optional[float] = None   # correlation stress↔negative input
    avg_reward:              Optional[float] = None
    avg_dreamer_loss:        Optional[float] = None
    panic_rate:              Optional[float] = None    # % turns in panic state
    latent_energy_mean:      Optional[float] = None
    latent_uncertainty_mean: Optional[float] = None


class ResearchMetrics:
    """
    Central metrics collector cho Nắng AI research.

    Usage:
        metrics = ResearchMetrics(mode="latent_embed")
        metrics.log_turn(turn_metrics)
        metrics.save_session()
        stats = metrics.compute_stats()
    """

    LOG_DIR = Path("research/logs")

    def __init__(self, conditioning_mode: str = "text_only"):
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._lock  = threading.Lock()
        self._embed_cache: dict = {}   # cache response embeddings

        session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = SessionMetrics(
            session_id        = session_id,
            start_time        = time.time(),
            conditioning_mode = conditioning_mode,
        )
        self._response_embeddings: list = []   # để tính PersonalityConsistency
        self._stress_inputs:       list = []   # (stress_factor, input_sentiment)

        logger.info(f"[Metrics] Session {session_id} — mode: {conditioning_mode}")

    def log_turn(
        self,
        turn_id:        int,
        user_input:     str,
        response:       str,
        h:              np.ndarray,
        z:              np.ndarray,
        stress_factor:  float,
        action_name:    str,
        soul_frozen:    bool,
        avg_entropy:    float,
        sentiment_score: float,
        reward:         float,
        dreamer_loss:   float,
        conditioning_mode: str = None,
    ):
        """Log metrics cho 1 turn."""
        # Compute latent statistics
        h_norm       = float(np.linalg.norm(h))
        z_std        = float(np.std(z))
        h_flat       = h.flatten()
        latent_val   = float(np.tanh(h_flat[:8].mean())) if len(h_flat) >= 8  else 0.0
        latent_aro   = float(np.tanh(h_flat[8:16].mean())) if len(h_flat) >= 16 else 0.0

        turn = TurnMetrics(
            turn_id           = turn_id,
            timestamp         = time.time(),
            user_input        = user_input[:200],   # truncate for storage
            response          = response[:500],
            h_norm            = round(h_norm, 4),
            z_std             = round(z_std, 4),
            latent_valence    = round(latent_val, 4),
            latent_arousal    = round(latent_aro, 4),
            stress_factor     = round(stress_factor, 4),
            action_name       = action_name,
            soul_frozen       = soul_frozen,
            avg_entropy       = round(avg_entropy, 4),
            response_len      = len(response),
            sentiment_score   = round(sentiment_score, 4),
            reward            = round(reward, 4),
            dreamer_loss      = round(dreamer_loss, 4),
            conditioning_mode = conditioning_mode or self.session.conditioning_mode,
        )

        with self._lock:
            self.session.turns.append(asdict(turn))
            self._stress_inputs.append((stress_factor, sentiment_score))

        logger.debug(
            f"[Metrics] Turn {turn_id}: stress={stress_factor:.3f} "
            f"action={action_name} reward={reward:.3f} loss={dreamer_loss:.4f}"
        )

    def log_response_embedding(self, embedding: np.ndarray):
        """
        Log embedding của response để tính PersonalityConsistency.
        Gọi sau mỗi turn với embedding của full_response.
        """
        with self._lock:
            self._response_embeddings.append(embedding.copy())

    def compute_stats(self) -> dict:
        """
        Tính aggregate statistics cho session.
        Trả về dict có thể dùng cho paper table.
        """
        turns = self.session.turns
        if not turns:
            return {}

        # 1. PersonalityConsistency — mean pairwise cosine similarity
        personality_score = None
        embs = self._response_embeddings
        if len(embs) >= 2:
            sims = []
            for i in range(len(embs)):
                for j in range(i+1, min(i+5, len(embs))):  # chỉ compare với 5 turn gần nhất
                    a, b = embs[i], embs[j]
                    sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
                    sims.append(sim)
            personality_score = float(np.mean(sims)) if sims else None

        # 2. StressCoherence — correlation giữa stress và negative sentiment
        stress_coherence = None
        if len(self._stress_inputs) >= 5:
            stresses   = [x[0] for x in self._stress_inputs]
            sentiments = [x[1] for x in self._stress_inputs]
            # Sentiment cao = positive → stress nên thấp → negative correlation
            if np.std(stresses) > 0 and np.std(sentiments) > 0:
                corr = float(np.corrcoef(stresses, sentiments)[0, 1])
                stress_coherence = corr   # negative = coherent (stress↑ khi sentiment↓)

        # 3. RL metrics
        rewards     = [t["reward"] for t in turns]
        losses      = [t["dreamer_loss"] for t in turns if t["dreamer_loss"] > 0]
        panic_turns = [t for t in turns if t["soul_frozen"]]

        # 4. Latent dynamics
        h_norms = [t["h_norm"] for t in turns]
        z_stds  = [t["z_std"] for t in turns]

        stats = {
            "session_id":              self.session.session_id,
            "conditioning_mode":       self.session.conditioning_mode,
            "n_turns":                 len(turns),
            "duration_min":            round((time.time() - self.session.start_time) / 60, 2),
            # Research metrics
            "personality_consistency": round(personality_score, 4) if personality_score else None,
            "stress_coherence":        round(stress_coherence, 4) if stress_coherence else None,
            # RL metrics
            "avg_reward":              round(float(np.mean(rewards)), 4) if rewards else None,
            "avg_dreamer_loss":        round(float(np.mean(losses)), 4) if losses else None,
            "panic_rate":              round(len(panic_turns) / len(turns), 4) if turns else None,
            # Latent dynamics
            "latent_energy_mean":      round(float(np.mean(h_norms)), 4) if h_norms else None,
            "latent_energy_std":       round(float(np.std(h_norms)), 4) if h_norms else None,
            "latent_uncertainty_mean": round(float(np.mean(z_stds)), 4) if z_stds else None,
        }

        # Update session
        self.session.personality_consistency = stats["personality_consistency"]
        self.session.stress_coherence        = stats["stress_coherence"]
        self.session.avg_reward              = stats["avg_reward"]
        self.session.avg_dreamer_loss        = stats["avg_dreamer_loss"]
        self.session.panic_rate              = stats["panic_rate"]
        self.session.latent_energy_mean      = stats["latent_energy_mean"]
        self.session.latent_uncertainty_mean = stats["latent_uncertainty_mean"]

        return stats

    def save_session(self):
        """Save session metrics ra JSON file."""
        stats = self.compute_stats()
        out = {
            "stats":  stats,
            "turns":  self.session.turns,
        }
        path = self.LOG_DIR / f"session_{self.session.session_id}.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            logger.info(f"[Metrics] Session saved: {path}")
        except Exception as e:
            logger.warning(f"[Metrics] Save failed: {e}")
        return stats

    def print_summary(self):
        """Print readable summary ra console."""
        stats = self.compute_stats()
        print("\n" + "="*60)
        print(f"SESSION SUMMARY — {stats.get('conditioning_mode', 'unknown')}")
        print("="*60)
        print(f"  Turns:                  {stats.get('n_turns', 0)}")
        print(f"  Duration:               {stats.get('duration_min', 0):.1f} min")
        print(f"  Personality Consistency:{stats.get('personality_consistency', 'N/A')}")
        print(f"  Stress Coherence:       {stats.get('stress_coherence', 'N/A')}")
        print(f"  Avg Reward:             {stats.get('avg_reward', 'N/A')}")
        print(f"  Avg Dreamer Loss:       {stats.get('avg_dreamer_loss', 'N/A')}")
        print(f"  Panic Rate:             {stats.get('panic_rate', 'N/A')}")
        print(f"  Latent Energy (mean):   {stats.get('latent_energy_mean', 'N/A')}")
        print("="*60 + "\n")


def compare_sessions(session_files: list) -> dict:
    """
    So sánh nhiều session (baseline vs latent conditioning).
    Input: list of JSON file paths
    Output: comparison table dict
    """
    results = []
    for f in session_files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            results.append(data.get("stats", {}))
        except Exception as e:
            logger.warning(f"[Metrics] Cannot load {f}: {e}")

    if not results:
        return {}

    # Group by conditioning_mode
    groups = {}
    for r in results:
        mode = r.get("conditioning_mode", "unknown")
        groups.setdefault(mode, []).append(r)

    comparison = {}
    for mode, sessions in groups.items():
        metrics_to_compare = [
            "personality_consistency",
            "stress_coherence",
            "avg_reward",
            "panic_rate",
            "latent_energy_mean",
        ]
        comparison[mode] = {}
        for metric in metrics_to_compare:
            vals = [s[metric] for s in sessions if s.get(metric) is not None]
            if vals:
                comparison[mode][metric] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std":  round(float(np.std(vals)), 4),
                    "n":    len(vals),
                }

    return comparison
