# ==============================================================================
# PROJECT: NẮNG AI - v47.6
# FILE: env.py
# MỤC ĐÍCH: ConversationEnv — môi trường Dreamer dựa trên conversation context
#   Thay thế SurvivalEnv grid 5x5 bằng observation space thực tế:
#     - State  = LLM embedding của conversation (OBS_DIM=384 → project OBS_PROJ_DIM=64)
#     - Action = {calm, warm, concerned, tool_use, tool_skip, memory_deep}
#     - Reward = composite(engagement, task_success, sentiment_shift)
#
#   Giữ nguyên interface với soul.py:
#     reset()  → np.ndarray state
#     step(action, external_stress) → (state, reward, entropy_norm, done)
#     _get_state(obs_vec, external_stress) → np.ndarray
#
#   SurvivalEnv giữ lại cuối file — không xoá, để tham chiếu và fallback.
# ==============================================================================

import numpy as np
from config import NangConfig


# Hằng số normalize
_MAX_ENTROPY = 3.0
_MAX_STEPS   = 50     # ConversationEnv không có "goal" cố định → dùng max steps


class ConversationEnv:
    """
    Môi trường Dreamer dựa trên conversation context thực tế.

    Khác với SurvivalEnv:
      - Không có grid, không có goal_pos cố định
      - State = embedding vector của user input (được project xuống OBS_PROJ_DIM)
      - "Goal" = đạt sentiment tích cực (score > REWARD_GOAL_TH)
      - "Danger" = stress cao / sentiment tiêu cực
      - Agent học policy: nên dùng tone nào, có nên gọi tool, có cần retrieve memory

    Reward composite:
      engagement_reward  — user gửi tin nhắn tiếp (signal qua update_reward())
      task_reward        — tool call thành công
      sentiment_reward   — sentiment shift tích cực
      step_penalty       — tránh agent idle
      goal_bonus         — khi sentiment rất tốt

    Interface giống SurvivalEnv để soul.py không cần thay đổi.
    """

    def __init__(self):
        self._obs_dim = NangConfig.OBS_PROJ_DIM
        # [#1] Direction vectors cố định cho mỗi action — mỗi action có "direction" riêng
        # trong embedding space thay vì shift đồng đều phá cấu trúc semantic
        # seed cố định để reproducible across sessions
        rng = np.random.RandomState(42)
        self._action_dirs = rng.randn(
            NangConfig.CONV_ACTION_DIM, self._obs_dim
        ).astype(np.float32) * 0.01
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset về trạng thái neutral đầu conversation."""
        self.steps              = 0
        self._sentiment_score   = 0.5
        self._prev_sentiment    = 0.5
        self._engagement_signal = False
        self._task_success      = False
        self._obs_vec           = np.zeros(self._obs_dim, dtype=np.float32)
        self._last_action       = 0
        self._memory_usefulness = 0.5
        self._pending_reward    = 0.0   # [#9] deferred reward — injected vào remember() turn sau
        return self._get_state(self._obs_vec, 0.0)

    def _get_state(self, obs_vec: np.ndarray, external_stress: float = 0.0) -> np.ndarray:
        """
        State = [obs_proj (OBS_PROJ_DIM), sentiment (1), stress (1), steps_norm (1), action_scalar (1)]
        Total dim = OBS_PROJ_DIM + 4

        [#3] Dùng scalar action thay one-hot — tránh shortcut learning.
        One-hot → RSSM có thể ignore dynamics, chỉ học action→reward mapping.
        Scalar normalize → vẫn có action signal nhưng không dominate.
        """
        sentiment_arr   = np.array([self._sentiment_score], dtype=np.float32)
        stress_arr      = np.array([min(external_stress / NangConfig.STRESS_CAP, 1.0)], dtype=np.float32)
        steps_arr       = np.array([self.steps / _MAX_STEPS], dtype=np.float32)
        action_scalar   = np.array([self._last_action / NangConfig.CONV_ACTION_DIM], dtype=np.float32)
        return np.concatenate([obs_vec, sentiment_arr, stress_arr, steps_arr, action_scalar])

    def update_obs(self, obs_vec: np.ndarray):
        """
        Cập nhật observation vector từ LLM embedding.
        [#8] Validate NaN/Inf trước khi update — embedding lỗi không corrupt state.
        """
        if (
            obs_vec is not None
            and len(obs_vec) == self._obs_dim
            and np.all(np.isfinite(obs_vec))
        ):
            self._obs_vec = np.clip(obs_vec.astype(np.float32), -1.0, 1.0)

    def update_reward_signal(
        self,
        sentiment_score:    float,
        engagement:         bool,
        task_success:       bool,
        memory_usefulness:  float = 0.5,
    ):
        """
        [#9] Deferred reward — update signal SAU khi stream xong.
        Reward thật được lưu vào _pending_reward để inject vào
        remember() ở turn TIẾP THEO thay vì step() của turn hiện tại.
        """
        self._prev_sentiment      = self._sentiment_score
        self._sentiment_score     = float(np.clip(sentiment_score, 0.0, 1.0))
        self._engagement_signal   = engagement
        self._task_success        = task_success
        self._memory_usefulness   = float(np.clip(memory_usefulness, 0.0, 1.0))
        # Tính reward thật và lưu pending để inject turn sau
        sentiment_shift = self._sentiment_score - self._prev_sentiment
        r = NangConfig.REWARD_STEP_PENALTY
        if abs(sentiment_shift) > 0.05:
            r += NangConfig.REWARD_SENTIMENT_W * sentiment_shift
        if self._sentiment_score >= NangConfig.REWARD_GOAL_TH:
            r += NangConfig.REWARD_GOAL_BONUS * 0.3
        if task_success:
            r += NangConfig.REWARD_TASK_W
        if memory_usefulness > 0.7:
            r += 0.1 * (memory_usefulness - 0.7) / 0.3
        self._pending_reward = float(np.clip(r, -1.0, 1.0))

    def pop_pending_reward(self) -> float:
        """Lấy reward thật từ turn trước — gọi ở đầu step() turn này."""
        r = self._pending_reward
        self._pending_reward = 0.0
        return r

    def _normalize_entropy(self, entropy: float) -> float:
        """Normalize entropy về [0,1] — giữ interface giống SurvivalEnv."""
        return float(np.clip(entropy / _MAX_ENTROPY, 0.0, 1.0))

    def _apply_action_effect(self, action: int):
        """
        [#1] Mỗi action có direction vector riêng trong embedding space.
        Shift đồng đều (+=0.01) phá cấu trúc semantic của embedding.
        Direction vector cố định (seed=42) → RSSM học được pattern ổn định.
        [#3] Decay nhẹ trước khi apply — tránh obs_vec drift xa embedding thật.
        """
        self._obs_vec *= 0.995   # [#3] decay về zero rất chậm
        self._obs_vec += self._action_dirs[action]
        self._obs_vec = np.clip(self._obs_vec, -1.0, 1.0)

    def step(self, action: int, external_stress: float = 0.0):
        action = int(np.clip(action, 0, NangConfig.CONV_ACTION_DIM - 1))
        self.steps += 1
        self._last_action = action

        # [#1] Action ảnh hưởng dynamics trước khi tính state mới
        self._apply_action_effect(action)

        # --- Reward ---
        reward = NangConfig.REWARD_STEP_PENALTY

        # [#2] Engagement bị tắt hoàn toàn — signal chưa thật, constant bias làm lệch policy
        # TODO: implement real engagement signal trước khi bật lại

        # Task reward
        if action == 3 and self._task_success:
            reward += NangConfig.REWARD_TASK_W
        elif action == 3 and not self._task_success:
            reward -= NangConfig.REWARD_TASK_W * 0.5

        # [#4] memory_deep — penalty nếu gọi vô nghĩa để tránh spam
        if action == 5:
            if self._task_success:
                reward += NangConfig.REWARD_TASK_W * 0.5
            else:
                reward -= 0.05

        # [#2] Sentiment shift với threshold
        sentiment_shift = self._sentiment_score - self._prev_sentiment
        if abs(sentiment_shift) > 0.05:
            reward += NangConfig.REWARD_SENTIMENT_W * sentiment_shift

        # Goal bonus
        if self._sentiment_score >= NangConfig.REWARD_GOAL_TH:
            reward += NangConfig.REWARD_GOAL_BONUS * 0.3

        # [M10] Memory usefulness bonus — feedback loop thật
        # memory_deep retrieve memory → usefulness cao → agent được reward
        # Giúp agent học: khi nào nên dùng memory để improve response
        if action == 5 and self._memory_usefulness > 0.7:
            reward += 0.1 * (self._memory_usefulness - 0.7) / 0.3  # scale 0→0.1

        # Action-context shaping
        # Agent học: dùng đúng tone đúng lúc, không phải chỉ optimize sentiment
        if action == 0 and self._sentiment_score < 0.4:
            reward += 0.05   # calm đúng lúc khi user đang stress
        if action == 1 and self._sentiment_score > 0.5:
            reward += 0.05   # warm đúng lúc khi mood tốt
        if action == 2 and self._sentiment_score > 0.7:
            reward -= 0.05   # concerned sai lúc khi mọi thứ ổn
        if action == 2 and self._sentiment_score < 0.3:
            reward += 0.05   # concerned đúng lúc khi user khó chịu

        reward = float(np.clip(reward, -1.0, 1.0))

        entropy_raw  = min(external_stress + (1.0 - self._sentiment_score) * 2.0, _MAX_ENTROPY)
        entropy_norm = self._normalize_entropy(entropy_raw)

        # [#5] min steps > 5 tránh early termination do sentiment model noisy
        done = (
            self.steps >= _MAX_STEPS
            or (self._sentiment_score > 0.9 and self.steps > 5)
            or (self._sentiment_score < 0.1 and self.steps > 5)
        )

        next_state = self._get_state(self._obs_vec, external_stress)
        return next_state, reward, entropy_norm, done

    @property
    def state_dim(self) -> int:
        """OBS_PROJ_DIM + 4 (sentiment/stress/steps/action_scalar)."""
        return self._obs_dim + 4

    @property
    def action_dim(self) -> int:
        return NangConfig.CONV_ACTION_DIM


# ==============================================================================
# SurvivalEnv — GIỮ NGUYÊN, không xoá
# Dùng làm reference và fallback nếu cần.
# Toàn bộ code giữ nguyên từ v47.5.
# ==============================================================================

# Hằng số normalize cho SurvivalEnv
_SURV_MAX_DIST    = np.linalg.norm(np.array([4, 4]))   # diagonal 5x5 grid ≈ 5.66
_SURV_MAX_ENTROPY = 3.0


class SurvivalEnv:
    """
    Grid world 5x5 làm metaphor cảm xúc cho Dreamer agent.
    GIỮ NGUYÊN từ v47.5 — không xoá.

    Ý định thiết kế (không phải production RL):
      - Agent = "tâm lý Nắng"
      - Goal  = trạng thái bình tĩnh (4,4)
      - Danger= nguồn stress (2,2)
      - Stress từ user input → external_stress → làm entropy tăng

    Reward shaping ở đây là intentional và minimal:
      step_penalty   → khuyến khích agent không đứng yên
      approach_bonus → shaped reward giúp học nhanh hơn trong env nhỏ
      goal_bonus     → terminal reward
    Không có claim về generalization vì đây không phải benchmark env.
    """

    def __init__(self, size: int = 5):
        self.size       = size
        self.danger_pos = np.array([2, 2])
        self.goal_pos   = np.array([4, 4])
        self.reset()

    def reset(self):
        self.agent_pos = np.array([0, 0])
        self.steps     = 0
        return self._get_state(self.agent_pos, 0.0)

    def _get_state(self, pos: np.ndarray, external_stress: float = 0.0) -> np.ndarray:
        dist_g = np.linalg.norm(pos - self.goal_pos)
        dist_d = np.linalg.norm(pos - self.danger_pos)
        # Entropy thô — inverse distance tới danger, capped 3.0
        base_entropy  = min(1.0 / (dist_d + 0.1), _SURV_MAX_ENTROPY)
        final_entropy = min(base_entropy + external_stress, _SURV_MAX_ENTROPY)
        return np.array([pos[0], pos[1], dist_g, final_entropy], dtype=np.float32)

    def _normalize_entropy(self, entropy: float) -> float:
        """
        [E1] Normalize entropy về [0,1] để scale match với reward.
        entropy ∈ [0, _SURV_MAX_ENTROPY=3.0] → [0, 1]
        """
        return float(np.clip(entropy / _SURV_MAX_ENTROPY, 0.0, 1.0))

    def step(self, action: int, external_stress: float = 0.0):
        sim_pos = np.copy(self.agent_pos)
        if   action == 0: sim_pos[0] = max(0, sim_pos[0] - 1)
        elif action == 1: sim_pos[0] = min(self.size - 1, sim_pos[0] + 1)
        elif action == 2: sim_pos[1] = max(0, sim_pos[1] - 1)
        elif action == 3: sim_pos[1] = min(self.size - 1, sim_pos[1] + 1)

        dist_g_old = np.linalg.norm(self.agent_pos - self.goal_pos)
        dist_g_new = np.linalg.norm(sim_pos - self.goal_pos)
        dist_d     = np.linalg.norm(sim_pos - self.danger_pos)

        # [E2] Reward normalize về [-1, 1]
        reward = -0.01
        if dist_g_new == 0:
            reward += 1.0                           # goal bonus
        elif dist_g_new < dist_g_old:
            # [2.4] Clip 0.3 thay vì 0.1
            improvement = (dist_g_old - dist_g_new) / _SURV_MAX_DIST
            reward += float(np.clip(improvement, 0.0, 0.3))

        base_entropy  = min(1.0 / (dist_d + 0.1), _SURV_MAX_ENTROPY)
        entropy_raw   = min(base_entropy + external_stress, _SURV_MAX_ENTROPY)
        # [E1] Entropy trả về đã normalize → soul.py dùng trực tiếp không bị scale lệch
        entropy_norm  = self._normalize_entropy(entropy_raw)

        self.agent_pos = sim_pos
        self.steps    += 1
        done = np.array_equal(self.agent_pos, self.goal_pos) or self.steps > 20

        return self._get_state(self.agent_pos, external_stress), reward, entropy_norm, done
