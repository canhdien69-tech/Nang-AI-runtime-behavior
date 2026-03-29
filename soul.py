# ==============================================================================
# PROJECT: NẮNG AI - v47.3
# FILE: soul.py
# MỤC ĐÍCH: Genome, toán học Dreamer, RSSM, DeepMindRSSMAgent, PersistentBrain
# ==============================================================================

import os, json, threading, copy, random, contextlib
from collections import deque
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as D

from config import NangConfig, CONF, logger


# ==============================================================================
# MODULE 4: THE SOUL & RSSM DEEPMIND ALGORITHMS
# ==============================================================================
@dataclass
class Genome:
    id: str = field(default_factory=lambda: __import__('uuid').uuid4().hex[:8])
    e_weights: list = field(default_factory=lambda: [0.25, 0.25, 0.1, 0.4])
    score: float = 0.0
    generation: int = 0
    trauma_history: list = field(default_factory=list)


# --- TOÁN HỌC DREAMER ---
def reparameterize(mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std


def straight_through_sample(logits, action_dim=NangConfig.CONV_ACTION_DIM):
    probs = F.softmax(logits, dim=-1)
    dist = D.Categorical(probs)
    action = dist.sample()
    onehot = F.one_hot(action, action_dim).float()
    onehot = onehot + probs - probs.detach()
    return onehot, action


# T+1 Bootstrap cho Lambda Return
def lambda_return(rewards, values, discount=NangConfig.DISCOUNT, lam=NangConfig.LAMBDA, not_done=None):
    """
    Tính lambda-return cho imagination horizon.
    not_done: tensor (H, B) — mask terminal states, tránh bootstrap qua done.
    """
    next_values = torch.cat([values[1:], values[-1:]], dim=0)
    returns = []
    acc = next_values[-1]
    for t in reversed(range(len(rewards))):
        # [#11] Mask bootstrap qua terminal — không overestimate returns
        mask = not_done[t] if not_done is not None else 1.0
        acc = rewards[t] + discount * mask * ((1 - lam) * next_values[t] + lam * acc)
        returns.insert(0, acc)
    return torch.stack(returns)


# EMA Tau để ổn định Critic
def update_ema(target, source, tau=NangConfig.EMA_TAU):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


class RSSM(nn.Module):
    def __init__(self, state_dim=4, action_dim=4, det_dim=32, stoch_dim=8):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_dim = stoch_dim

        self.cell = nn.GRUCell(stoch_dim + action_dim, det_dim)

        self.encoder = nn.Sequential(nn.LayerNorm(state_dim), nn.Linear(state_dim, 16), nn.ELU())
        self.post_mu = nn.Linear(16 + det_dim, stoch_dim)
        self.post_std = nn.Sequential(nn.Linear(16 + det_dim, stoch_dim), nn.Softplus())

        self.prior_net = nn.Sequential(nn.Linear(det_dim, 16), nn.ELU())
        self.prior_mu = nn.Linear(16, stoch_dim)
        self.prior_std = nn.Sequential(nn.Linear(16, stoch_dim), nn.Softplus())

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.det_dim).to(device)
        z = torch.zeros(batch_size, self.stoch_dim).to(device)
        return h, z

    def forward_prior(self, h_prev, z_prev, action):
        h = self.cell(torch.cat([z_prev, action], dim=-1), h_prev)
        feat = self.prior_net(h)
        mu, std = self.prior_mu(feat), self.prior_std(feat) + 0.1
        z = reparameterize(mu, std)
        return h, z, mu, std

    def forward_posterior(self, h_prev, z_prev, action, obs):
        h = self.cell(torch.cat([z_prev, action], dim=-1), h_prev)
        obs_embed = self.encoder(obs)
        feat = torch.cat([obs_embed, h], dim=-1)
        mu, std = self.post_mu(feat), self.post_std(feat) + 0.1
        z = reparameterize(mu, std)
        return h, z, mu, std


# ==============================================================================
# ConversationRSSM — RSSM cho observation space mới (embedding-based)
# Giữ toán học Dreamer giống RSSM cũ, chỉ thay:
#   - state_dim: OBS_PROJ_DIM + 3 (embedding + sentiment + stress + steps)
#   - action_dim: CONV_ACTION_DIM = 6
#   - det_dim, stoch_dim: lớn hơn vì obs phức tạp hơn
#   - encoder: deeper (2 layer) + LayerNorm để handle embedding input
# ==============================================================================
class ConversationRSSM(nn.Module):
    """
    RSSM cho ConversationEnv.
    Toán học giống RSSM gốc (Dreamer v2):
      posterior: p(z_t | h_t, o_t)
      prior:     p(z_t | h_t)
      GRU transition: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})

    Khác biệt so với RSSM gốc:
      - state_dim lớn hơn (OBS_PROJ_DIM + 3 thay vì 4)
      - encoder deeper để handle embedding vector
      - det_dim=CONV_DET_DIM, stoch_dim=CONV_STOCH_DIM
    """

    def __init__(
        self,
        state_dim  = NangConfig.CONV_STATE_DIM,   # 68: OBS_PROJ_DIM(64) + 4 scalars
        action_dim = NangConfig.CONV_ACTION_DIM,
        det_dim    = NangConfig.CONV_DET_DIM,
        stoch_dim  = NangConfig.CONV_STOCH_DIM,
        hidden_dim = NangConfig.CONV_HIDDEN_DIM,
    ):
        super().__init__()
        self.det_dim   = det_dim
        self.stoch_dim = stoch_dim

        # GRU transition: nhận (z_{t-1}, a_{t-1}), output h_t
        self.cell = nn.GRUCell(stoch_dim + action_dim, det_dim)

        # Posterior encoder: obs → embed → concat với h → posterior dist
        self.encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
        )
        post_in = hidden_dim // 2 + det_dim
        self.post_mu  = nn.Linear(post_in, stoch_dim)
        self.post_std = nn.Sequential(nn.Linear(post_in, stoch_dim), nn.Softplus())

        # Prior: h → prior dist (không dùng obs)
        self.prior_net = nn.Sequential(
            nn.Linear(det_dim, hidden_dim // 2),
            nn.ELU(),
        )
        self.prior_mu  = nn.Linear(hidden_dim // 2, stoch_dim)
        self.prior_std = nn.Sequential(nn.Linear(hidden_dim // 2, stoch_dim), nn.Softplus())

    def initial_state(self, batch_size: int, device):
        h = torch.zeros(batch_size, self.det_dim).to(device)
        z = torch.zeros(batch_size, self.stoch_dim).to(device)
        return h, z

    def step_h(self, h_prev, z_prev, action):
        """
        [FIX GRU DOUBLE UPDATE] GRU transition tách riêng.
        Dreamer: prior và posterior phải dùng CÙNG h_t.
        Nếu mỗi hàm tự gọi GRU → 2 h khác nhau → KL loss sai hoàn toàn.
        """
        return self.cell(torch.cat([z_prev, action], dim=-1), h_prev)

    def forward_prior(self, h_prev, z_prev, action):
        """Prior: p(z_t | h_t) — không dùng obs. Gọi step_h() trước nếu cần h mới."""
        h    = self.cell(torch.cat([z_prev, action], dim=-1), h_prev)
        feat = self.prior_net(h)
        mu   = self.prior_mu(feat)
        std  = self.prior_std(feat) + 0.1
        z    = reparameterize(mu, std)
        return h, z, mu, std

    def forward_prior_from_h(self, h):
        """Prior từ h đã tính sẵn — dùng trong replay() để tránh double GRU."""
        feat = self.prior_net(h)
        mu   = self.prior_mu(feat)
        std  = self.prior_std(feat) + 0.1
        z    = reparameterize(mu, std)
        return mu, std

    def forward_posterior(self, h_prev, z_prev, action, obs):
        """Posterior: p(z_t | h_t, o_t) — dùng obs."""
        h         = self.cell(torch.cat([z_prev, action], dim=-1), h_prev)
        obs_embed = self.encoder(obs)
        feat      = torch.cat([obs_embed, h], dim=-1)
        mu        = self.post_mu(feat)
        std       = self.post_std(feat) + 0.1
        z         = reparameterize(mu, std)
        return h, z, mu, std

    def forward_posterior_from_h(self, h, obs):
        """Posterior từ h đã tính sẵn — dùng trong replay() để tránh double GRU."""
        obs_embed = self.encoder(obs)
        feat      = torch.cat([obs_embed, h], dim=-1)
        mu        = self.post_mu(feat)
        std       = self.post_std(feat) + 0.1
        z         = reparameterize(mu, std)
        return z, mu, std


class StateDecoder(nn.Module):
    """Decoder cho SurvivalEnv — giữ nguyên."""
    def __init__(self, in_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ELU(), nn.Linear(16, 4))

    def forward(self, hz):
        shape = hz.shape
        return self.net(hz.view(-1, shape[-1])).view(*shape[:-1], 4)


class ConversationDecoder(nn.Module):
    """
    Decoder cho ConversationRSSM.
    Reconstruct state vector từ latent (h, z).
    out_dim = CONV_STATE_DIM = OBS_PROJ_DIM + 4 = 68
    (64 embedding + sentiment + stress + steps_norm + action_scalar)
    Phải khớp với _get_state() trong env.py.
    """
    def __init__(
        self,
        in_dim  = NangConfig.CONV_DET_DIM + NangConfig.CONV_STOCH_DIM,
        out_dim = NangConfig.CONV_STATE_DIM,   # 68, không phải OBS_PROJ_DIM+3=67
        hidden  = NangConfig.CONV_HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, hz):
        shape = hz.shape
        return self.net(hz.view(-1, shape[-1])).view(*shape[:-1], self.out_dim)


class RewardModel(nn.Module):
    """Reward model — nhận in_dim linh hoạt, default cho SurvivalEnv."""
    def __init__(self, in_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ELU(), nn.Linear(16, 1))

    def forward(self, hz):
        shape = hz.shape
        return self.net(hz.view(-1, shape[-1])).view(*shape[:-1], 1)


class ValueModel(nn.Module):
    """Value model — nhận in_dim linh hoạt."""
    def __init__(self, in_dim=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ELU(), nn.Linear(16, 1))

    def forward(self, hz):
        shape = hz.shape
        return self.net(hz.view(-1, shape[-1])).view(*shape[:-1], 1)


class ActorModel(nn.Module):
    """Actor model — nhận in_dim và action_dim linh hoạt."""
    def __init__(self, in_dim=40, action_dim=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 16), nn.ELU(), nn.Linear(16, action_dim))

    def forward(self, hz):
        shape = hz.shape
        return self.net(hz.view(-1, shape[-1])).view(*shape[:-1], -1)


class DeepMindRSSMAgent(nn.Module):
    def __init__(self, batch_size=NangConfig.BATCH_SIZE):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dimensions cho ConversationEnv
        _hz_dim     = NangConfig.CONV_DET_DIM + NangConfig.CONV_STOCH_DIM
        _action_dim = NangConfig.CONV_ACTION_DIM
        _state_dim  = NangConfig.CONV_STATE_DIM   # 68: OBS_PROJ_DIM(64) + 4 scalars

        # [CONV] Dùng ConversationRSSM thay RSSM gốc
        # Toán học Dreamer giữ nguyên, chỉ thay observation encoder + dims
        self.rssm         = ConversationRSSM().to(self.device)
        self.decoder      = ConversationDecoder().to(self.device)
        self.reward_model = RewardModel(in_dim=_hz_dim).to(self.device)

        self.value_model   = ValueModel(in_dim=_hz_dim).to(self.device)
        self.target_critic = copy.deepcopy(self.value_model)
        for p in self.target_critic.parameters(): p.requires_grad = False

        self.actor = ActorModel(in_dim=_hz_dim, action_dim=_action_dim).to(self.device)

        wm_params = (list(self.rssm.parameters()) +
                     list(self.decoder.parameters()) +
                     list(self.reward_model.parameters()))
        self.opt_model = optim.Adam(wm_params, lr=1e-3)
        self.opt_value = optim.Adam(self.value_model.parameters(), lr=1e-3)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=1e-3)

        # [#4] AMP GradScaler cho training ổn định trên RTX 3050 + 4bit
        # Chú ý: unsloth 4bit dùng bfloat16/float16 riêng → AMP chỉ áp dụng cho
        # các module nhỏ (RSSM, Actor, Critic) KHÔNG phải LLM chính
        # Wrap trong try/except: nếu conflict với quantized setup → fallback fp32
        self._amp_enabled = False
        self._scaler      = None
        if self.device.type == "cuda":
            try:
                self._scaler     = torch.cuda.amp.GradScaler()
                self._amp_enabled = True
                logger.info("[RSSM] AMP GradScaler enabled.")
            except Exception as e:
                logger.warning(f"[RSSM] AMP không khởi tạo được: {e} — fallback fp32")

        self.memory = deque(maxlen=NangConfig.MEMORY_MAXLEN)
        self.current_episode = []
        self.seq_len   = NangConfig.SEQ_LEN
        self.batch_size = batch_size
        # [F13] deque(maxlen) tự drop phần tử cũ → không cần pop(0) O(n)
        self.loss_history = deque(maxlen=NangConfig.LOSS_HISTORY_MAX)

        self.reset_latent()

    def reset_latent(self):
        self.h_online, self.z_online = self.rssm.initial_state(1, self.device)
        self.prev_action = torch.zeros(1, NangConfig.CONV_ACTION_DIM).to(self.device)
        self.prev_action[0, 0] = 1.0  # default action = calm

    def get_latent(self) -> tuple:
        """
        Export current latent state (h, z) cho LatentAdapter.
        Trả về numpy arrays để dùng trong research/latent_adapter.py.
        Gọi SAU get_action() để lấy state đã update.
        """
        return (
            self.h_online.detach().cpu().numpy(),   # (1, det_dim)
            self.z_online.detach().cpu().numpy(),   # (1, stoch_dim)
        )

    def get_action(self, state, explore=True):
        with torch.no_grad():
            obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            self.h_online, self.z_online, _, _ = \
                self.rssm.forward_posterior(
                    self.h_online,
                    self.z_online,
                    self.prev_action,
                    obs
                )

            hz = torch.cat([self.h_online, self.z_online], dim=-1)
            logits = self.actor(hz)

            onehot, action = straight_through_sample(logits, action_dim=NangConfig.CONV_ACTION_DIM)

            if not explore:
                action = torch.argmax(logits, dim=-1)
                onehot = F.one_hot(action, NangConfig.CONV_ACTION_DIM).float()

            self.prev_action = onehot.detach()
            return action.item()

    def remember(self, state, action, next_state, reward, done):
        self.current_episode.append((state, action, next_state, reward, done))
        if done or len(self.current_episode) >= 200:
            self.memory.append(self.current_episode)
            self.current_episode = []

    def replay(self):
        valid_eps = [ep for ep in self.memory if len(ep) >= self.seq_len]
        # [H3] MIN_BATCH guard: tránh training với quá ít data → index lỗi + noise
        if len(valid_eps) < NangConfig.MIN_BATCH:
            return 0.0

        b_size = min(self.batch_size, len(valid_eps))
        # [#5] Recency-biased sampling thay vì uniform random
        # Episodes gần nhất có reward signal tốt hơn (agent đã học được policy tốt hơn)
        # Chia memory thành: 70% từ nửa sau (recent), 30% từ nửa đầu (old)
        # Đơn giản hơn PER đầy đủ nhưng cải thiện learning speed đáng kể
        n      = len(valid_eps)
        split  = max(1, n // 2)
        recent = valid_eps[split:]   # nửa sau — episodes gần nhất
        older  = valid_eps[:split]   # nửa đầu — episodes cũ hơn
        n_recent = max(1, int(b_size * 0.7))
        n_older  = b_size - n_recent
        sampled_recent = random.sample(recent, min(n_recent, len(recent)))
        sampled_older  = random.sample(older,  min(n_older,  len(older)))
        episodes = sampled_recent + sampled_older
        # Shuffle để tránh batch toàn recent ở đầu
        random.shuffle(episodes)

        s_batch, a_batch, r_batch, d_batch = [], [], [], []
        for ep in episodes:
            start = random.randint(0, len(ep) - self.seq_len)
            chunk = ep[start:start+self.seq_len]
            s_batch.append([x[0] for x in chunk])
            a_batch.append([x[1] for x in chunk])
            # [FIX] Bỏ /10.0 tùy tiện — ConversationEnv đã normalize reward về [-1,1]
            # Chia 10 làm reward quá nhỏ → reward_model không học được signal thực
            r_batch.append([x[3] for x in chunk])
            d_batch.append([float(x[4]) for x in chunk])

        states  = torch.tensor(np.array(s_batch), dtype=torch.float32).transpose(0, 1).to(self.device)
        actions = F.one_hot(torch.tensor(np.array(a_batch), dtype=torch.int64).transpose(0, 1), NangConfig.CONV_ACTION_DIM).float().to(self.device)
        rewards = torch.tensor(np.array(r_batch), dtype=torch.float32).unsqueeze(-1).transpose(0, 1).to(self.device)
        dones   = torch.tensor(np.array(d_batch), dtype=torch.float32).unsqueeze(-1).transpose(0, 1).to(self.device)

        h, z = self.rssm.initial_state(b_size, self.device)

        kl_loss = 0
        recon_loss = 0
        reward_loss = 0

        self.opt_model.zero_grad()

        # [#4] AMP autocast cho world model training
        # Chỉ áp dụng cho RSSM/Decoder/RewardModel — không đụng LLM chính
        _amp_ctx = torch.cuda.amp.autocast() if self._amp_enabled else contextlib.nullcontext()

        # ------------------------------------------------------------------ #
        # [P1] FREE BITS: clamp tổng KL theo Dreamer v2 gốc                  #
        #   → free_bits = 1.0 (scalar), clamp SAU khi sum(dim=-1)            #
        #   → KHÔNG clamp per-dimension như trước                            #
        # ------------------------------------------------------------------ #
        free_bits = torch.tensor(1.0).to(self.device)

        with _amp_ctx:
            for t in range(self.seq_len):
                # Reset latent nếu timestep trước đã done
                if t > 0:
                    mask = 1.0 - dones[t-1]
                    h = h * mask
                    z = z * mask

                h_prev, z_prev = h, z

                # [FIX GRU DOUBLE UPDATE] Tính h_t MỘT LẦN, dùng cho cả posterior và prior
                # Dreamer v2: prior và posterior phải dùng cùng h_t
                h = self.rssm.step_h(h_prev, z_prev, actions[t])

                # Posterior: p(z_t | h_t, o_t)
                z_post, mu_post, std_post = self.rssm.forward_posterior_from_h(h, states[t])

                # Prior: p(z_t | h_t) — cùng h_t, không tính lại GRU
                mu_prior, std_prior = self.rssm.forward_prior_from_h(h)

                post_sg  = D.Normal(mu_post.detach(), std_post.detach())
                prior_sg = D.Normal(mu_prior.detach(), std_prior.detach())
                post     = D.Normal(mu_post, std_post)
                prior    = D.Normal(mu_prior, std_prior)

                # [P1] Sum per-sample TRƯỚC khi clamp → đúng chuẩn Dreamer v2
                kl_lhs = torch.max(
                    D.kl_divergence(post_sg, prior).sum(dim=-1, keepdim=True),
                    free_bits
                )
                kl_rhs = torch.max(
                    D.kl_divergence(post, prior_sg).sum(dim=-1, keepdim=True),
                    free_bits
                )
                kl_step = 0.8 * kl_lhs + 0.2 * kl_rhs

                hz = torch.cat([h, z_post], dim=-1)

                recon  = self.decoder(hz)
                r_pred = self.reward_model(hz)

                # [F11] Mask reward/KL/recon sau done → tránh noise vào world model
                step_mask = (1.0 - dones[t-1]) if t > 0 else torch.ones_like(dones[0])

                kl_loss    += (kl_step * step_mask).mean()
                recon_loss += (F.mse_loss(recon, states[t], reduction='none').mean(dim=-1, keepdim=True) * step_mask).mean()
                reward_loss += (F.mse_loss(r_pred, rewards[t], reduction='none') * step_mask).mean()

                z = z_post

            model_loss = (recon_loss + reward_loss + 0.1 * kl_loss) / self.seq_len

        # [#8] Check overflow trước khi backward — AMP có thể sinh NaN silent
        if not torch.isfinite(model_loss):
            logger.warning(f"[RSSM] model_loss không finite ({model_loss.item()}) — skip update")
            # [#8] Vẫn gọi scaler.update() để GradScaler có thể recover scale
            if self._amp_enabled and self._scaler is not None:
                self._scaler.update()
            for m in [self.rssm, self.reward_model, self.decoder]:
                for p in m.parameters(): p.requires_grad = True
            return 0.0

        if self._amp_enabled and self._scaler is not None:
            self._scaler.scale(model_loss).backward()
            self._scaler.unscale_(self.opt_model)
            torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 1.0)
            self._scaler.step(self.opt_model)
            self._scaler.update()
        else:
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.rssm.parameters(), 1.0)
            self.opt_model.step()

        # ================================================================== #
        # STEP 2: IMAGINED ROLLOUT (TRAIN ACTOR & VALUE)                     #
        # ================================================================== #
        # [FIX] Detach h và z_post sau world model loop trước khi dùng cho
        # imagined rollout. Nếu không detach, gradient từ actor/critic
        # backward có thể chảy ngược qua entire world model sequence — đây là
        # silent gradient leak nguy hiểm nhất trong Dreamer implementation.
        z_last  = z_post.detach()
        h_im    = h.detach()
        z_im    = z_last

        imag_hz           = []
        imag_rewards      = []
        imag_values       = []   # online critic  → dùng train critic
        imag_target_values = []  # [P2] EMA target → dùng bootstrap λ-return

        # [FIX] Freeze World Model TRƯỚC khi bắt đầu imagined rollout.
        # Đặt freeze ở đây thay vì chỉ trước actor backward để đảm bảo
        # reward_model và rssm KHÔNG accumulate gradient trong toàn bộ
        # imagination step — không chỉ ở actor_loss.backward().
        wm_modules = [self.rssm, self.reward_model, self.decoder]
        for m in wm_modules:
            for p in m.parameters(): p.requires_grad = False

        horizon = NangConfig.HORIZON
        # [FIX #6] Track không-done mask trong imagination
        # Khi agent reach terminal → discount = 0 → value không bootstrap qua terminal
        imag_discount = []
        _not_done = torch.ones(b_size, 1).to(self.device)   # bắt đầu tất cả alive

        for _ in range(horizon):
            hz_current = torch.cat([h_im, z_im], dim=-1)
            logits     = self.actor(hz_current)
            a_onehot, _ = straight_through_sample(logits, action_dim=NangConfig.CONV_ACTION_DIM)

            h_im, z_im, _, _ = self.rssm.forward_prior(h_im, z_im, a_onehot)
            hz_next = torch.cat([h_im, z_im], dim=-1)

            r        = torch.clamp(self.reward_model(hz_next), -3.0, 3.0)
            v        = self.value_model(hz_next)
            v_target = self.target_critic(hz_next)

            imag_hz.append(hz_next)
            imag_rewards.append(r * _not_done)         # mask reward sau terminal
            imag_values.append(v * _not_done)
            imag_target_values.append(v_target * _not_done)
            imag_discount.append(_not_done.clone())
            # Approximate done: nếu reward < -0.5 coi như terminal signal
            # Không có done flag thực trong imagination → dùng reward làm proxy
            _not_done = _not_done * (torch.abs(r) < 2.5).float()   # ổn định hơn khi reward spike

        imag_hz            = torch.stack(imag_hz)            # [H, B, hz_dim]
        imag_rewards       = torch.stack(imag_rewards)       # [H, B, 1]
        imag_values        = torch.stack(imag_values)        # [H, B, 1]
        imag_target_values = torch.stack(imag_target_values) # [H, B, 1]

        # [FIX] KHÔNG detach imag_hz — actor cần gradient chảy qua imag_hz
        # WM params đã freeze (requires_grad=False) nên gradient KHÔNG chảy vào WM
        # Nếu detach: actor_loss.backward() không có gì để backprop → actor chỉ học entropy

        # ------------------------------------------------------------------ #
        # [P2] Lambda return bootstrap từ target_critic (EMA)                 #
        # ------------------------------------------------------------------ #
        # [FIX 1.1] BỎ with torch.no_grad() bao quanh returns.               #
        # Lý do: returns phải nằm trong computation graph để gradient         #
        # chảy ngược qua imag_rewards → imag_hz → Actor khi backward().      #
        # imag_target_values.detach() đã đủ để ngăn gradient vào Critic      #
        # Target — KHÔNG cần no_grad ở đây.                                  #
        # Nếu bọc no_grad: Actor chỉ học từ entropy bonus, hoàn toàn mù      #
        # với reward signal → Actor không bao giờ học được policy tốt.       #
        returns = lambda_return(
            imag_rewards,
            imag_target_values.detach().clamp(-10.0, 10.0),
            discount  = NangConfig.DISCOUNT,
            lam       = NangConfig.LAMBDA,
            not_done  = torch.stack(imag_discount) if imag_discount else None,
        )

        # --- CRITIC ---
        self.opt_value.zero_grad()
        critic_loss = F.mse_loss(imag_values, returns.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 1.0)
        self.opt_value.step()

        # --- ACTOR ---
        self.opt_actor.zero_grad()
        actor_logits = self.actor(imag_hz)
        actor_dist    = D.Categorical(logits=actor_logits)
        actor_entropy = actor_dist.entropy().mean()

        returns_norm = (returns - returns.mean()) / (returns.std() + 1e-8)
        # [GPT] detach returns trước actor loss — graph vẫn build dù WM frozen
        # tốn VRAM không cần thiết và risk nếu refactor sau
        actor_loss = -returns_norm.detach().mean() - NangConfig.ENTROPY_COEF * actor_entropy

        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.opt_actor.step()

        # Mở khóa lại World Model sau khi update Actor
        for m in wm_modules:
            for p in m.parameters(): p.requires_grad = True

        update_ema(self.target_critic, self.value_model, tau=0.005)

        loss_item = model_loss.item()
        # [F13] deque(maxlen=100) tự drop phần tử cũ → không cần pop(0) O(n)
        self.loss_history.append(loss_item)

        return loss_item


class PersistentBrain:
    def __init__(self):
        self.path = CONF["FILES"]["SOUL"]
        self.active = self._load_soul()
        self.population = []
        self.is_frozen = False
        self.action_trace = []
        self.lock = threading.Lock()
        self.avg_entropy = 0.0
        self.entropy_trend = 0.0
        self.last_action = -1
        self.cooldown = 0
        self.dreamer = DeepMindRSSMAgent()

    def _load_soul(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                # [#10] Filter extra keys — tránh crash khi JSON cũ có field lạ
                valid_keys = Genome.__dataclass_fields__.keys()
                filtered   = {k: v for k, v in data.items() if k in valid_keys}
                missing    = set(valid_keys) - set(filtered)
                if missing:
                    logger.warning(f"[Soul] Missing keys in saved soul: {missing} — dùng default")
                return Genome(**filtered)
            except Exception as e:
                logger.warning(f"[Soul] _load_soul failed: {e} — dùng Genome() mới")
        return Genome()

    def save_soul(self):
        try:
            with open(self.path, "w") as f:
                json.dump(asdict(self.active), f)
        except Exception as e:
            logger.warning(f"[Soul] save_soul failed: {e}")

    def evolve(self, entropy, reward, action):
        with self.lock:
            self.action_trace.append({'a': action, 'ent': entropy, 'r': reward})
            # [FIX] Cap action_trace — không giới hạn sẽ grow vô hạn sau vài nghìn turn
            if len(self.action_trace) > 200:
                self.action_trace = self.action_trace[-200:]
            self.last_action = action

            if self.cooldown > 0: self.cooldown -= 1

            prev_entropy = self.avg_entropy
            # [STRESS_SMOOTHING] EMA với config constant thay vì hardcode 0.7/0.3
            alpha = NangConfig.STRESS_SMOOTHING
            self.avg_entropy  = alpha * self.avg_entropy + (1 - alpha) * entropy
            self.entropy_trend = self.avg_entropy - prev_entropy

            if len(self.active.trauma_history) > 50:
                self.active.trauma_history = self.active.trauma_history[-50:]

            if not self.is_frozen:
                self.active.score += reward - (entropy * 0.1)

            # [FIX] PANIC_ENTROPY_TH = 0.6 (entropy đã normalize [0,1])
            # Threshold cũ 1.5 > max entropy 1.0 → không bao giờ trigger
            # Threshold mới 0.6 = 60% max entropy → trigger khi stress cao bền vững
            trigger_panic = (
                self.avg_entropy > NangConfig.PANIC_ENTROPY_TH
                or (self.avg_entropy > 0.5 and self.entropy_trend > 0.15)
            )

            if not self.is_frozen and trigger_panic and self.cooldown == 0:
                self.is_frozen = True
                reason = "PANIC_ATTACK" if self.avg_entropy > 0.8 else "STRESS_SPIKE"
                self.active.trauma_history.append({"r": reason, "t": __import__('time').time()})

                self.population = []
                parent_clone = copy.deepcopy(self.active)
                parent_clone.score = 0.0
                self.population.append(parent_clone)

                for i in range(4):
                    if i < 2:
                        self.population.append(self._directed_mutate(self.active, reason))
                    else:
                        self.population.append(self._crossover(self.active))
                for g in self.population: g.score = 0.0

            if self.is_frozen:
                for g in self.population:
                    g.score += reward - (entropy * g.e_weights[1])

                # [FIX] Unfreeze threshold cũ 0.8 > PANIC_ENTROPY_TH 0.6 → đúng
                # Unfreeze khi entropy về mức an toàn và không còn tăng
                if self.avg_entropy < NangConfig.PANIC_ENTROPY_TH * 0.8 and self.entropy_trend <= 0:
                    best = max(self.population, key=lambda x: x.score)
                    old_id = self.active.id
                    self.active = copy.deepcopy(best)
                    if best.id != old_id: self.active.generation += 1
                    self.active.score = 0.0
                    self.save_soul()
                    self.is_frozen = False
                    self.population = []
                    # [ACTION_COOLDOWN] Dùng config constant thay vì hardcode 5
                    self.cooldown = NangConfig.ACTION_COOLDOWN * 2
                    self.avg_entropy = NangConfig.PANIC_ENTROPY_TH * 0.3

    def _directed_mutate(self, parent, reason):
        child = copy.deepcopy(parent)
        child.id = __import__('uuid').uuid4().hex[:8]
        w = np.array(child.e_weights)
        if reason in ("PANIC_ATTACK", "STRESS_SPIKE"):
            w[1] *= 1.3
            w[3] *= 1.5
        noise = np.random.normal(0, 0.1, 4)
        w_new = np.clip(w + noise, 0.01, 2.0)
        child.e_weights = (w_new / np.sum(w_new)).tolist()
        return child

    def _crossover(self, parent):
        child = copy.deepcopy(parent)
        child.id = __import__('uuid').uuid4().hex[:8]
        w_p    = np.array(parent.e_weights)
        w_base = np.array([0.25, 0.25, 0.1, 0.4])
        w_new  = (w_p + w_base) / 2.0 + np.random.normal(0, 0.05, 4)
        w_new  = np.clip(w_new, 0.01, 2.0)
        child.e_weights = (w_new / np.sum(w_new)).tolist()
        return child
