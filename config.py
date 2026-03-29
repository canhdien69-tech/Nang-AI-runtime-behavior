# ==============================================================================
# PROJECT: NẮNG AI - v47.3
# FILE: config.py
# MỤC ĐÍCH: Config tập trung, logger, SAFE_MODE
# ==============================================================================

import sys, logging
from logging.handlers import RotatingFileHandler

# ==============================================================================
# [H6] CONFIG TẬP TRUNG — xoá magic numbers rải rác
# ==============================================================================
class NangConfig:
    # Model — xem MODEL_SIZE bên dưới để switch 3B/8B
    MAX_TOKENS        = 4096
    # [#4] 32 thay vì 128 — khi VRAM tight không bị ép generate dài
    MAX_NEW_TOKENS    = 512
    MIN_NEW_TOKENS    = 32
    # Memory
    # [#11] 25 turns thay vì 15 — context ổn định hơn, personality nhất quán hơn
    MEMORY_LIMIT      = 25
    MAX_LOG_SIZE      = 10 * 1024 * 1024  # 10 MB
    # Stress
    STRESS_CAP        = 1.5
    STRESS_WORD_W     = 0.2
    STRESS_BANG_W     = 0.3
    STRESS_BANG_CAP   = 0.6
    STRESS_UPPER_W    = 0.5
    # [#2] 0.3 thay vì 0.8 — tool fail 1 lần không nên gần panic ngay
    STRESS_TOOL_FAIL  = 0.3
    # [#3] 0.6 thay vì 1.5 — entropy đã normalize về [0,1], threshold 1.5 không bao giờ trigger
    PANIC_ENTROPY_TH  = 0.6
    # [NEW] Stress smoothing — tránh emotion spike đột ngột
    STRESS_SMOOTHING  = 0.7   # EMA weight: new_stress = 0.7*old + 0.3*raw
    # Dreamer / RL
    FREE_BITS         = 1.0
    EMA_TAU           = 0.005
    DISCOUNT          = 0.99
    LAMBDA            = 0.95
    # [#9] 0.03 thay vì 0.01 — tránh actor collapse, giữ exploration
    ENTROPY_COEF      = 0.03
    # [#10] 10 thay vì 5 — imagination dài hơn, học strategy xa hơn
    HORIZON           = 10
    SEQ_LEN           = 20
    BATCH_SIZE        = 16
    MIN_BATCH         = 5
    MEMORY_MAXLEN     = 500
    LOSS_HISTORY_MAX  = 100
    # Generate watchdog
    GENERATE_TIMEOUT  = 120
    # VRAM guard
    VRAM_WARN_RATIO   = 0.80
    VRAM_CRIT_RATIO   = 0.92
    # Tool
    TOOL_RETRY_MAX    = 2
    TOOL_COOLDOWN_SEC = 5.0
    # [NEW] Action cooldown — tránh agent spam cùng 1 action liên tục
    ACTION_COOLDOWN   = 2     # số turns minimum trước khi lặp lại cùng action
    # FastAPI / WebSocket server
    SERVER_HOST       = "0.0.0.0"
    SERVER_PORT       = 8000
    # RAG / ChromaDB long-term memory
    CHROMA_DIR        = "nang_chroma_db"
    CHROMA_COLLECTION = "nang_memory"
    EMBED_MODEL       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # [#6] TOP_K=6, MIN_SCORE=0.25 — coverage rộng hơn, bỏ sót ít context hơn
    RAG_TOP_K         = 6
    RAG_MIN_SCORE     = 0.25
    # [#7] 10000 thay vì 2000 — memory dài hạn thực sự, không bị xóa quá sớm
    MAX_CHROMA_ENTRIES = 10000
    # Context window budget — rebalanced cho MAX_TOKENS=4096
    # [#5] Budget cũ 300+800+600+150=1850 ~ full 2048, không còn chỗ generate
    # Budget mới cho MAX_TOKENS=4096: tổng ~2400, còn ~1600 cho generate
    CTX_SYS_BUDGET    = 400
    CTX_HIST_BUDGET   = 1000
    CTX_TOOL_BUDGET   = 600
    CTX_SAFETY_MARGIN = 400
    # [#8] Sentiment anchors mở rộng — cover sarcasm, mixed sentiment, nhiều tình huống hơn
    SENTIMENT_NEG_ANCHORS = [
        "tức giận, bực bội, chửi mắng, xúc phạm",
        "buồn, thất vọng, chán nản, khóc",
        "đe dọa, doạ nạt, ép buộc, bắt phải làm",
        "lạnh lùng, thờ ơ, không quan tâm, bỏ mặc",
        "mỉa mai, châm biếm, nói móc, giễu cợt",
        "lo lắng, căng thẳng, stress, áp lực nặng nề",
        "tuyệt vọng, không còn hy vọng, bỏ cuộc",
        "giận dỗi, hờn mát, không thèm nói chuyện",
        "phán xét, chỉ trích, chê bai, phủ nhận",
        "đau khổ, đau lòng, tổn thương, bị phản bội",
    ]
    SENTIMENT_POS_ANCHORS = [
        "yêu thương, nhớ nhung, ngọt ngào, âu yếm",
        "vui vẻ, hạnh phúc, cảm ơn, khen ngợi",
        "bình thường, hỏi thăm, trò chuyện thân thiện",
        "hào hứng, phấn khích, vui mừng, háo hức",
        "quan tâm, chăm sóc, hỏi han, lo lắng tốt",
        "tự hào, hài lòng, thỏa mãn, thành công",
        "tha thứ, thông cảm, thấu hiểu, đồng cảm",
        "thoải mái, nhẹ nhõm, bình an, thư giãn",
        "tin tưởng, hy vọng, lạc quan, tích cực",
        "hài hước, vui đùa, trêu chọc thân thiện",
    ]
    SENTIMENT_WEIGHT  = 0.7
    # [NEW] Sentiment scale — boost sensitivity của sentiment scoring
    SENTIMENT_SCALE   = 1.2
    # Function calling / tool use
    TOOL_CALL_MAX_TOKENS = 256
    # ==============================================================================
    # ConversationEnv — thay thế SurvivalEnv grid 5x5
    # ==============================================================================
    OBS_DIM              = 384
    OBS_PROJ_DIM         = 64
    CONV_ACTION_DIM      = 6
    ACTION_NAMES         = ["calm", "warm", "concerned", "tool_use", "tool_skip", "memory_deep"]
    REWARD_ENGAGEMENT_W  = 0.4
    REWARD_TASK_W        = 0.4
    REWARD_SENTIMENT_W   = 0.2
    REWARD_STEP_PENALTY  = -0.01
    REWARD_GOAL_BONUS    = 1.0
    REWARD_GOAL_TH       = 0.7
    CONV_DET_DIM         = 64
    CONV_STOCH_DIM       = 16
    CONV_HIDDEN_DIM      = 128
    # State dim = OBS_PROJ_DIM + 4 (sentiment/stress/steps/action_scalar)
    # [#3 env fix] Scalar action thay one-hot — tránh shortcut learning
    CONV_STATE_DIM       = 64 + 4   # = OBS_PROJ_DIM + 4 = 68
    # ==============================================================================
    # Research — Latent Conditioning
    # ==============================================================================
    # ==============================================================================
    # Model Selection — switch giữa 3B (dev/research) và 8B (production)
    # ==============================================================================
    MODEL_SIZE           = "3B"   # "3B" | "8B" — đổi cái này để switch model

    # Model IDs
    MODEL_ID_3B          = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    MODEL_ID_8B          = "unsloth/Hermes-3-Llama-3.1-8B-bnb-4bit"
    MODEL_ID             = MODEL_ID_3B if MODEL_SIZE == "3B" else MODEL_ID_8B

    # LLM hidden dim — auto-adjust theo model
    # Qwen 2.5 3B = 2048, Hermes-3 8B (Llama) = 4096
    LATENT_LLM_DIM       = 2048 if MODEL_SIZE == "3B" else 4096

    # Prefix tokens — 3B có VRAM dư nên có thể dùng nhiều hơn
    N_PREFIX_TOKENS      = 8 if MODEL_SIZE == "3B" else 4
    # Research mode toggle — True = log metrics đầy đủ, False = production mode
    RESEARCH_MODE        = True
    # BASELINE_MODE: True = không inject latent vào LLM, chỉ log
    # Dùng để collect baseline data trước khi so sánh với latent conditioning
    # Toggle giữa 2 mode để có A/B comparison data
    BASELINE_MODE        = True
    ADAPTER_PATH         = "research/latent_adapter.pt"
    # Self-reflection
    REFLECTION_MODE      = "reflect_only"   # "reflect_only" | "reflect_regen"
    REFLECTION_THRESHOLD = 0.45
    REFLECTION_PATH      = "research/reflection_scorer.pt"
    # Tool Verification — LLM verify tool output trước khi inject vào prompt
    # Auto-enable cho 3B vì còn VRAM, tắt cho 8B để tránh OOM
    TOOL_VERIFY_ENABLED  = MODEL_SIZE == "3B"
    TOOL_VERIFY_MAX_TOK  = 128
    # Hallucination Detector — embedding confidence check
    HALLUCINATION_CHECK  = MODEL_SIZE == "3B"
    HALLUCINATION_THRESH = 0.3   # thấp = uncertain = potential hallucination
    # Self-Evaluation Loop — tắt mặc định, tốn thêm 1 LLM call
    SELF_EVAL_ENABLED    = False
    SELF_EVAL_MAX_TOK    = 64
    # Files
    FILES = {
        "MEMORY": "nang_memory.json",
        "DIARY":  "nang_diary.txt",
        "SOUL":   "agi_soul.json",
        "LOG":    "nang_ai.log",
    }

# Giữ CONF alias để tương thích code cũ
CONF = {
    "MODEL_ID":    NangConfig.MODEL_ID,
    "MAX_TOKENS":  NangConfig.MAX_TOKENS,   # 4096
    "MEMORY_LIMIT":NangConfig.MEMORY_LIMIT, # 25
    "MAX_LOG_SIZE":NangConfig.MAX_LOG_SIZE,
    "FILES":       NangConfig.FILES,
}

# ==============================================================================
# [H5] LOGGING CHUẨN — RotatingFileHandler, 5MB, backup 3
# [#12] Guard chống duplicate handler khi module bị reload
# ==============================================================================
_log_handler = RotatingFileHandler(
    NangConfig.FILES["LOG"], maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
)
_log_handler.setFormatter(logging.Formatter(
    "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
))
logger = logging.getLogger("NangAI")
logger.setLevel(logging.DEBUG)
# [#12] Chỉ add handler nếu chưa có — tránh spam log x2, x3 khi reload
if not logger.handlers:
    logger.addHandler(_log_handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))

# ==============================================================================
# [H4] SAFE MODE — fallback khi torch/model lỗi nghiêm trọng
# ==============================================================================
SAFE_MODE = False   # True → disable tool + training, chỉ chat echo cơ bản
