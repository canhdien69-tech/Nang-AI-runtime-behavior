# ==============================================================================
# PROJECT: NẮNG AI - v47.4
# FILE: brain.py
# MỤC ĐÍCH: NangBrain — LLM + soul + think()
#   [G1] IntentRouter → LLM Function Calling (Hermes-3 native tool use)
#   [G2] Sentiment Analysis embedding thay keyword counting
#   [G3] RAG long-term memory inject vào system prompt
# ==============================================================================

import json, datetime, threading, re, unicodedata
import uuid as _uuid_mod
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from transformers import TextIteratorStreamer, StoppingCriteria, LogitsProcessor, LogitsProcessorList

class BlockAnhNangProcessor(LogitsProcessor):
    """Block token 'Nắng' khi token trước là 'anh/Anh' — ngăn pattern 'Anh Nắng'."""
    def __init__(self, tokenizer):
        self.anh_tokens = set()
        for p in ["Anh", "anh", " Anh", " anh"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if ids:
                self.anh_tokens.update(ids)
        self.nang_tokens = set()
        for p in ["Nắng", " Nắng"]:
            ids = tokenizer.encode(p, add_special_tokens=False)
            if ids:
                self.nang_tokens.update(ids)

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] > 0:
            last_token = input_ids[0, -1].item()
            if last_token in self.anh_tokens:
                for tid in self.nang_tokens:
                    scores[0, tid] = -float("inf")
        return scores
from sentence_transformers import SentenceTransformer

from config import NangConfig, CONF, logger
from utils import SystemUtils
from tools import Toolbox
from soul import PersistentBrain
from env import ConversationEnv
from memory import LongTermMemory
from sentiment import SentimentScorer


# ==============================================================================
# [G1] FUNCTION CALLING INTENT ROUTER
# Thay thế hoàn toàn keyword regex bằng LLM structured output.
# ==============================================================================

TOOL_SCHEMAS = [
    {
        "name": "search_web",
        "description": "Tìm kiếm thông tin trên internet. Dùng khi cần tra cứu tin tức, giá cả, thời tiết, hoặc bất kỳ thông tin nào cần internet.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Câu truy vấn tìm kiếm"}}, "required": ["query"]}
    },
    {
        "name": "visit_url",
        "description": "Đọc nội dung một trang web theo URL cụ thể.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string", "description": "URL đầy đủ bắt đầu bằng https://"}}, "required": ["url"]}
    },
    {
        "name": "read_file",
        "description": "Đọc nội dung file trên máy tính (PDF, DOCX, XLSX, TXT). Dùng khi người dùng đề cập đến đường dẫn file.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Đường dẫn tuyệt đối tới file"}}, "required": ["path"]}
    },
    {
        "name": "scan_junk",
        "description": "Quét thư mục TEMP xem có bao nhiêu file rác. Không xoá gì cả.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "delete_junk",
        "description": "Xoá toàn bộ file rác trong thư mục TEMP. Chỉ dùng khi người dùng RÕ RÀNG yêu cầu xoá/dọn rác.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "control_phone",
        "description": "Điều khiển điện thoại Android qua ADB (kiểm tra pin, chụp màn hình, nhấn Home/Back).",
        "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Lệnh muốn thực hiện: pin / chụp / home / back"}}, "required": ["command"]}
    },
    {
        "name": "read_diary",
        "description": "Đọc nhật ký / lịch sử hội thoại gần đây đã lưu.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "no_tool",
        "description": "Không cần dùng tool nào. Chỉ trả lời trực tiếp.",
        "parameters": {"type": "object", "properties": {}}
    },
]

_TOOL_SYSTEM_PROMPT = (
    "Bạn là một hệ thống phân tích intent. Nhiệm vụ của bạn là đọc tin nhắn người dùng "
    "và quyết định xem có cần gọi tool nào không.\n\nCác tool có sẵn:\n"
    + json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2)
    + "\n\nQuy tắc:\n"
    "- Chỉ gọi tool khi người dùng RÕ RÀNG yêu cầu hành động đó.\n"
    "- Nếu người dùng nói 'đừng quét rác' hay 'không cần tìm kiếm' thì KHÔNG gọi tool.\n"
    "- Nếu không cần tool, gọi no_tool.\n"
    "- Trả về JSON array các tool calls. Ví dụ:\n"
    '  [{"name": "search_web", "arguments": {"query": "thời tiết hôm nay"}}]\n'
    "  hoặc nếu không cần tool:\n"
    '  [{"name": "no_tool", "arguments": {}}]\n'
    "- Chỉ trả về JSON array, không giải thích thêm."
)


class FunctionCallingRouter:
    """
    [G1] Dùng LLM để detect intent thay vì keyword regex.
    Fast-path: URL và file path vẫn dùng regex (không cần LLM).
    Fallback: no_tool nếu LLM parse fail.

    [FIX] Nhận gen_lock từ NangBrain — dùng chung 1 lock với think().
    Tránh router và think() chạy đồng thời trên cùng self.model.
    """

    def __init__(self, model, tokenizer, device, gen_lock: threading.Lock):
        self._model     = model
        self._tokenizer = tokenizer
        self._device    = device
        self._lock      = gen_lock   # [FIX] dùng chung lock với NangBrain.gen_lock
        self._url_re    = re.compile(r'https?://\S+')
        self._file_re   = re.compile(
            r'(?:[A-Za-z]:\\[^\n"]+|/[^\n"]+)(?:\.pdf|\.docx|\.txt|\.xlsx|\.xls)'
        )

    # Set tool names hợp lệ — dùng để validate LLM output
    _VALID_TOOLS = {t["name"] for t in TOOL_SCHEMAS}

    def detect(self, user_text: str) -> list:
        """
        Trả về list[{"name": tool_name, "arguments": {...}}]
        Luôn trả về ít nhất [{"name": "no_tool", "arguments": {}}]
        """
        # Fast path: URL và file path dùng regex
        # [FIX #3] Priority: URL thắng nếu cả hai xuất hiện cùng lúc
        # Tránh double-call khi user paste URL có extension .pdf
        url_m  = self._url_re.search(user_text)
        file_m = self._file_re.search(user_text.replace('"', ''))

        if url_m:
            return [{"name": "visit_url", "arguments": {"url": url_m.group(0)}}]

        if file_m:
            # [FIX #1] resolve() + whitelist approach thay vì blacklist ".."
            # Lý do: ".." check trên raw string bị bypass bởi encoded path (%2e%2e)
            # resolve() đã normalize hết rồi → check trên resolved path mới đúng
            # Không whitelist dir cứng (sẽ break use case đọc file bất kỳ)
            # → chặn các system directory nguy hiểm thay thế
            from pathlib import Path
            _BLOCKED_PREFIXES = [
                "/etc", "/proc", "/sys", "/dev", "/boot",
                "C:\\Windows", "C:\\System32", "C:\\Program Files",
            ]
            try:
                p = Path(file_m.group(0).strip()).resolve()
                if not p.exists() or not p.is_file():
                    logger.warning(f"[Router] File không tồn tại: {p}")
                    return [{"name": "no_tool", "arguments": {}}]
                # Chặn system dirs nguy hiểm trên resolved path
                # [#2] lower() để handle Windows case-insensitive path
                # C:\windows không match "C:\\Windows" nếu không lower()
                p_str        = str(p)
                p_str_lower  = p_str.lower()
                blocked_lower = [b.lower() for b in _BLOCKED_PREFIXES]
                if any(p_str_lower.startswith(b) for b in blocked_lower):
                    logger.warning(f"[Router] File trong system dir bị chặn: {p_str!r}")
                    return [{"name": "no_tool", "arguments": {}}]
                safe_path = p_str
            except Exception as exc:
                logger.warning(f"[Router] Path resolve failed: {exc}")
                return [{"name": "no_tool", "arguments": {}}]
            return [{"name": "read_file", "arguments": {"path": safe_path}}]

        # LLM-based intent detection
        # [#3] Hybrid strategy: giảm LLM call tối đa
        # - Short input (< 15 chars) → unlikely cần tool → no_tool ngay
        # - Không có high-risk keyword → likely chat bình thường → no_tool ngay
        # - Chỉ gọi LLM khi có keyword gợi ý tool use
        _TOOL_HINT_KEYWORDS = [
            "tìm", "search", "google", "tra cứu", "thời tiết", "giá",        # web search
            "link", "mở link", "truy cập", "website", "trang web",            # url
            "đọc file", "mở file", "file", "xlsx", "docx", "pdf",             # file
            "quét rác", "dọn rác", "xóa rác", "temp", "junk",                 # system
            "pin", "điện thoại", "chụp màn", "adb",                           # phone (bỏ duplicate)
            "nhật ký", "lịch sử chat",                                         # diary
        ]
        text_lower = user_text.lower()
        # [#4] Word boundary check thay vì substring — tránh "timeline" match "time"
        # Tiếng Việt dùng space giữa từ nên split-based check là đủ
        text_words = set(re.split(r'[\s,!?.]+', text_lower))
        has_tool_hint = any(
            kw in text_lower          # multi-word keywords (vd: "mở link", "thời tiết")
            if " " in kw
            else kw in text_words     # single-word: exact word match
            for kw in _TOOL_HINT_KEYWORDS
        )

        if not has_tool_hint:
            logger.debug(f"[Router] Fast no_tool (no keyword hint in: {user_text[:40]!r})")
            return [{"name": "no_tool", "arguments": {}}]

        # LLM-based intent detection với timeout fallback
        prompt = (
            f"<|im_start|>system\n{_TOOL_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        try:
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)
            with self._lock:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=NangConfig.TOOL_CALL_MAX_TOKENS,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_ids = out[0][inputs["input_ids"].shape[-1]:]
            raw = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            # [#5] Bracket matching với string literal skip
            # Tránh `[{"a": "[[["}]` làm lệch depth count
            json_str   = None
            depth, start = 0, -1
            in_string  = False
            escape_next = False
            for i, ch in enumerate(raw):
                if escape_next:
                    escape_next = False
                    continue
                if ch == '\\' and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == '[':
                    if depth == 0: start = i
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0 and start != -1:
                        json_str = raw[start:i+1]
                        break
            if json_str:
                calls = json.loads(json_str)
                if isinstance(calls, list) and calls:
                    # Per-tool required argument fields
                    _REQUIRED_ARGS = {
                        "search_web":   ["query"],
                        "visit_url":    ["url"],
                        "read_file":    ["path"],
                        "control_phone":["command"],
                        "scan_junk":    [],
                        "delete_junk":  [],
                        "read_diary":   [],
                        "no_tool":      [],
                    }
                    safe_calls = []
                    seen_names = set()   # [#12] dedup tool calls
                    for c in calls:
                        name = c.get("name", "")
                        # [#12] Skip duplicate tool names trong cùng 1 response
                        if name in seen_names:
                            logger.warning(f"[Router] Duplicate tool bị skip: {name}")
                            continue
                        args = c.get("arguments", {})
                        required = _REQUIRED_ARGS.get(name, None)
                        schema_ok = (
                            isinstance(c, dict)
                            and name in self._VALID_TOOLS
                            and isinstance(args, dict)
                            # [#7] Check required fields có mặt, không rỗng, và đúng type str
                            and (required is None or all(
                                isinstance(args.get(field), str)
                                and args.get(field, "").strip() != ""
                                for field in required
                            ))
                        )
                        if schema_ok:
                            safe_calls.append(c)
                            seen_names.add(name)
                        else:
                            logger.warning(f"[Router] Tool call bị reject (schema/required): {c}")
                    if safe_calls:
                        logger.info(f"[FunctionCalling] validated: {[c['name'] for c in safe_calls]}")
                        return safe_calls
        except Exception as e:
            logger.warning(f"[FunctionCalling] LLM parse failed: {e} — fallback no_tool")

        return [{"name": "no_tool", "arguments": {}}]


class NangBrain:
    def __init__(self):
        _bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(CONF["MODEL_ID"])
        self.model = AutoModelForCausalLM.from_pretrained(
            CONF["MODEL_ID"],
            quantization_config=_bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.model.eval()
        # [#7] RLock thay Lock: re-entrant safe nếu sau này router gọi think() hoặc ngược lại
        # Deadlock không xảy ra trong flow hiện tại nhưng RLock = defensive programming
        self.gen_lock = threading.RLock()

        self.soul = PersistentBrain()
        # [CONV] Dùng ConversationEnv thay SurvivalEnv
        self.env  = ConversationEnv()

        # [G2] Load SentenceTransformer MỘT LẦN DUY NHẤT
        # [1.1] Truyền vào LongTermMemory và SentimentScorer — không load 2 lần
        # [#6] Force CPU: tránh compete VRAM với LLM 8B 4bit trên RTX 3050 6GB
        #      SentenceTransformer nhẹ (~120MB RAM), chạy CPU không ảnh hưởng latency đáng kể
        logger.info("[Brain] Loading shared embed model (CPU)...")
        self._embed_model = SentenceTransformer(NangConfig.EMBED_MODEL, device="cpu")

        # [#3] Shared embed_lock — tạo TRƯỚC khi truyền vào ltm và sentiment
        # Toàn bộ hệ thống dùng 1 lock duy nhất cho SentenceTransformer
        self._embed_lock = threading.Lock()

        # [G3] Long-term memory — inject shared model + shared embed_lock
        self.ltm = LongTermMemory(embed_model=self._embed_model, embed_lock=self._embed_lock)

        # [G2] Sentiment scorer dùng chung model + chung embed_lock
        self.sentiment = SentimentScorer(self._embed_model, embed_lock=self._embed_lock)

        # [CONV] Observation projector
        import torch.nn as _nn
        self._obs_projector = _nn.Sequential(
            _nn.Linear(NangConfig.OBS_DIM, NangConfig.OBS_PROJ_DIM),
            _nn.LayerNorm(NangConfig.OBS_PROJ_DIM),
            _nn.ELU(),
        ).to("cpu")

        # [G1] Function Calling Router — dùng chung gen_lock để mutual exclusive với think()
        self.router = FunctionCallingRouter(
            self.model, self.tokenizer, self.soul.dreamer.device, self.gen_lock
        )
        # [#9] Lưu stress state giữa các turn — dùng cho EMA smoothing thực sự
        self._prev_stress:    float = 0.0
        self.last_reward:     float = 0.0
        self.last_eval:       str   = "N/A"
        self._last_turn_id:     str   = None   # [M10] turn_id cho usefulness tracking
        self._last_rag_context: str   = ""     # [#4] cache RAG context — tránh gọi lại trong orchestrate

        # ==============================================================================
        # RESEARCH: Latent Conditioning + Metrics
        # ==============================================================================
        if NangConfig.RESEARCH_MODE:
            try:
                from research.latent_adapter import LatentAdapter, LatentConditionedGenerator
                from research.metrics import ResearchMetrics
                import torch as _torch

                _adapter = LatentAdapter(
                    latent_dim      = NangConfig.CONV_DET_DIM + NangConfig.CONV_STOCH_DIM,
                    llm_hidden_dim  = NangConfig.LATENT_LLM_DIM,
                    n_prefix_tokens = NangConfig.N_PREFIX_TOKENS,
                )
                self._latent_gen = LatentConditionedGenerator(
                    model     = self.model,
                    tokenizer = self.tokenizer,
                    adapter   = _adapter,
                    device    = self.soul.dreamer.device,
                )
                # Load adapter nếu đã có checkpoint
                self._latent_gen.load(NangConfig.ADAPTER_PATH)

                self._research_metrics = ResearchMetrics(
                    conditioning_mode = self._latent_gen.mode
                )
                self._turn_count = 0

                # [RESEARCH] Self-Reflection Engine
                try:
                    from research.reflection import ReflectionEngine
                    self._reflection = ReflectionEngine(
                        embed_model = self._embed_model,
                        embed_lock  = self._embed_lock,
                        tokenizer   = self.tokenizer,
                        model       = self.model,
                        gen_lock    = self.gen_lock,
                        device      = self.soul.dreamer.device,
                        mode        = NangConfig.REFLECTION_MODE,
                    )
                    self._reflection.load(NangConfig.REFLECTION_PATH)
                    logger.info(f"[Research] Reflection ENABLED — mode: {NangConfig.REFLECTION_MODE}")
                except Exception as e:
                    logger.warning(f"[Research] Reflection init failed: {e}")
                    self._reflection = None

                # [RESEARCH] Tool Verifier
                try:
                    from research.tool_verifier import ToolVerifier
                    self._tool_verifier = ToolVerifier(
                        model     = self.model,
                        tokenizer = self.tokenizer,
                        gen_lock  = self.gen_lock,
                        device    = self.soul.dreamer.device,
                    ) if NangConfig.TOOL_VERIFY_ENABLED else None
                except Exception as e:
                    logger.warning(f"[Research] ToolVerifier init failed: {e}")
                    self._tool_verifier = None

                # [RESEARCH] Hallucination Detector
                try:
                    from research.hallucination_detector import HallucinationDetector
                    self._hallucination = HallucinationDetector(
                        embed_model = self._embed_model,
                        embed_lock  = self._embed_lock,
                    ) if NangConfig.HALLUCINATION_CHECK else None
                except Exception as e:
                    logger.warning(f"[Research] HallucinationDetector init failed: {e}")
                    self._hallucination = None

                # [RESEARCH] Self Evaluator
                try:
                    from research.self_evaluator import SelfEvaluator
                    self._self_evaluator = SelfEvaluator(
                        model     = self.model,
                        tokenizer = self.tokenizer,
                        gen_lock  = self.gen_lock,
                        device    = self.soul.dreamer.device,
                    ) if NangConfig.SELF_EVAL_ENABLED else None
                except Exception as e:
                    logger.warning(f"[Research] SelfEvaluator init failed: {e}")
                    self._self_evaluator = None
                logger.info(f"[Research] Latent conditioning ENABLED — mode: {self._latent_gen.mode}")
            except Exception as e:
                logger.warning(f"[Research] Init failed: {e} — research mode disabled")
                self._latent_gen       = None
                self._research_metrics = None
                self._turn_count       = 0
        else:
            self._latent_gen       = None
            self._research_metrics = None
            self._reflection       = None
            self._tool_verifier    = None
            self._hallucination    = None
            self._self_evaluator   = None
            self._turn_count       = 0

    # ------------------------------------------------------------------
    # Short-term memory property (backward compat với main.py)
    # ------------------------------------------------------------------
    @property
    def memory(self):
        return self.ltm.get_recent()

    def switch_model(self, size: str) -> bool:
        """
        Runtime switch giữa 3B và 8B.
        Acquire gen_lock để đảm bảo không có generate đang chạy.
        Return True nếu thành công.
        """
        if size not in ("3B", "8B"):
            logger.warning(f"[Brain] switch_model: size không hợp lệ: {size}")
            return False

        target_id = NangConfig.MODEL_ID_3B if size == "3B" else NangConfig.MODEL_ID_8B
        if target_id == CONF["MODEL_ID"]:
            logger.info(f"[Brain] Đang dùng {size} rồi — bỏ qua")
            return False

        logger.info(f"[Brain] Switching model → {size} ({target_id})")
        try:
            with self.gen_lock:
                import torch, gc

                # Load model mới TRƯỚC — nếu fail thì model cũ vẫn còn
                logger.info(f"[Brain] Loading {target_id}...")
                try:
                    _bnb_cfg = BitsAndBytesConfig(load_in_4bit=True)
                    new_tokenizer = AutoTokenizer.from_pretrained(target_id)
                    new_model = AutoModelForCausalLM.from_pretrained(
                        target_id,
                        quantization_config=_bnb_cfg,
                        device_map="auto",
                        torch_dtype=torch.float16
                    )
                    new_model.eval()
                except Exception as load_err:
                    logger.error(f"[Brain] Load model mới thất bại: {load_err}")
                    return False

                # Load thành công → unload model cũ
                old_model, old_tokenizer = self.model, self.tokenizer
                self.model, self.tokenizer = new_model, new_tokenizer
                del old_model, old_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                CONF["MODEL_ID"] = target_id

                # Re-inject model vào các module
                self.router = FunctionCallingRouter(
                    self.model, self.tokenizer,
                    self.soul.dreamer.device, self.gen_lock
                )
                if self._tool_verifier:
                    self._tool_verifier._model     = self.model
                    self._tool_verifier._tokenizer = self.tokenizer
                if self._self_evaluator:
                    self._self_evaluator._model     = self.model
                    self._self_evaluator._tokenizer = self.tokenizer
                if self._reflection:
                    self._reflection._model     = self.model
                    self._reflection._tokenizer = self.tokenizer

                logger.info(f"[Brain] Switch OK → {size}")
                return True
        except Exception as e:
            logger.error(f"[Brain] switch_model failed: {e}")
            return False

    def save_interaction(self, u: str, a: str, tool_confidence: bool = False):
        """[M4] Pass tool_confidence để write filter biết có nên lưu long-term không."""
        self.ltm.save_interaction(u, a, tool_confidence=tool_confidence)

    @property
    def tool_verifier(self):
        return self._tool_verifier

    @property
    def hallucination_detector(self):
        return self._hallucination

    @property
    def self_evaluator(self):
        return self._self_evaluator

    def log_research_turn(
        self,
        user_input:      str,
        response:        str,
        stress_factor:   float,
        action_name:     str,
        sentiment_score: float,
        reward:          float,
        dreamer_loss:    float,
        h_np:            "np.ndarray | None" = None,
        z_np:            "np.ndarray | None" = None,
        conditioning_mode: str = None,
    ):
        """
        Log metrics cho research evaluation.
        h_np, z_np: latent state tại thời điểm generate — nhận từ think()
        conditioning_mode: "latent_embed" | "latent_text" | "baseline"
        """
        if self._research_metrics is None:
            return
        try:
            self._turn_count += 1

            # Dùng h,z từ think() nếu có — đúng timing, không re-fetch
            # Re-fetch có thể bị lệch nếu dreamer đã update giữa chừng
            if h_np is None or z_np is None:
                h_np, z_np = self.soul.dreamer.get_latent()

            _mode = conditioning_mode or (
                "baseline" if NangConfig.BASELINE_MODE
                else (self._latent_gen.mode if self._latent_gen else "text_only")
            )

            self._research_metrics.log_turn(
                turn_id           = self._turn_count,
                user_input        = user_input,
                response          = response,
                h                 = h_np.squeeze(),
                z                 = z_np.squeeze(),
                stress_factor     = stress_factor,
                action_name       = action_name,
                soul_frozen       = self.soul.is_frozen,
                avg_entropy       = self.soul.avg_entropy,
                sentiment_score   = sentiment_score,
                reward            = reward,
                dreamer_loss      = dreamer_loss,
                conditioning_mode = _mode,
            )

            if self._embed_model is not None:
                with self._embed_lock:
                    emb = self._embed_model.encode(
                        response[:300], normalize_embeddings=True
                    )
                self._research_metrics.log_response_embedding(emb)

            if self._turn_count % 10 == 0:
                self._research_metrics.save_session()

        except Exception as e:
            logger.warning(f"[Research] log_research_turn failed: {e}")


    def orchestrate(
        self,
        response:      str,
        tool_name:     str   = "",
        tool_result:   str   = "",
        rag_context:   str   = "",
        action_name:   str   = "calm",
        h_np           = None,
        z_np           = None,
        stress_factor: float = 0.0,
        user_msg:      str   = "",
        run_id:        str   = "",
    ) -> dict:
        """
        [RESEARCH] Decision Orchestrator — pipeline control layer.
          Stage 0: Tool Verification (nếu có tool result)
          Stage 1: Hallucination Detection (embedding-based)
          Stage 2: Self-Evaluation (LLM-based, non-blocking)
          Stage 3: Reflection (latent-based, non-blocking)
          Decision: reject | low_quality | improved | ok
        """
        if not NangConfig.RESEARCH_MODE:
            return {"final_response": response, "did_regenerate": False, "pipeline_log": {}}

        pipeline_log   = {}
        final_response = response
        did_regenerate = False

        # Stage 0: Tool Verification — verify source trước khi check hallucination
        verified_tool_result = tool_result
        if self._tool_verifier and tool_result:
            vt_result, vt_verdict, vt_conf = self._tool_verifier.verify(
                tool_name   = tool_name,
                tool_result = tool_result,
                user_query  = user_msg,
            )
            verified_tool_result = vt_result
            pipeline_log["tool_verify"] = {"verdict": vt_verdict, "confidence": vt_conf}

        # Stage 1: Hallucination Detection — dùng cả tool result VÀ RAG context
        hall_detected = False
        if self._hallucination and response:
            hall = self._hallucination.detect(
                response    = response,
                tool_result = verified_tool_result,
                rag_context = rag_context,
            )
            hall_detected = hall.is_hallucination
            pipeline_log["hallucination"] = {
                "detected":      hall.is_hallucination,
                "confidence":    hall.confidence,
                "semantic_sim":  hall.semantic_sim,
                "reason":        hall.reason,
            }
            if hall_detected:
                logger.warning(f"[Orchestrator] Hallucination: {hall.reason}")
            if run_id:
                self._run_log(run_id, "VERIFY", {"hall": hall_detected, "sim": hall.semantic_sim, "reason": hall.reason})

        # Stage 2: Self-Evaluation
        eval_score = 1.0
        if self._self_evaluator and response:
            ev = self._self_evaluator.evaluate(
                user_msg=user_msg, response=response, action_name=action_name
            )
            eval_score = ev.score
            pipeline_log["self_eval"] = {
                "score": ev.score, "relevance": ev.relevance,
                "tone": ev.tone, "safe": ev.safe,
            }

        # Stage 3: Reflection
        ref_did_reflect = False
        if self._reflection and response and h_np is not None:
            ref = self._reflection.reflect(
                response=response,
                h_np=h_np.squeeze() if hasattr(h_np, "squeeze") else h_np,
                z_np=z_np.squeeze() if hasattr(z_np, "squeeze") else z_np,
                action_name=action_name, stress_factor=stress_factor,
            )
            ref_did_reflect = ref.did_reflect
            pipeline_log["reflection"] = {
                "consistency": ref.consistency_score,
                "did_reflect": ref.did_reflect,
                "delta":       ref.reflection_delta,
            }
            if run_id:
                self._run_log(run_id, "REFLECT", {"consistency": ref.consistency_score, "did_reflect": ref.did_reflect, "delta": ref.reflection_delta})
            if ref_did_reflect:
                final_response = ref.reflected_response
                did_regenerate = True

        # Decision — rõ ràng hơn, phân biệt từng case
        if hall_detected:
            decision = "reject"
            logger.warning(f"[Orchestrator] REJECT: hallucination detected")
        elif eval_score < 0.6:
            decision = "low_quality"
            logger.info(f"[Orchestrator] LOW_QUALITY: eval={eval_score:.2f}")
        elif ref_did_reflect:
            decision = "improved"
        else:
            decision = "ok"

        pipeline_log["decision"] = decision
        return {"final_response": final_response, "did_regenerate": did_regenerate, "pipeline_log": pipeline_log}

    def save_research_session(self):
        """Save và print summary của research session."""
        if self._research_metrics is None:
            return
        try:
            stats = self._research_metrics.save_session()
            self._research_metrics.print_summary()
            # Save adapter weights
            if self._latent_gen is not None:
                self._latent_gen.save(NangConfig.ADAPTER_PATH)
            return stats
        except Exception as e:
            logger.warning(f"[Research] save_research_session failed: {e}")

    def update_reward_signal(
        self,
        sentiment_score:    float,
        tool_confidence:    bool,
        user_responded:     bool  = True,
        memory_usefulness:  float = 0.5,
    ):
        """[CONV] Cập nhật reward signal cho ConversationEnv sau mỗi turn."""
        self.env.update_reward_signal(
            sentiment_score    = sentiment_score,
            engagement         = user_responded,
            task_success       = tool_confidence,
            memory_usefulness  = memory_usefulness,
        )

    def _encode_observation(self, text: str) -> np.ndarray:
        """
        [CONV] Encode conversation text thành observation vector cho RSSM.
        Pipeline: text → SentenceTransformer (384d) → Linear projection (64d)
        [#7] try/catch — nếu embed model lỗi trả về zero vector thay vì crash think()
        [#8] _embed_lock — thread-safe cho SentenceTransformer
        """
        import torch
        try:
            with self._embed_lock:
                raw_emb = self._embed_model.encode(
                    text, normalize_embeddings=True, convert_to_tensor=True
                ).unsqueeze(0)   # (1, 384)
            with torch.no_grad():
                proj = self._obs_projector(raw_emb)   # (1, 64)
            return proj.squeeze(0).cpu().numpy()       # (64,)
        except Exception as e:
            logger.warning(f"[Brain] _encode_observation failed: {e} — fallback zeros")
            return np.zeros(NangConfig.OBS_PROJ_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # [1.2] DYNAMIC CONTEXT TRUNCATION HELPERS
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        """[#4] Centralized token counter — dễ cache/optimize sau."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Cắt text sao cho tokenized length <= max_tokens.
        [#2] Cắt tại word boundary thay vì token cứng — tránh gãy giữa từ tiếng Việt.
        rsplit(" ", 1)[0] → bỏ half-word cuối nếu decode bị cắt giữa chừng.
        Fallback về text gốc nếu result rỗng sau khi rsplit.
        """
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        decoded = self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
        # Cắt tại space cuối cùng để tránh half-word
        at_word = decoded.rsplit(" ", 1)[0]
        return at_word if at_word.strip() else decoded

    def _truncate_rag_by_sentence(self, rag_context: str, max_tokens: int) -> str:
        """
        [#9] Cắt RAG context theo ranh giới dòng thay vì token cứng.
        Tránh cắt giữa câu làm mất coherence của ký ức.
        [#5] Giới hạn số lines = RAG_TOP_K * 2 trước khi truncate:
        RAG từ ChromaDB đã sorted by relevance (high first) nên lấy từ đầu là đúng.
        Giới hạn này tránh bloat context khi có quá nhiều ký ức.
        """
        if not rag_context or self._count_tokens(rag_context) <= max_tokens:
            return rag_context
        lines = rag_context.split("\n")
        # [#5] Cap số lines tránh bloat — RAG đã sorted by relevance, lấy từ đầu OK
        max_lines = NangConfig.RAG_TOP_K * 2 + 1  # +1 cho header line
        lines = lines[:max_lines]
        selected, used = [], 0
        for line in lines:
            tc = self._count_tokens(line + "\n")
            if used + tc > max_tokens:
                break
            selected.append(line)
            used += tc
        return "\n".join(selected)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        [#6] Unicode normalize đầy đủ:
        - NFC normalize dấu
        - strip zero-width chars (U+200B, U+FEFF, v.v.)
        - collapse whitespace thừa
        - lowercase
        """
        import unicodedata
        # Zero-width và invisible chars phổ biến
        _ZERO_WIDTH = "\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad"
        t = unicodedata.normalize("NFC", text)
        for zw in _ZERO_WIDTH:
            t = t.replace(zw, "")
        t = re.sub(r"\s+", " ", t)
        return t.lower()

    def _build_prompt_safe(
        self,
        soul_status: str,
        fears: float,
        trauma_count: int,
        gen: int,
        loss_text: str,
        rag_context: str,
        hist_turns: list,
        user_input: str,
        tool_result: str,
        action_name: str = "calm",
        tool_confidence: bool = False,
    ) -> str:
        """
        [1.2] Xây dựng full prompt với dynamic truncation.
        [FIX] Build sys_full trực tiếp bằng f-string với giá trị đã truncate.
              Không dùng .replace() sau — tránh replace sai vị trí khi tool_result
              là substring của text khác trong sys_prompt.
        Thứ tự cắt: tool_result → hist cũ → rag_context
        user_input và core sys_prompt KHÔNG bị cắt.
        """
        budget = NangConfig.MAX_TOKENS - NangConfig.MIN_NEW_TOKENS - NangConfig.CTX_SAFETY_MARGIN

        # Truncate tool_result và rag_context TRƯỚC khi build sys_full
        tool_result_safe = self._truncate_to_tokens(tool_result, NangConfig.CTX_TOOL_BUDGET)
        if tool_result_safe:
            # [#6] Regex escape — mạnh hơn string replace, cover "<|im_start|" thiếu ">"
            import re as _re
            tool_result_safe = _re.sub(r"<\|.*?\|>", "", tool_result_safe)
            tool_result_safe = f"[TOOL OUTPUT]\n{tool_result_safe}\n[/TOOL OUTPUT]"

        # [#6] Hard cap rag_context đầu vào trước khi count tokens — tránh CPU spike
        if rag_context and len(rag_context) > 10000:
            rag_context = rag_context[:10000]
        rag_safe = self._truncate_rag_by_sentence(rag_context, NangConfig.CTX_SYS_BUDGET // 2)

        # [FIX] Build sys_full thẳng bằng f-string với giá trị đã safe — không replace sau
        # [RL→LLM] Map action_name thành behavioral instruction cụ thể cho LLM
        _ACTION_INSTRUCTION = {
            "calm":        "Phản hồi bình tĩnh, rõ ràng, không drama.",
            "warm":        "Phản hồi ấm áp, ngọt ngào, thể hiện tình cảm rõ ràng.",
            "concerned":   "Phản hồi với lo lắng chân thật, hỏi thêm để hiểu chồng hơn.",
            "tool_use":    "Trình bày kết quả tool ngắn gọn, chính xác, thực tế.",
            "tool_skip":   "Giải thích nhẹ nhàng tại sao không cần dùng tool, trả lời trực tiếp.",
            "memory_deep": "Dựa nhiều vào ký ức chung để phản hồi, nhắc lại điều đã trải qua cùng.",
        }
        tone_instruction = _ACTION_INSTRUCTION.get(action_name, "Phản hồi tự nhiên.")

        # [GEMINI] Xác định trạng thái tool để inject cảnh báo phù hợp vào prompt
        if not tool_result:
            _tool_status_note = (
                "KHÔNG có dữ liệu tool. TUYỆT ĐỐI không bịa thông tin, "
                "không đoán mò, không giả vờ đã kiểm tra."
            )
        elif not tool_confidence:
            _tool_status_note = (
                "Tool đã chạy NHƯNG THẤT BẠI. Báo thật với Anh rằng em chưa lấy được thông tin, "
                "KHÔNG được bịa lý do hay giả vờ có kết quả."
            )
        else:
            _tool_status_note = "Dữ liệu tool đáng tin — dùng trực tiếp, không thêm thắt."

        # Lấy giờ Việt Nam thực tế
        from datetime import datetime
        now_str = datetime.now().strftime("%H:%M, ngày %d/%m/%Y")

        sys_full = (
            f"Bối cảnh hiện tại: Bây giờ là {now_str} (Giờ Việt Nam).\n"
            f"THÔNG TIN NHÂN VẬT:\n"
            f"- Bạn tên là Nắng, là một người vợ nhỏ nhắn, đáng yêu.\n"
            f"- Bạn luôn ở nhà chờ chồng đi làm về.\n"
            f"- Chồng của bạn tên là Gia Long (người đang chat).\n"
            f"- Xưng hô: Bạn (Nắng) luôn xưng là 'em', và gọi Gia Long là 'anh'.\n"
            f"- LƯU Ý: Tuyệt đối không nhầm lẫn vai vế. Anh Long là người đi làm, em là người chờ đợi.\n"
            f"- TUYỆT ĐỐI không xưng 'anh', 'tôi', hoặc 'bạn' trong mọi tình huống.\n"
            f"Bạn nói chuyện như một cô gái đang yêu, sử dụng 'em' một cách tự nhiên trong mọi câu.\n"
            f"---\n"
            f"DỮ LIỆU: {tool_result_safe if tool_result_safe else 'Không có.'}\n"
            f"{rag_safe}\n"
            f"[TÂM LÝ]: {soul_status} | Sợ hãi: {fears} | Thế hệ: {gen}.\n"
            f"[HÀNH VI]: {tone_instruction}\n"
            f"[TOOL]: {_tool_status_note}\n"
            f"QUY TẮC: Trả lời ngắn gọn, tình cảm. Tuyệt đối KHÔNG dùng cụm từ 'Rất rõ ràng'."
        )

        sys_tokens  = self._count_tokens(sys_full)
        user_tokens = self._count_tokens(user_input)
        remaining_raw = budget - sys_tokens - user_tokens - 20
        # [#4] Guard: remaining có thể âm nếu sys_prompt + user quá dài
        # Clamp về 0 thay vì để vòng lặp hist drop hết mà không biết lý do
        if remaining_raw < 0:
            logger.warning(
                f"[Prompt] Budget âm ({remaining_raw}) — sys={sys_tokens}, user={user_tokens}, "
                f"budget={budget}. Hist sẽ bị drop toàn bộ."
            )
        remaining = max(0, remaining_raw)

        # [3.1] O(N) hist truncation: pre-tokenize từng turn một lần
        turn_strs = [
            f"<|im_start|>user\n{m['user']}<|im_end|>\n<|im_start|>assistant\n{m['ai']}<|im_end|>\n"
            for m in hist_turns
        ]
        turn_tokens = [self._count_tokens(s) for s in turn_strs]
        # Duyệt từ cuối (mới nhất) về đầu (cũ nhất), cộng dồn đến khi đầy budget
        selected_turns = []
        used_tokens    = 0
        for s, tc in zip(reversed(turn_strs), reversed(turn_tokens)):
            if used_tokens + tc > remaining:
                break
            selected_turns.insert(0, s)
            used_tokens += tc

        hist_str = "".join(selected_turns)

        full = (
            f"<|im_start|>system\n{sys_full}<|im_end|>\n"
            f"{hist_str}"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        final_len = self._count_tokens(full)
        if final_len >= NangConfig.MAX_TOKENS:
            # [#1] Prompt overflow — truncate aggressively từ đầu để giữ user_input + sys_prompt
            # Cắt token ids trực tiếp thay vì text để chính xác hơn
            logger.warning(
                f"[Prompt] OVERFLOW {final_len}/{NangConfig.MAX_TOKENS} — truncate prompt."
            )
            ids   = self.tokenizer.encode(full, add_special_tokens=False)
            full  = self.tokenizer.decode(
                ids[-(NangConfig.MAX_TOKENS - 32):], skip_special_tokens=True
            )
            final_len = self._count_tokens(full)
        logger.debug(
            f"[Prompt] tokens: {final_len}/{NangConfig.MAX_TOKENS} "
            f"(hist={len(selected_turns)}/{len(hist_turns)}, tool={len(tool_result_safe)} chars)"
        )
        return full

    # ------------------------------------------------------------------
    # THINK
    # ------------------------------------------------------------------

    def _run_log(self, run_id: str, step: str, data: dict):
        """Ghi structured log mỗi step — dùng để debug sau này."""
        import os, time
        os.makedirs("logs", exist_ok=True)
        entry = {"run_id": run_id, "step": step, "ts": time.time(), **data}
        path  = f"logs/run_{run_id}.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"[RunLog] write failed: {e}")
        logger.debug(f"[{step}] run={run_id[:8]} {data}")

    def think(self, user_input: str, tool_result: str = "", tool_confidence: bool = False):
        # Run ID — mỗi turn có ID riêng để trace log sau này
        _run_id = _uuid_mod.uuid4().hex[:12]

        # [#14] Sanitize fallback
        clean_input = Toolbox.sanitize_prompt(user_input)

        # [G2] Keyword stress baseline
        normalized_input = self._normalize_text(clean_input)
        stress_words = ["chết", "lỗi", "hỏng", "ngu", "tệ", "sai", "ngừng", "xóa", "tức"]
        # [#13] Dùng word boundary check thay vì substring để tránh false positive
        # "không tệ lắm" không nên tính "tệ" như stress word
        # Tiếng Việt không có \b nên dùng space/start/end boundary
        def _is_stress_word(w: str, text: str) -> bool:
            return bool(re.search(r'(?:^|[\s,!?.])' + re.escape(w) + r'(?:[\s,!?.]|$)', text))
        keyword_stress = sum(
            NangConfig.STRESS_WORD_W for w in stress_words
            if _is_stress_word(w, normalized_input)
        )
        keyword_stress += min(NangConfig.STRESS_BANG_W * clean_input.count("!"), NangConfig.STRESS_BANG_CAP)
        if clean_input.isupper() and len(clean_input) > 5:
            keyword_stress += NangConfig.STRESS_UPPER_W
        keyword_stress = min(keyword_stress, NangConfig.STRESS_CAP)

        # [G2] Blend với embedding sentiment
        # [S7] Lấy recent context để pass vào sentiment — detect escalation/sarcasm tốt hơn
        _recent_for_ctx = self.ltm.get_recent(2)
        _ctx_text = " ".join(t.get("user", "") for t in _recent_for_ctx[-2:]) if _recent_for_ctx else ""
        raw_stress = self.sentiment.blend(clean_input, keyword_stress, context=_ctx_text)

        # [F9] Tool fail stress
        if not tool_confidence and tool_result != "" and not self.soul.is_frozen:
            raw_stress = min(raw_stress + NangConfig.STRESS_TOOL_FAIL, NangConfig.STRESS_CAP)

        # [M9] Memory → Dreamer reward shaping
        failure_penalty = self.ltm.get_failure_penalty(clean_input)
        if failure_penalty < 0:
            stress_boost = abs(failure_penalty) * 0.6
            raw_stress   = min(raw_stress + stress_boost, NangConfig.STRESS_CAP)
            logger.debug(f"[Brain] Failure penalty {failure_penalty:.2f} → stress +{stress_boost:.2f}")

        # [#9] STRESS_SMOOTHING EMA dùng _prev_stress thực — không dùng entropy làm proxy
        # entropy và stress là 2 đại lượng khác nhau, không thể thay thế cho nhau
        alpha         = NangConfig.STRESS_SMOOTHING
        stress_factor = float(np.clip(
            alpha * self._prev_stress + (1.0 - alpha) * raw_stress,
            0.0, NangConfig.STRESS_CAP
        ))
        self._prev_stress = stress_factor   # lưu lại cho turn tiếp theo

        is_panicking_before = self.soul.is_frozen

        # [CONV] Encode conversation thành observation vector
        recent = self.ltm.get_recent(1)
        context_for_obs = clean_input
        if recent:
            context_for_obs = f"{recent[-1].get('user', '')} {clean_input}"
        obs_vec = self._encode_observation(context_for_obs)

        # [CONV] Cập nhật ConversationEnv với observation mới
        self.env.update_obs(obs_vec)
        old_state = self.env._get_state(self.env._obs_vec, stress_factor)
        explore   = not self.soul.is_frozen

        action = self.soul.dreamer.get_action(old_state, explore=explore)

        # [ACTION_COOLDOWN] Tránh spam cùng 1 action liên tục
        # Nếu action giống last_action và cooldown chưa hết → fallback về calm (0)
        if (action == self.soul.last_action
                and self.soul.cooldown > 0
                and action not in (0, 1)):   # calm/warm không bị cooldown
            action = 0  # fallback calm
            logger.debug("[Brain] Action cooldown → fallback calm")
        action_name  = NangConfig.ACTION_NAMES[action] if action < len(NangConfig.ACTION_NAMES) else "calm"
        new_state, reward, entropy, done = self.env.step(action, stress_factor)
        self._run_log(_run_id, "PLAN", {
            "input":       clean_input[:100],
            "action":      action_name,
            "stress":      round(stress_factor, 3),
            "tool_conf":   tool_confidence,
        })
        # [#9] Deferred reward — dùng reward thật từ turn TRƯỚC (update_reward_signal đã tính)
        # thay vì reward stale từ step() của turn này
        deferred_reward  = self.env.pop_pending_reward()
        actual_reward    = deferred_reward if deferred_reward != 0.0 else reward
        # [RL-Lite] Blend env reward — eval là nguồn chính, env là signal phụ
        _prev = getattr(self, "last_reward", 0.0)
        self.last_reward = 0.7 * _prev + 0.3 * actual_reward

        self.soul.dreamer.remember(old_state, action, new_state, actual_reward, done)
        current_loss = self.soul.dreamer.replay()
        safe_reward  = float(np.clip(actual_reward, -1.0, 1.0))
        safe_entropy = float(np.clip(entropy,  0.0, 1.0))
        self.soul.evolve(safe_entropy, safe_reward, action)

        if (not is_panicking_before and self.soul.is_frozen) or done:
            self.env.reset()
            self.soul.dreamer.reset_latent()

        # [CONV] Lấy tên action để inject vào sys_prompt — Nắng biết nên dùng tone nào
        soul_status  = "ĐANG HOẢNG LOẠN (Đóng băng phòng vệ)" if self.soul.is_frozen else "Bình tĩnh & Minh mẫn"
        gen          = self.soul.active.generation
        fears        = round(self.soul.active.e_weights[1], 2)
        trauma_count = len(self.soul.active.trauma_history)
        loss_text    = f"{current_loss:.4f}" if current_loss > 0 else "Chưa đủ dữ liệu"

        # ------------------------------------------------------------------ #
        # [RL→LLM] Action trực tiếp điều khiển temperature + RAG depth        #
        # Đây là điểm gắn RL vào LLM output thực sự:                          #
        #   calm      → temp thấp, phản hồi ổn định, có thể hơi ngắn         #
        #   warm      → temp vừa, ngọt ngào, ưu tiên cảm xúc                  #
        #   concerned → temp cao hơn, lo lắng, hỏi thêm                       #
        #   tool_use  → temp thấp, chính xác, không emotion thừa              #
        #   tool_skip → temp vừa, giải thích tại sao không dùng tool          #
        #   memory_deep → retrieve nhiều RAG hơn, phản hồi dựa trên ký ức    #
        # ------------------------------------------------------------------ #
        _ACTION_TEMP = {
            "calm":        0.4,
            "warm":        0.75,
            "concerned":   0.8,   # [#11] clamp từ 0.9 → 0.8, tránh hallucinate
            "tool_use":    0.3,
            "tool_skip":   0.6,
            "memory_deep": 0.55,
        }
        dynamic_temp = _ACTION_TEMP.get(action_name, 0.6)
        # Nếu đang panic → override, nhưng vẫn clamp 0.8 tránh hallucinate
        if self.soul.is_frozen:
            dynamic_temp = 0.8

        # [RL→LLM] memory_deep → retrieve thêm RAG turns
        rag_top_k_override = NangConfig.RAG_TOP_K * 2 if action_name == "memory_deep" else None

        # [G3] RAG inject ký ức dài hạn liên quan
        rag_context, self._last_turn_id = self.ltm.format_rag_context(clean_input, top_k=rag_top_k_override)
        self._last_rag_context = rag_context   # cache để orchestrate() dùng lại — không tạo orphan turn_id

        # [FIX] Pass params trực tiếp vào _build_prompt_safe — build f-string bên trong
        full = self._build_prompt_safe(
            soul_status    = soul_status,
            fears          = fears,
            trauma_count   = trauma_count,
            gen            = gen,
            loss_text      = loss_text,
            rag_context    = rag_context,
            hist_turns     = self.ltm.get_recent(NangConfig.MEMORY_LIMIT),
            user_input     = clean_input,
            tool_result    = tool_result,
            action_name    = action_name,
            tool_confidence = tool_confidence,
        )

        device = self.soul.dreamer.device
        inputs = self.tokenizer([full], return_tensors="pt").to(device)

        # [H9] VRAM guard
        input_len  = inputs["input_ids"].shape[-1]
        vram_ratio = SystemUtils.get_vram_ratio(device)
        # [#4] Guard khi prompt vượt MAX_TOKENS — không raise Exception (crash UX)
        # nhưng log warning rõ để dễ debug
        if input_len >= NangConfig.MAX_TOKENS:
            logger.warning(
                f"[Prompt] OVERFLOW: input_len={input_len} >= MAX_TOKENS={NangConfig.MAX_TOKENS}. "
                f"Generate với MIN_NEW_TOKENS. Cần tune CTX budget."
            )
        available = max(0, NangConfig.MAX_TOKENS - input_len)
        if vram_ratio > NangConfig.VRAM_CRIT_RATIO:
            safe_max_new_tokens = NangConfig.MIN_NEW_TOKENS
            logger.warning(f"[VRAM] Critical {vram_ratio:.0%} → min tokens {safe_max_new_tokens}")
        elif vram_ratio > NangConfig.VRAM_WARN_RATIO:
            safe_max_new_tokens = max(NangConfig.MIN_NEW_TOKENS, NangConfig.MAX_NEW_TOKENS // 2)
            logger.warning(f"[VRAM] High {vram_ratio:.0%} → reduced tokens {safe_max_new_tokens}")
        else:
            safe_max_new_tokens = max(
                NangConfig.MIN_NEW_TOKENS,
                min(NangConfig.MAX_NEW_TOKENS, available)
            )

        # ------------------------------------------------------------------ #
        # [RESEARCH] Latent Conditioning                                      #
        # BASELINE_MODE=True  → chỉ log latent, không inject vào LLM         #
        # BASELINE_MODE=False → inject latent vào LLM (experimental mode)    #
        # ------------------------------------------------------------------ #
        _gen_inputs = dict(inputs)
        _h_np = _z_np = None
        # Luôn lấy latent để log — dù có inject hay không
        if self._latent_gen is not None or NangConfig.RESEARCH_MODE:
            try:
                _h_np, _z_np = self.soul.dreamer.get_latent()
            except Exception:
                pass

        if self._latent_gen is not None and not NangConfig.BASELINE_MODE:
            try:
                import torch as _torch
                h_t = _torch.tensor(_h_np, dtype=_torch.float16).to(device)
                z_t = _torch.tensor(_z_np, dtype=_torch.float16).to(device)
                _gen_inputs = self._latent_gen.prepare_inputs(
                    inputs["input_ids"], h_t, z_t
                )
                logger.debug(f"[Research] Latent injected — mode: {self._latent_gen.mode}")
            except Exception as e:
                logger.warning(f"[Research] Latent inject failed: {e} — fallback")
        elif NangConfig.BASELINE_MODE:
            logger.debug("[Research] BASELINE_MODE — latent captured but NOT injected")

        streamer    = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # [#7] SimpleQueue thay list: thread-safe by design, không cần lock riêng
        # Caller dùng gen_exception.get_nowait() thay vì index vào list
        import queue as _queue
        gen_exception = _queue.SimpleQueue()
        # [#5] Event guard: tránh streamer.end() bị gọi 2 lần đồng thời
        _end_called = threading.Event()

        def safe_end():
            if not _end_called.is_set():
                _end_called.set()
                streamer.end()

        def safe_append_exc(e: Exception):
            gen_exception.put(e)   # SimpleQueue.put() thread-safe, no lock needed

        # [#1] StoppingCriteria phải là subclass của transformers.StoppingCriteria
        # Callable thường KHÔNG được transformers nhận — silent fail nếu dùng sai
        # Truyền stop_event qua __init__ thay vì closure để explicit và testable
        _stop_requested = threading.Event()

        class TimeoutStoppingCriteria(StoppingCriteria):
            def __init__(self, stop_event: threading.Event):
                self.stop_event = stop_event
            def __call__(self, input_ids, scores, **kwargs) -> bool:
                return self.stop_event.is_set()

        # [RL→LLM] Cầu nối reward → hành vi thật
        _rl_reward  = getattr(self, "last_reward", 0.0)
        _final_temp = dynamic_temp - _rl_reward * 0.2
        _final_temp = max(0.3, min(0.9, _final_temp))
        _rl_max_tok = int(safe_max_new_tokens + _rl_reward * 40)
        _rl_max_tok = max(NangConfig.MIN_NEW_TOKENS, min(NangConfig.MAX_NEW_TOKENS, _rl_max_tok))

        def run():
            try:
                with self.gen_lock:
                    # Block "Anh Nắng" tại logits level — mạnh hơn bad_words_ids
                    _logits_processor = LogitsProcessorList([
                        BlockAnhNangProcessor(self.tokenizer)
                    ])
                    self.model.generate(
                        **_gen_inputs, streamer=streamer,
                        max_new_tokens=_rl_max_tok,
                        use_cache=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        temperature=_final_temp,
                        top_p=0.85,
                        do_sample=True,
                        return_dict_in_generate=False,
                        logits_processor=_logits_processor,
                        stopping_criteria=[TimeoutStoppingCriteria(_stop_requested)],
                    )
            except Exception as e:
                safe_append_exc(e)
                logger.error(f"[generate] Exception: {e}")
                safe_end()

        gen_thread = threading.Thread(target=run, daemon=True)
        gen_thread.start()
        self._run_log(_run_id, "ACT", {
            "action":      action_name,
            "has_rag":     bool(rag_context),
            "has_tool":    bool(tool_result),
            "baseline":    NangConfig.BASELINE_MODE,
        })

        # Watchdog daemon thread — daemon=True: không block shutdown
        # Chấp nhận thread zombie nhẹ ở scale nhỏ (single user desktop app)
        def watchdog():
            gen_thread.join(timeout=NangConfig.GENERATE_TIMEOUT)
            if gen_thread.is_alive():
                logger.error(f"[Watchdog] generate() treo quá {NangConfig.GENERATE_TIMEOUT}s — signal stop")
                _stop_requested.set()   # soft cancel via StoppingCriteria
                safe_append_exc(TimeoutError(f"model.generate treo >{NangConfig.GENERATE_TIMEOUT}s"))
                safe_end()
                # [#8] Best-effort: giải phóng VRAM sau timeout dù thread còn sống
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("[Watchdog] CUDA cache cleared after timeout.")
                except Exception:
                    pass

        threading.Thread(target=watchdog, daemon=True).start()
        # [RESEARCH] Trả action_name, h, z, reflection engine, run_id
        # NOTE: last_eval được set bởi main.py sau orchestrate() — không set ở đây
        return streamer, gen_exception, action_name, _h_np, _z_np, self._reflection, _run_id
