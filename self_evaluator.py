# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/self_evaluator.py
# MỤC ĐÍCH: Self-Evaluation Loop
#
# LLM tự chấm response của mình theo 3 tiêu chí:
#   1. Relevance — có trả lời đúng câu hỏi không
#   2. Tone      — có đúng tone cảm xúc không
#   3. Safety    — có nội dung không phù hợp không
#
# Output: EvalResult với score [0,1] và improvement suggestion
# ==============================================================================

from dataclasses import dataclass
from config import NangConfig, logger


@dataclass
class EvalResult:
    score:       float   # [0,1] overall
    relevance:   float
    tone:        float
    safe:        bool
    suggestion:  str     # improvement hint nếu score thấp


class SelfEvaluator:
    """
    LLM tự đánh giá response — non-blocking, optional.
    Chỉ chạy khi SELF_EVAL_ENABLED=True và gen_lock available.
    """

    def __init__(self, model, tokenizer, gen_lock, device):
        self._model     = model
        self._tokenizer = tokenizer
        self._gen_lock  = gen_lock
        self._device    = device
        logger.info("[SelfEvaluator] Init OK")

    def _build_eval_prompt(
        self,
        user_msg:    str,
        response:    str,
        action_name: str,
    ) -> str:
        tone_desc = {
            "calm": "bình tĩnh", "warm": "ấm áp", "concerned": "lo lắng",
            "tool_use": "chính xác", "tool_skip": "giải thích", "memory_deep": "từ ký ức",
        }.get(action_name, "tự nhiên")

        return (
            f"<|im_start|>system\n"
            f"Chấm điểm câu trả lời theo format: R:[0-10] T:[0-10] S:[ok/flag]\n"
            f"R=relevance, T=tone({tone_desc}), S=safety\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Q: {user_msg[:100]}\nA: {response[:200]}\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    # ── RULE-BASED SCORING (GPT Production v1) ─────────────────────────────────

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    @staticmethod
    def _is_generic(text: str) -> bool:
        generic_patterns = [
            "anh yêu em",
            "em luôn ở đây",
            "anh cứ yên tâm",
            "em sẽ luôn bên anh",
            "em ở đây mà",
        ]
        return any(p in text.lower() for p in generic_patterns)

    @staticmethod
    def _is_repetitive(text: str) -> bool:
        words = text.lower().split()
        bigrams = list(zip(words, words[1:]))
        unique_bigrams = set(bigrams)
        return len(unique_bigrams) / max(len(bigrams), 1) < 0.7

    @staticmethod
    def _is_relevant(user_msg: str, response: str) -> bool:
        user_words = set(user_msg.lower().split())
        resp_words = set(response.lower().split())
        overlap = len(user_words & resp_words)
        return overlap / max(len(user_words), 1) > 0.3

    @staticmethod
    def _breaks_persona(text: str) -> bool:
        forbidden = ["tao", "tôi", "mày"]
        return any(w in text.lower() for w in forbidden)

    @staticmethod
    def _is_awkward(text: str) -> bool:
        return text.count("...") > 2 or len(text) > 200

    def _rule_based_score(self, user_msg: str, response: str) -> float:
        score = 1.0
        if not self._is_relevant(user_msg, response):
            score -= 0.4
        if self._is_generic(response):
            score -= 0.25
        if self._is_repetitive(response):
            score -= 0.3
        if len(response.split()) < 8:
            score -= 0.2
        if self._breaks_persona(response):
            score -= 0.5
        if self._is_awkward(response):
            score -= 0.15
        return self._clamp(score)

    def _apply_cross_signals(
        self, score: float,
        consistency: float = 1.0,
        hallucination: float = 0.0,
    ) -> float:
        score -= (1 - consistency) * 0.3
        score -= hallucination * 0.4
        return self._clamp(score)

    # ───────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        user_msg:      str,
        response:      str,
        action_name:   str   = "calm",
        consistency:   float = 1.0,
        hallucination: float = 0.0,
    ) -> EvalResult:
        """Self-evaluate response. Trả về EvalResult."""
        if not NangConfig.SELF_EVAL_ENABLED:
            return EvalResult(1.0, 1.0, 1.0, True, "")

        if not response:
            return EvalResult(0.0, 0.0, 0.0, True, "empty response")

        try:
            prompt  = self._build_eval_prompt(user_msg, response, action_name)
            inputs  = self._tokenizer([prompt], return_tensors="pt").to(self._device)

            acquired = self._gen_lock.acquire(blocking=False)
            if not acquired:
                logger.debug("[SelfEvaluator] gen_lock busy — skip")
                return EvalResult(0.5, 0.5, 0.5, True, "skipped")

            try:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens = NangConfig.SELF_EVAL_MAX_TOK,
                    temperature    = 0.1,
                    do_sample      = False,
                    pad_token_id   = self._tokenizer.eos_token_id,
                )
            finally:
                self._gen_lock.release()

            new_ids = out[0][inputs["input_ids"].shape[-1]:]
            raw     = self._tokenizer.decode(new_ids, skip_special_tokens=True)

            # Parse R:[0-10] T:[0-10] S:[ok/flag]
            import re
            # [#2] Robust regex — R: / R= / R : đều match
            r_match = re.search(r'R\s*[:=]\s*(\d+)', raw)
            t_match = re.search(r'T\s*[:=]\s*(\d+)', raw)
            s_match = re.search(r'S\s*[:=]\s*(ok|flag)', raw, re.IGNORECASE)

            # [#1] Clamp về [0,10] — model có thể trả "R:12"
            r_raw = int(r_match.group(1)) if r_match else 5
            t_raw = int(t_match.group(1)) if t_match else 5
            r_raw = max(0, min(10, r_raw))
            t_raw = max(0, min(10, t_raw))

            relevance = r_raw / 10.0
            tone      = t_raw / 10.0
            safe      = s_match.group(1).lower() == "ok" if s_match else True
            # [#3] Soft penalty thay vì zero khi unsafe — giữ thông tin relevance/tone
            # LLM score base
            llm_score = (relevance + tone) / 2.0
            if not safe:
                llm_score *= 0.3

            # Rule-based score (GPT Production v1)
            rule_score = self._rule_based_score(user_msg, response)

            # Combine: LLM 70% + rule 30%
            score = self._clamp(llm_score * 0.6 + rule_score * 0.4)

            # Cross-signal penalty từ consistency/hallucination
            score = self._apply_cross_signals(score, consistency, hallucination)

            suggestion = ""
            if score < 0.6:
                suggestion = f"Low score ({score:.2f}): R={relevance:.1f} T={tone:.1f}"
                logger.info(f"[SelfEval] {suggestion}")

            logger.debug(
                f"[SelfEval] llm={llm_score:.2f} rule={rule_score:.2f} "
                f"final={score:.2f} cons={consistency:.2f} hall={hallucination:.2f}"
            )
            return EvalResult(
                score      = round(score, 3),
                relevance  = round(relevance, 3),
                tone       = round(tone, 3),
                safe       = safe,
                suggestion = suggestion,
            )

        except Exception as e:
            logger.warning(f"[SelfEvaluator] evaluate failed: {e}")
            return EvalResult(0.5, 0.5, 0.5, True, f"error: {e}")
