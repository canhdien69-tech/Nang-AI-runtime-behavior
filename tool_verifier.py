# ==============================================================================
# PROJECT: NẮNG AI — RESEARCH MODULE
# FILE: research/tool_verifier.py
# MỤC ĐÍCH: Tool Verification Layer
#
# LLM verify tool output trước khi inject vào prompt.
# Tránh hallucination loop khi tool trả về data sai/rác.
#
# Pipeline:
#   tool_result → LLM verify prompt → verdict (VALID/INVALID/UNCERTAIN)
#   → nếu INVALID → replace bằng fallback message
#   → nếu UNCERTAIN → flag để LLM downstream biết
# ==============================================================================

import re
from config import NangConfig, logger


class ToolVerifier:
    """
    Verify tool output bằng LLM trước khi inject vào prompt.

    Dùng non-blocking gen_lock — nếu LLM busy thì pass-through,
    không block main generate thread.
    """

    VERDICT_VALID     = "VALID"
    VERDICT_INVALID   = "INVALID"
    VERDICT_UNCERTAIN = "UNCERTAIN"

    def __init__(self, model, tokenizer, gen_lock, device):
        self._model     = model
        self._tokenizer = tokenizer
        self._gen_lock  = gen_lock
        self._device    = device
        logger.info("[ToolVerifier] Init OK")

    def _build_verify_prompt(self, tool_name: str, tool_result: str, user_query: str) -> str:
        return (
            f"<|im_start|>system\n"
            f"Bạn là verifier. Đánh giá tool output có hợp lệ và liên quan không.\n"
            f"Trả lời ĐÚNG 1 trong 3: VALID / INVALID / UNCERTAIN\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Tool: {tool_name}\n"
            f"Query: {user_query[:100]}\n"
            f"Output: {tool_result[:300]}\n"
            f"Verdict?\n"
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def verify(
        self,
        tool_name:   str,
        tool_result: str,
        user_query:  str,
    ) -> tuple:
        """
        Verify tool result.
        Returns: (verified_result, verdict, confidence)
        - verified_result: result đã được verify hoặc fallback
        - verdict: VALID | INVALID | UNCERTAIN
        - confidence: float [0,1]
        """
        if not NangConfig.TOOL_VERIFY_ENABLED:
            return tool_result, self.VERDICT_VALID, 1.0

        if not tool_result or len(tool_result) < 10:
            return tool_result, self.VERDICT_UNCERTAIN, 0.5

        try:
            prompt = self._build_verify_prompt(tool_name, tool_result, user_query)
            inputs = self._tokenizer([prompt], return_tensors="pt").to(self._device)

            # Non-blocking — skip nếu LLM đang busy
            acquired = self._gen_lock.acquire(blocking=False)
            if not acquired:
                logger.debug("[ToolVerifier] gen_lock busy — partial truncate")
                # [#4] Không pass-through hoàn toàn — truncate output dài để giảm hallucination risk
                if len(tool_result) > 200:
                    return tool_result[:200] + "...[TRUNCATED-UNVERIFIED]", self.VERDICT_UNCERTAIN, 0.4
                return tool_result, self.VERDICT_UNCERTAIN, 0.5

            try:
                out = self._model.generate(
                    **inputs,
                    max_new_tokens = NangConfig.TOOL_VERIFY_MAX_TOK,
                    temperature    = 0.1,   # deterministic
                    do_sample      = False,
                    pad_token_id   = self._tokenizer.eos_token_id,
                )
            finally:
                self._gen_lock.release()

            new_ids  = out[0][inputs["input_ids"].shape[-1]:]
            raw      = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip().splitlines()[0].upper()

            if "INVALID" in raw:
                logger.warning(f"[ToolVerifier] INVALID: {tool_name} — {tool_result[:60]}")
                return f"[TOOL OUTPUT KHÔNG HỢP LỆ — {tool_name}]", self.VERDICT_INVALID, 0.8
            elif re.search(r"\bVALID\b", raw):
                # [#1] Dùng word boundary — "INVALID" cũng chứa "VALID" nếu check đơn giản
                return tool_result, self.VERDICT_VALID, 0.8
            else:
                return tool_result, self.VERDICT_UNCERTAIN, 0.5

        except Exception as e:
            logger.warning(f"[ToolVerifier] verify failed: {e}")
            return tool_result, self.VERDICT_UNCERTAIN, 0.5
