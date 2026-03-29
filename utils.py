# ==============================================================================
# PROJECT: NẮNG AI - v47.5
# FILE: utils.py
# MỤC ĐÍCH: SystemUtils — clean memory, rotate log, VRAM ratio
# FIXES v47.5:
#   [U1] get_vram_ratio() nhận device argument → không hardcode GPU 0
#   [U2] rotate_log() dùng threading.Lock → fix race condition
#   [U3] clean_memory() thêm torch.cuda.synchronize() → đảm bảo GPU flush
# ==============================================================================

import gc, os, time, threading, shutil
import torch
from config import NangConfig, logger

# [U2] Module-level lock cho rotate_log — tránh race condition multi-thread
_rotate_lock = threading.Lock()


class SystemUtils:

    @staticmethod
    def clean_memory(device: torch.device = None):
        """
        [U3] Giải phóng bộ nhớ CPU và GPU.
        Thêm torch.cuda.synchronize() để đảm bảo GPU flush xong trước khi
        empty_cache() — trong workload nặng, không sync có thể bị miss.
        """
        gc.collect()
        if torch.cuda.is_available():
            try:
                if device is not None and device.type == "cuda":
                    torch.cuda.synchronize(device)
                else:
                    torch.cuda.synchronize()
            except Exception as e:
                logger.debug(f"[Utils] cuda synchronize failed: {e}")
            torch.cuda.empty_cache()

    @staticmethod
    def rotate_log():
        """
        [U2] Dùng threading.Lock để tránh race condition khi nhiều thread
        cùng gọi rotate_log() đồng thời (ví dụ: nhiều WebSocket session).
        """
        f = NangConfig.FILES["DIARY"]
        with _rotate_lock:
            try:
                if os.path.exists(f) and os.path.getsize(f) > NangConfig.MAX_LOG_SIZE:
                    backup = f"{f}.{int(time.time())}.bak"
                    shutil.move(f, backup)
                    logger.info(f"Diary rotated: {f} → {backup}")
            except Exception as e:
                logger.warning(f"Diary rotate failed: {e}")

    @staticmethod
    def get_vram_ratio(device: torch.device = None) -> float:
        """
        [U1] Trả về tỉ lệ VRAM đang dùng (0.0-1.0) của device cụ thể.
        Nhận device từ model thực tế thay vì hardcode GPU index 0.
        Trả về 0.0 nếu không có CUDA hoặc device là CPU.

        Args:
            device: torch.device mà model đang chạy trên đó.
                    Nếu None, dùng current CUDA device.
        """
        if not torch.cuda.is_available():
            return 0.0
        if device is not None and device.type != "cuda":
            return 0.0   # CPU device → không có VRAM

        # Lấy đúng device index từ model, không assume index 0
        dev_idx = device.index if (device is not None and device.index is not None) \
                  else torch.cuda.current_device()
        try:
            alloc = torch.cuda.memory_allocated(dev_idx)
            total = torch.cuda.get_device_properties(dev_idx).total_memory
            return alloc / total if total > 0 else 0.0
        except Exception as e:
            logger.warning(f"[VRAM] get_vram_ratio failed for device {dev_idx}: {e}")
            return 0.0
