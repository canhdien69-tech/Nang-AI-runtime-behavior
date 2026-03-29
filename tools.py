# ==============================================================================
# PROJECT: NẮNG AI - v47.3
# FILE: tools.py
# MỤC ĐÍCH: ToolGuard (permission layer), Toolbox (file/web/phone/system tools)
# ==============================================================================

import os, re, shutil, subprocess, shlex, uuid, time
import requests
import openpyxl
import PyPDF2
from bs4 import BeautifulSoup
from docx import Document
from ddgs import DDGS

from config import NangConfig, logger


# ==============================================================================
# [H7] TOOL GUARD — permission layer + [H8] retry logic + [H12] cooldown
# ==============================================================================
class ToolGuard:
    # Các tool được phép gọi bình thường — phải khớp với tên trong router/main.py
    ALLOWED   = {"read_file", "search_web", "visit_url", "read_diary", "scan_junk"}
    # Các tool nguy hiểm — chỉ chạy khi KHÔNG panic
    DANGEROUS = {"delete_junk", "control_phone"}
    # ADB commands được whitelist — tránh inject bất kỳ lệnh nào khác
    ADB_WHITELIST = {"shell dumpsys battery", "shell input keyevent 3", "shell input keyevent 4"}
    # Prefix được phép cho screencap (dynamic filename)
    ADB_SCREENCAP_PREFIX = "shell screencap -p /sdcard/"
    ADB_PULL_PREFIX      = "pull /sdcard/"
    ADB_RM_PREFIX        = "shell rm /sdcard/"

    def __init__(self):
        self._tool_cooldown_until: dict = {}  # tool_name → timestamp hết cooldown

    def is_allowed(self, tool_name: str, is_panicking: bool) -> tuple[bool, str]:
        """Kiểm tra xem tool có được phép chạy không. Trả về (ok, lý_do)."""
        now = time.time()
        # Kiểm tra cooldown
        if tool_name in self._tool_cooldown_until:
            remaining = self._tool_cooldown_until[tool_name] - now
            if remaining > 0:
                return False, f"Tool {tool_name} đang cooldown ({remaining:.1f}s)"
        # Dangerous tool bị block khi panic
        if tool_name in self.DANGEROUS and is_panicking:
            return False, f"Từ chối {tool_name}: đang panic"
        # Tool lạ không trong danh sách
        if tool_name not in self.ALLOWED and tool_name not in self.DANGEROUS:
            return False, f"Tool {tool_name} không có trong whitelist"
        return True, ""

    def set_cooldown(self, tool_name: str):
        """[H12] Đặt cooldown sau khi tool fail."""
        self._tool_cooldown_until[tool_name] = time.time() + NangConfig.TOOL_COOLDOWN_SEC
        logger.warning(f"[ToolGuard] Cooldown {tool_name} {NangConfig.TOOL_COOLDOWN_SEC}s")

    def validate_adb_cmd(self, cmd_str: str) -> bool:
        """[H1] Kiểm tra ADB command có nằm trong whitelist không."""
        if cmd_str in self.ADB_WHITELIST:
            return True
        # Cho phép screencap / pull / rm với filename động (chỉ alphanum + _ + .)
        safe_fn = r'^[a-zA-Z0-9_]+\.[a-zA-Z0-9]+$'
        for prefix in (self.ADB_SCREENCAP_PREFIX, self.ADB_PULL_PREFIX, self.ADB_RM_PREFIX):
            if cmd_str.startswith(prefix):
                suffix = cmd_str[len(prefix):]
                if re.match(safe_fn, suffix):
                    return True
        logger.warning(f"[ToolGuard] ADB command bị chặn: {cmd_str!r}")
        return False


class Toolbox:
    def __init__(self, tokenizer, guard: "ToolGuard" = None):
        self.tokenizer = tokenizer
        self.adb_path  = "adb"
        # [FIX] Guard được inject từ ngoài vào — không tạo mới trong control_phone()
        # ToolGuard() mới mỗi lần gọi sẽ mất cooldown state
        # Nếu không inject (standalone usage), tạo instance mặc định
        self._guard = guard if guard is not None else ToolGuard()

    def read_file(self, path):
        path = path.strip().replace('"', '')

        # [SECURITY] Defense-in-depth: validate path ngay trong tool
        # Không trust caller đã validate — security phải đa lớp
        from pathlib import Path
        try:
            resolved = str(Path(path).resolve())
            _SYS_PREFIXES = [
                "/etc", "/proc", "/sys", "/dev", "/boot", "/root",
                "C:\\Windows", "C:\\System32", "C:\\Program Files",
            ]
            if any(resolved.lower().startswith(p.lower()) for p in _SYS_PREFIXES):
                return "[FILE]: Truy cập bị từ chối (system directory).", False
        except Exception:
            return "[FILE]: Path không hợp lệ.", False

        if not os.path.exists(path):
            return "[FILE]: Không thấy file.", False
        try:
            raw = ""; ext = path.lower()
            if ext.endswith(('.xlsx', '.xls')):
                wb = openpyxl.load_workbook(path, data_only=True)
                for s in wb:
                    raw += f"\n[Sheet: {s.title}]\n"
                    for r in s.iter_rows(values_only=True):
                        row_d = [str(c) for c in r if c is not None]
                        if row_d: raw += " | ".join(row_d) + "\n"
            elif ext.endswith('.pdf'):
                reader = PyPDF2.PdfReader(path)
                for p in reader.pages: raw += (p.extract_text() or "") + "\n"
            elif ext.endswith('.docx'):
                for p in Document(path).paragraphs: raw += p.text + "\n"
            else:
                # [TOCTOU] O_NOFOLLOW trên Linux chặn symlink attack
                import platform
                if platform.system() != "Windows" and hasattr(os, "O_NOFOLLOW"):
                    try:
                        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(fd, 'r', encoding='utf-8', errors='ignore') as f:
                            raw = f.read()
                    except OSError as e:
                        return f"[FILE]: Truy cập bị từ chối (symlink?): {e}", False
                else:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        raw = f.read()
            return f"\n[DATA FILE]:\n{raw[:6000]}", True
        except Exception as e:
            return f"[LỖI FILE]: {e}", False

    def search_web(self, q):
        """
        Web search với internal timeout cứng.
        [#6 ZOMBIE FIX] Dùng concurrent.futures với timeout để có thể cancel future
        ngay trong tool — không phụ thuộc vào caller timeout.
        DDGS network hang → thread zombie nếu không có guard nội bộ.
        """
        import concurrent.futures as _cf
        def _do_search():
            with DDGS() as ddgs:
                return ddgs.text(q, max_results=3)
        try:
            with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
                future = _ex.submit(_do_search)
                try:
                    r = future.result(timeout=8)   # 8s hard limit — caller có 10s
                except _cf.TimeoutError:
                    return "[WEB]: Search timeout sau 8 giây.", False
            if r:
                res = "\n[INTERNET DATA]:\n"
                for i, x in enumerate(r):
                    res += f"{i+1}. {x['title']}\n   - {x['body']}\n   - Link: {x['href']}\n"
                return res, True
            return "[WEB]: Không có kết quả.", False
        except Exception as e:
            return f"[WEB ERROR]: {e}", False

    def visit_website(self, url):
        """
        Visit URL với timeout cứng trên requests.
        [#6] requests timeout=10 là internal guard — không phụ thuộc caller.
        """
        try:
            h = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=h, timeout=8)   # 8s < caller 10s
            s = BeautifulSoup(r.text, 'html.parser')
            for tag in s(["script", "style", "nav", "footer"]): tag.extract()
            return f"\n[WEB CONTENT]:\n{s.get_text()[:6000]}...", True
        except requests.Timeout:
            return "[WEB]: Timeout sau 8 giây.", False
        except Exception:
            return "Lỗi đọc link.", False

    def control_phone(self, cmd):
        try:
            # [H1] shlex.split thay c.split() → xử lý đúng arg có dấu cách
            # [FIX] Dùng self._guard (inject từ ngoài) thay vì ToolGuard() mới mỗi lần
            def run(c):
                if not self._guard.validate_adb_cmd(c):
                    raise PermissionError(f"ADB command không được phép: {c!r}")
                return subprocess.check_output(
                    [self.adb_path] + shlex.split(c), timeout=5
                ).decode().strip()

            cmd = cmd.lower()

            if "pin" in cmd:
                res = run("shell dumpsys battery")
                lvl = [x.split(':')[1] for x in res.split('\n') if "level" in x]
                return f"[PHONE]: Pin: {lvl[0].strip()}%", True

            elif "chụp" in cmd:
                fn = f"cap_{uuid.uuid4().hex[:4]}.png"
                run(f"shell screencap -p /sdcard/{fn}")
                run(f"pull /sdcard/{fn} .")
                run(f"shell rm /sdcard/{fn}")
                return f"[PHONE]: Ảnh đã lưu: {fn}", True

            elif "home" in cmd:
                # [FIX BLIND RETURN] Chạy lệnh rồi verify bằng dumpsys activity
                # thay vì return "Home." mà không biết điện thoại có phản hồi không
                run("shell input keyevent 3")
                # Lấy top activity sau lệnh để xác nhận đã về home
                try:
                    activity = run("shell dumpsys activity activities | grep mResumedActivity")
                    return f"[PHONE]: Về Home. Activity hiện tại: {activity[:80]}", True
                except Exception:
                    return "[PHONE]: Đã gửi lệnh Home — không verify được activity.", True

            elif "back" in cmd:
                run("shell input keyevent 4")
                try:
                    activity = run("shell dumpsys activity activities | grep mResumedActivity")
                    return f"[PHONE]: Quay lại. Activity hiện tại: {activity[:80]}", True
                except Exception:
                    return "[PHONE]: Đã gửi lệnh Back — không verify được activity.", True

            elif "app" in cmd or "mở" in cmd or "open" in cmd:
                # Cố gắng lấy tên package đang được focus sau khi gửi lệnh
                # Thay vì return "Đã gửi lệnh" không biết có chạy không
                output = run(f"shell {cmd}") if len(cmd) > 5 else ""
                try:
                    focused = run("shell dumpsys window windows | grep mCurrentFocus")
                    return f"[PHONE]: Kết quả: {output[:100] if output else 'OK'} | Focus: {focused[:80]}", True
                except Exception:
                    result_text = output[:200] if output else "Lệnh đã gửi nhưng không có output."
                    return f"[PHONE]: {result_text}", bool(output)

            else:
                # Lệnh tùy ý — chạy và trả về stdout thực thay vì "Đã gửi lệnh"
                # Nếu stdout rỗng → báo rõ thay vì giả vờ thành công
                output = run(f"shell {cmd}")
                if output:
                    return f"[PHONE]: {output[:300]}", True
                else:
                    return "[PHONE]: Lệnh đã thực thi nhưng không có output từ thiết bị.", True

        except PermissionError as e:
            return f"[PHONE]: Lệnh bị chặn bảo mật: {e}", False
        except subprocess.TimeoutExpired:
            return "[PHONE]: Timeout — thiết bị không phản hồi trong 5 giây.", False
        except Exception as e:
            return f"[PHONE]: Lỗi ADB: {e}", False

    def scan_junk(self):
        try:
            # [F3] TEMP có thể None trên Linux/WSL/Termux → fallback /tmp
            t = os.environ.get('TEMP') or os.environ.get('TMP') or '/tmp'
            if not t or not os.path.exists(t):
                return "[SYSTEM]: Không tìm thấy thư mục TEMP.", False
            sz = 0
            for f in os.listdir(t):
                try:
                    p = os.path.join(t, f)
                    if os.path.isfile(p): sz += os.path.getsize(p)
                except: continue
            return f"[SYSTEM]: Rác {round(sz/(1024*1024),2)}MB.", True
        except: return "Lỗi quét.", False

    def delete_junk(self):
        try:
            # [F3] TEMP có thể None trên Linux/WSL/Termux → fallback /tmp
            t = os.environ.get('TEMP') or os.environ.get('TMP') or '/tmp'
            if not t or not os.path.exists(t):
                return "[SYSTEM]: Không tìm thấy thư mục TEMP.", False
            # Guard: không xóa nếu quá nhiều files (config sai hoặc TEMP = system dir)
            _items = os.listdir(t)
            if len(_items) > 1000:
                return f"[SYSTEM]: Quá nhiều files ({len(_items)}) — abort để an toàn.", False
            c = 0
            for f in os.listdir(t):
                p = os.path.join(t, f)
                try:
                    if os.path.isfile(p) or os.path.islink(p): os.unlink(p); c += 1
                    elif os.path.isdir(p): shutil.rmtree(p); c += 1
                except: pass
            return f"[SYSTEM]: Đã dọn {c} mục.", True
        except: return "Lỗi dọn.", False

    def read_diary(self):
        if not os.path.exists(NangConfig.FILES["DIARY"]): return "[DIARY]: Trống.", False
        with open(NangConfig.FILES["DIARY"], 'r', encoding='utf-8') as f: return "".join(f.readlines()[-15:]), True

    @staticmethod
    def sanitize_prompt(text: str) -> str:
        """[H10] Lọc các injection pattern nguy hiểm trước khi đưa vào model.
        Không xoá nội dung hợp lệ — chỉ strip các chuỗi kỹ thuật rõ ràng."""
        # Xoá các tag system injection phổ biến
        patterns = [
            r'<\|im_start\|>.*?<\|im_end\|>',   # chatml injection
            r'\[INST\].*?\[/INST\]',              # llama instruction injection
            r'###\s*(System|Human|Assistant)\s*:', # alpaca/vicuna injection
            r'(?<!\w)<s>(?!\w)|(?<!\w)</s>(?!\w)', # BOS/EOS — chỉ match standalone, không phải <s>text</s>
        ]
        sanitized = text
        for p in patterns:
            sanitized = re.sub(p, '[FILTERED]', sanitized, flags=re.DOTALL | re.IGNORECASE)
        return sanitized
