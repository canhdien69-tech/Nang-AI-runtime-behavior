# ==============================================================================
# PROJECT: NẮNG AI - v47.3
# FILE: audio.py
# MỤC ĐÍCH: AudioMouth (TTS), AudioEar (STT)
# ==============================================================================

import queue, threading, time, re, uuid, os, asyncio
import pygame
import edge_tts
import speech_recognition as sr


class AudioMouth:
    def __init__(self):
        self.q = queue.Queue(maxsize=50); self.lock = threading.Lock()
        try: pygame.mixer.init()
        except Exception as e: print(f"[Audio] pygame init failed: {e}")
        # [F5] Persistent event loop thay vì asyncio.run() mỗi lần
        # asyncio.run() tạo + huỷ event loop liên tục → lag TTS + tốn tài nguyên
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    def speak(self, text):
        t = re.sub(r'http\S+|[#*<>@|]', '', text).strip()
        if len(t) >= 2:
            if not self.q.full():
                self.q.put(t)

    def _worker(self):
        while True:
            text = self.q.get()
            try:
                with self.lock:
                    fn = f"v_{uuid.uuid4().hex[:6]}.mp3"
                    # [F5] Dùng persistent loop thay vì asyncio.run()
                    future = asyncio.run_coroutine_threadsafe(
                        edge_tts.Communicate(text, "vi-VN-HoaiMyNeural", rate="+15%").save(fn),
                        self._loop
                    )
                    future.result(timeout=15)
                    if os.path.exists(fn):
                        pygame.mixer.music.load(fn); pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy(): time.sleep(0.05)
                        pygame.mixer.music.unload(); os.remove(fn)
            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally: self.q.task_done()


    def close(self):
        """Shutdown event loop — gọi khi app shutdown để tránh loop leak."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception as e:
            print(f"[Audio] close failed: {e}")


class AudioEar:
    def __init__(self, callback_func):
        self.recognizer = sr.Recognizer(); self.callback = callback_func; self.is_listening = False

    def listen_once(self):
        if self.is_listening: return
        threading.Thread(target=self._listen_thread, daemon=True).start()

    def _listen_thread(self):
        self.is_listening = True
        try:
            with sr.Microphone() as source:
                self.callback("SYSTEM", "👂 Đang nghe...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=4, phrase_time_limit=8)
                text = self.recognizer.recognize_google(audio, language="vi-VN")
                self.callback("USER_VOICE", text)
        except: self.callback("SYSTEM", "⚠️ Không rõ.")
        finally: self.is_listening = False
