import customtkinter as ctk
from tkinterdnd2 import TkinterDnD, DND_FILES
import threading
import queue
import sys
import os
import traceback
import torch

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("Shorts Auto Editor")
        self.geometry("700x740")
        self.minsize(620, 600)

        self.file_paths = {"shorts": "", "movie": ""}
        self.hint_labels = {}
        self.prefix_var     = ctk.StringVar(value="output")
        self.visual_only_var = ctk.BooleanVar(value=True)
        self.monotonic_var  = ctk.BooleanVar(value=True)
        self.device_var     = ctk.StringVar(value="auto")

        self.log_queue = queue.Queue()
        self.running   = False

        self._build_ui()
        self._poll_log()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=20, pady=(16, 6), sticky="ew")
        ctk.CTkLabel(hdr, text="Shorts Auto Editor",
                     font=("Segoe UI", 22, "bold")).pack(side="left")
        if torch.cuda.is_available():
            gpu_txt, gpu_col = f"GPU: {torch.cuda.get_device_name(0)}", "#10b981"
        else:
            gpu_txt, gpu_col = "GPU 없음 (CPU 사용)", "#f59e0b"
        ctk.CTkLabel(hdr, text=gpu_txt, text_color=gpu_col,
                     font=("Segoe UI", 11)).pack(side="right")

        # Drop zones
        drop = ctk.CTkFrame(self)
        drop.grid(row=1, column=0, padx=20, pady=4, sticky="ew")
        drop.grid_columnconfigure((0, 1), weight=1)
        self._make_drop_zone(drop, "📱 숏츠 (Shorts)", "shorts", 0)
        self._make_drop_zone(drop, "🎬 풀영상 (Full Movie)", "movie", 1)

        # Options
        opt = ctk.CTkFrame(self)
        opt.grid(row=2, column=0, padx=20, pady=4, sticky="ew")
        opt.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opt, text="저장 폴더명:").grid(
            row=0, column=0, padx=(14, 6), pady=10, sticky="e")
        ctk.CTkEntry(opt, textvariable=self.prefix_var).grid(
            row=0, column=1, padx=6, pady=10, sticky="ew")
        ctk.CTkCheckBox(opt, text="Visual Only\n(BGM 있는 숏츠)",
                        variable=self.visual_only_var, width=148).grid(
            row=0, column=2, padx=8)
        ctk.CTkCheckBox(opt, text="시간순 정렬\n(중복 방지)",
                        variable=self.monotonic_var, width=130).grid(
            row=0, column=3, padx=(0, 10))

        ctk.CTkLabel(opt, text="처리 장치:").grid(
            row=1, column=0, padx=(14, 6), pady=(0, 10), sticky="e")
        dev_row = ctk.CTkFrame(opt, fg_color="transparent")
        dev_row.grid(row=1, column=1, columnspan=3, padx=6, pady=(0, 10), sticky="w")
        for val, txt in [
            ("auto", "자동 (GPU→CPU 폴백)"),
            ("cuda", "GPU 전용"),
            ("cpu",  "CPU 전용"),
        ]:
            ctk.CTkRadioButton(dev_row, text=txt,
                               variable=self.device_var, value=val).pack(
                side="left", padx=8)

        # Run
        self.run_btn = ctk.CTkButton(
            self, text="▶  실행", height=46,
            font=("Segoe UI", 15, "bold"), command=self._toggle)
        self.run_btn.grid(row=3, column=0, padx=20, pady=6, sticky="ew")

        # Log
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 11), state="disabled")
        self.log_box.grid(row=4, column=0, padx=20, pady=(0, 6), sticky="nsew")

        # Open output
        ctk.CTkButton(
            self, text="📁 출력 폴더 열기", height=36,
            fg_color="transparent", border_width=1, border_color="#444",
            command=self._open_output,
        ).grid(row=5, column=0, padx=20, pady=(0, 16), sticky="ew")

    def _make_drop_zone(self, parent, label, key, col):
        frame = ctk.CTkFrame(parent, height=88, border_width=2,
                              border_color="#333", cursor="hand2")
        frame.grid(row=0, column=col, padx=6, pady=10, sticky="ew")
        frame.grid_propagate(False)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure((0, 1), weight=1)

        ctk.CTkLabel(frame, text=label,
                     font=("Segoe UI", 13, "bold")).grid(row=0, column=0)
        hint = ctk.CTkLabel(frame, text="클릭하거나 파일을 드래그",
                             font=("Segoe UI", 10), text_color="#555")
        hint.grid(row=1, column=0, padx=8)
        self.hint_labels[key] = hint

        def on_drop(event):
            path = event.data.strip()
            if path.startswith("{") and path.endswith("}"):
                path = path[1:-1]
            self._set_file(key, path, frame)

        frame.drop_target_register(DND_FILES)
        frame.dnd_bind("<<Drop>>", on_drop)
        frame.bind("<Button-1>", lambda e: self._browse(key, frame))
        hint.bind("<Button-1>",  lambda e: self._browse(key, frame))

    def _set_file(self, key, path, frame):
        self.file_paths[key] = path
        self.hint_labels[key].configure(
            text=os.path.basename(path), text_color="#ccc")
        frame.configure(border_color="#10b981")

    def _browse(self, key, frame):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.mkv *.avi *.mov *.ts"),
                       ("All files", "*.*")])
        if path:
            self._set_file(key, path, frame)

    # ── Logging ─────────────────────────────────────────────────────────────

    def _log(self, msg):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _poll_log(self):
        try:
            while True:
                self._log(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        self.after(80, self._poll_log)

    # ── Run ─────────────────────────────────────────────────────────────────

    def _toggle(self):
        if self.running:
            import main as m
            m._stop_event.set()
            self.run_btn.configure(text="⛔ 중단 중...", state="disabled",
                                   fg_color="#7f1d1d")
            return
        self._start()

    def _start(self):
        shorts = self.file_paths["shorts"]
        movie  = self.file_paths["movie"]
        if not shorts or not movie:
            self._log("❌ 숏츠와 풀영상을 모두 선택해주세요.")
            return
        if not os.path.exists(shorts):
            self._log(f"❌ 파일 없음: {shorts}"); return
        if not os.path.exists(movie):
            self._log(f"❌ 파일 없음: {movie}"); return

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.run_btn.configure(state="normal", text="■  중단",
                               fg_color="#991b1b", hover_color="#7f1d1d")
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _build_argv(self, device):
        argv = ["main.py",
                "-s", self.file_paths["shorts"],
                "-m", self.file_paths["movie"],
                "-p", self.prefix_var.get().strip() or "output",
                "--device", device]
        if self.visual_only_var.get():
            argv.append("--visual-only")
        if self.monotonic_var.get():
            argv.append("--monotonic")
        return argv

    def _worker(self):
        import main as m

        class QueueStream:
            def __init__(self, q):
                self.q = q
                self.buf = ""
            def write(self, s):
                self.buf += s
                while "\n" in self.buf:
                    line, self.buf = self.buf.split("\n", 1)
                    self.q.put(line)
            def flush(self):
                pass

        import main as m
        m._stop_event.clear()

        old_out, old_err = sys.stdout, sys.stderr
        old_argv = sys.argv
        sys.stdout = sys.stderr = QueueStream(self.log_queue)

        choice = self.device_var.get()
        first_device = "cpu" if choice == "cpu" else "cuda"

        try:
            sys.argv = self._build_argv(first_device)
            m.main()
            self.log_queue.put("✅ 완료!")
        except m.StopProcessing:
            self.log_queue.put("⛔ 처리가 중단되었습니다.")
        except Exception as e:
            cuda_err = any(k in str(e).lower() for k in ("cuda", "out of memory", "gpu"))
            if cuda_err and choice == "auto":
                self.log_queue.put(f"⚠️ GPU 오류 → CPU로 재시도 중...")
                try:
                    m._stop_event.clear()
                    sys.argv = self._build_argv("cpu")
                    m.main()
                    self.log_queue.put("✅ 완료! (CPU 사용)")
                except m.StopProcessing:
                    self.log_queue.put("⛔ 처리가 중단되었습니다.")
                except Exception:
                    self.log_queue.put(f"❌ 오류:\n{traceback.format_exc()}")
            else:
                self.log_queue.put(f"❌ 오류:\n{traceback.format_exc()}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            sys.argv = old_argv
            self.after(0, self._done)

    def _done(self):
        self.run_btn.configure(state="normal", text="▶  실행",
                               fg_color=["#3B8ED0", "#1F6AA5"])
        self.running = False

    def _open_output(self):
        if self.running:
            self._log("⏳ 아직 처리 중입니다. 완료 후 다시 눌러주세요.")
            return
        prefix  = self.prefix_var.get().strip() or "output"
        base    = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(base, "data", "output", prefix)
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            self._log(f"⚠️ 출력 폴더 없음: {out_dir}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
