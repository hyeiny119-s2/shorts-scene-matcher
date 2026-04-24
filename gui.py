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

        self.file_paths  = {"shorts": "", "movie": []}
        self.hint_labels = {}
        self.drop_frames = {}
        self.prefix_var      = ctk.StringVar(value="output")
        self.visual_only_var = ctk.BooleanVar(value=True)
        self.monotonic_var   = ctk.BooleanVar(value=True)

        self.log_queue = queue.Queue()
        self.running   = False

        self._build_ui()
        self._poll_log()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=20, pady=(16, 6), sticky="ew")
        ctk.CTkLabel(hdr, text="Shorts Auto Editor",
                     font=("Segoe UI", 22, "bold")).pack(side="left")
        if torch.cuda.is_available():
            gpu_txt, gpu_col = f"GPU: {torch.cuda.get_device_name(0)}", "#10b981"
        else:
            gpu_txt, gpu_col = "No GPU (using CPU)", "#f59e0b"
        ctk.CTkLabel(hdr, text=gpu_txt, text_color=gpu_col,
                     font=("Segoe UI", 11)).pack(side="right")

        # Drop zones
        drop = ctk.CTkFrame(self)
        drop.grid(row=1, column=0, padx=20, pady=4, sticky="ew")
        drop.grid_columnconfigure((0, 1), weight=1)
        self._make_drop_zone(drop, "📱 Shorts", "shorts", 0)
        self._make_drop_zone(drop, "🎬 Movie (multiple files supported)", "movie", 1)

        # Options
        opt = ctk.CTkFrame(self)
        opt.grid(row=2, column=0, padx=20, pady=4, sticky="ew")
        opt.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opt, text="Output folder:").grid(
            row=0, column=0, padx=(14, 6), pady=10, sticky="e")
        ctk.CTkEntry(opt, textvariable=self.prefix_var).grid(
            row=0, column=1, padx=6, pady=10, sticky="ew")
        ctk.CTkCheckBox(opt, text="Visual Only\n(for BGM shorts)",
                        variable=self.visual_only_var, width=148).grid(
            row=0, column=2, padx=8)
        ctk.CTkCheckBox(opt, text="Chronological\n(prevent duplicates)",
                        variable=self.monotonic_var, width=148).grid(
            row=0, column=3, padx=(0, 10))

        # Run + Stop buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=3, column=0, padx=20, pady=6, sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)

        self.run_btn = ctk.CTkButton(
            btn_row, text="▶  Run", height=46,
            font=("Segoe UI", 15, "bold"), command=self._start)
        self.run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_row, text="■  Stop", height=46, width=110,
            font=("Segoe UI", 13, "bold"),
            fg_color="#2a2a2a", hover_color="#7f1d1d",
            state="disabled", command=self._stop)
        self.stop_btn.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        ctk.CTkButton(
            btn_row, text="↺  Reset", height=46,
            font=("Segoe UI", 12), fg_color="#2a2a2a", hover_color="#3a3a3a",
            command=self._reset).grid(row=0, column=2)

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self, height=8)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=4, column=0, padx=20, pady=(0, 4), sticky="ew")

        # Log
        self.log_box = ctk.CTkTextbox(self, font=("Consolas", 11), state="disabled")
        self.log_box.grid(row=5, column=0, padx=20, pady=(0, 6), sticky="nsew")

        # Open output
        ctk.CTkButton(
            self, text="📁 Open Output Folder", height=36,
            fg_color="transparent", border_width=1, border_color="#444",
            command=self._open_output,
        ).grid(row=6, column=0, padx=20, pady=(0, 16), sticky="ew")

    def _make_drop_zone(self, parent, label, key, col):
        frame = ctk.CTkFrame(parent, height=88, border_width=2,
                              border_color="#333", cursor="hand2")
        frame.grid(row=0, column=col, padx=6, pady=10, sticky="ew")
        self.drop_frames[key] = frame
        frame.grid_propagate(False)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure((0, 1), weight=1)

        ctk.CTkLabel(frame, text=label,
                     font=("Segoe UI", 13, "bold")).grid(row=0, column=0)
        hint = ctk.CTkLabel(frame, text="Click or drag file here",
                             font=("Segoe UI", 10), text_color="#555")
        hint.grid(row=1, column=0, padx=8)
        self.hint_labels[key] = hint

        def on_drop(event):
            if key == "movie":
                paths = self._parse_paths(event.data)
            else:
                data = event.data.strip()
                paths = [data[1:-1] if data.startswith("{") and data.endswith("}") else data]
            self._set_file(key, paths, frame)

        frame.drop_target_register(DND_FILES)
        frame.dnd_bind("<<Drop>>", on_drop)
        frame.bind("<Button-1>", lambda e: self._browse(key, frame))
        hint.bind("<Button-1>",  lambda e: self._browse(key, frame))

    def _parse_paths(self, data):
        paths, remaining = [], data.strip()
        while remaining:
            if remaining.startswith("{"):
                end = remaining.find("}")
                if end == -1:
                    paths.append(remaining[1:]); break
                paths.append(remaining[1:end])
                remaining = remaining[end+1:].strip()
            else:
                parts = remaining.split(None, 1)
                paths.append(parts[0])
                remaining = parts[1].strip() if len(parts) > 1 else ""
        return [p for p in paths if p]

    def _set_file(self, key, paths, frame):
        if key == "movie":
            self.file_paths[key] = paths
            label = os.path.basename(paths[0]) if len(paths) == 1 else f"{len(paths)} files selected"
        else:
            self.file_paths[key] = paths[0] if paths else ""
            label = os.path.basename(paths[0]) if paths else ""
        self.hint_labels[key].configure(text=label, text_color="#ccc")
        frame.configure(border_color="#10b981")

    def _browse(self, key, frame):
        from tkinter import filedialog
        ft = [("Video files", "*.mp4 *.mkv *.avi *.mov *.ts"), ("All files", "*.*")]
        if key == "movie":
            paths = filedialog.askopenfilenames(filetypes=ft)
            if paths:
                self._set_file(key, list(paths), frame)
        else:
            path = filedialog.askopenfilename(filetypes=ft)
            if path:
                self._set_file(key, [path], frame)

    # ── Logging ─────────────────────────────────────────────────────────────

    def _poll_log(self):
        msgs = []
        try:
            for _ in range(30):
                msgs.append(self.log_queue.get_nowait())
        except queue.Empty:
            pass
        if msgs:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", "\n".join(msgs) + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(100, self._poll_log)

    # ── Run ─────────────────────────────────────────────────────────────────

    def _start(self):
        if self.running:
            return
        shorts = self.file_paths["shorts"]
        movies = self.file_paths["movie"]
        if not shorts or not movies:
            self.log_queue.put("❌ Please select both a shorts file and a movie file.")
            return
        if not os.path.exists(shorts):
            self.log_queue.put(f"❌ File not found: {shorts}"); return
        for m in movies:
            if not os.path.exists(m):
                self.log_queue.put(f"❌ File not found: {m}"); return

        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

        self.run_btn.configure(state="disabled", text="⏳ Processing...")
        self.stop_btn.configure(state="normal", fg_color="#991b1b", hover_color="#7f1d1d")
        self.progress_bar.set(0)
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()
        self._poll_progress()

    def _stop(self):
        import main as m
        m._stop_event.set()
        self.stop_btn.configure(state="disabled", text="Stopping...")

    def _reset(self):
        if self.running:
            return
        self.file_paths["shorts"] = ""
        self.file_paths["movie"]  = []
        for key in ("shorts", "movie"):
            self.hint_labels[key].configure(
                text="Click or drag file here", text_color="#555")
        for frame in self.drop_frames.values():
            frame.configure(border_color="#333")
        self.prefix_var.set("output")
        self.progress_bar.set(0)
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def _build_argv(self, device="auto"):
        argv = ["main.py",
                "-s", self.file_paths["shorts"],
                "-m"] + self.file_paths["movie"] + [
                "-p", self.prefix_var.get().strip() or "output",
                "--device", device]
        if self.visual_only_var.get():
            argv.append("--visual-only")
        if self.monotonic_var.get():
            argv.append("--monotonic")
        return argv

    def _worker(self):
        import main as m
        import logging, warnings, time

        class QueueStream:
            def __init__(self, q):
                self.q = q
                self.buf = ""
            def write(self, s):
                self.buf += s
                while "\n" in self.buf:
                    line, self.buf = self.buf.split("\n", 1)
                    if "\r" in line:
                        line = line.split("\r")[-1]
                    line = line.strip()
                    if line:
                        self.q.put(line)
            def flush(self):
                pass

        m._stop_event.clear()
        _t0 = time.time()

        old_out, old_err = sys.stdout, sys.stderr
        old_argv = sys.argv

        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")

        logging.raiseExceptions = False
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
        warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")

        sys.stdout = sys.stderr = QueueStream(self.log_queue)

        def _elapsed():
            s = int(time.time() - _t0)
            h, m_, s = s // 3600, (s % 3600) // 60, s % 60
            return f"{h}:{m_:02d}:{s:02d}" if h else f"{m_}m {s}s"

        try:
            sys.argv = self._build_argv("cuda")
            m.main()
            self.log_queue.put(f"✅ Done! (elapsed: {_elapsed()})")
        except m.StopProcessing:
            self.log_queue.put(f"⛔ Processing stopped. (elapsed: {_elapsed()})")
        except Exception as e:
            cuda_err = any(k in str(e).lower() for k in ("cuda", "out of memory", "gpu"))
            if cuda_err:
                self.log_queue.put("⚠️ GPU error → retrying with CPU...")
                try:
                    m._stop_event.clear()
                    sys.argv = self._build_argv("cpu")
                    m.main()
                    self.log_queue.put(f"✅ Done! (CPU mode, elapsed: {_elapsed()})")
                except m.StopProcessing:
                    self.log_queue.put(f"⛔ Processing stopped. (elapsed: {_elapsed()})")
                except Exception:
                    self.log_queue.put(f"❌ Error:\n{traceback.format_exc()}")
            else:
                self.log_queue.put(f"❌ Error:\n{traceback.format_exc()}")
        finally:
            sys.stdout, sys.stderr = old_out, old_err
            sys.argv = old_argv
            logging.raiseExceptions = True
            warnings.resetwarnings()
            self.after(0, self._done)

    def _poll_progress(self):
        if not self.running:
            return
        try:
            import main as m
            self.progress_bar.set(m._progress)
        except Exception:
            pass
        self.after(200, self._poll_progress)

    def _done(self):
        import main as m
        self.progress_bar.set(m._progress)
        self.run_btn.configure(state="normal", text="▶  Run")
        self.stop_btn.configure(state="disabled", text="■  Stop",
                                fg_color="#2a2a2a", hover_color="#7f1d1d")
        self.running = False

    def _open_output(self):
        if self.running:
            self.log_queue.put("⏳ Still processing. Please wait until complete.")
            return
        prefix  = self.prefix_var.get().strip() or "output"
        base    = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        movies  = self.file_paths["movie"]
        if len(movies) > 1:
            out_dir = os.path.join(base, "data", "output")
        else:
            stem    = os.path.splitext(os.path.basename(movies[0]))[0] if movies else ""
            out_dir = os.path.join(base, "data", "output", f"{prefix}_{stem}")
        if os.path.exists(out_dir):
            os.startfile(out_dir)
        else:
            self.log_queue.put(f"⚠️ Output folder not found: {out_dir}")


if __name__ == "__main__":
    app = App()
    app.mainloop()
