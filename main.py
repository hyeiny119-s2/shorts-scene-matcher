import cv2
import numpy as np
import librosa
import argparse
import os
import sys
import subprocess
import tempfile
import queue
import threading
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoImageProcessor, AutoModel
from PIL import Image as PILImage
from scenedetect import detect, AdaptiveDetector
from moviepy import VideoFileClip, concatenate_videoclips
from report import generate_report

SUPPORTED_EXTS       = ('.mp4', '.mkv', '.avi', '.mov', '.ts')
AUDIO_SR             = 11025
AUDIO_CONF_THRESHOLD = 5.0
FRAME_POSITIONS      = [0.1, 0.25, 0.5, 0.75, 0.9]
CROP_H_POSITIONS     = [0.25, 0.5, 0.75]
VISUAL_FPS_SAMPLE    = 2
BATCH_SIZE           = 64
FEAT_DIM             = 384   # DINOv2-Small

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_stop_event      = threading.Event()
_READER_SENTINEL = object()  # sentinel to signal reader thread completion
_progress        = 0.0

class StopProcessing(Exception):
    pass

def _check_stop():
    if _stop_event.is_set():
        raise StopProcessing("Processing stopped by user.")

def _set_progress(val):
    global _progress
    _progress = min(max(float(val), 0.0), 1.0)

# ── Utilities ────────────────────────────────────────────────────────────────

def format_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"

def get_video_size(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def get_shorts_scenes(path, threshold=3.0):
    print(f"\n🎬 Detecting cuts... (threshold={threshold})")
    scenes = []
    for i, sc in enumerate(detect(path, AdaptiveDetector(adaptive_threshold=threshold))):
        s, e = sc[0].get_seconds(), sc[1].get_seconds()
        scenes.append((s, e))
        print(f"  - Cut {i+1}: {format_time(s)} ~ {format_time(e)}")
    print(f"  → {len(scenes)} cuts found")
    return scenes

# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio(video_path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', video_path, '-ac', '1', '-ar', str(AUDIO_SR), '-vn', tmp_path],
            check=True, capture_output=True)
        audio, _ = librosa.load(tmp_path, sr=AUDIO_SR, mono=True)
    finally:
        os.unlink(tmp_path)
    return audio

# ── Audio matching: NCC ───────────────────────────────────────────────────────

def ncc_topk(movie_audio, scene_audio, k=8, gap_s=30.0):
    """Top-k NCC peaks with gap suppression. Returns [(time_s, conf), ...]"""
    N = len(scene_audio)
    valid_len = len(movie_audio) - N + 1
    if valid_len <= 0:
        return []

    scene_n = scene_audio / (np.linalg.norm(scene_audio) + 1e-8)
    n_fft   = 2 ** int(np.ceil(np.log2(len(movie_audio) + N - 1)))
    corr    = np.fft.irfft(
        np.fft.rfft(movie_audio, n_fft) * np.conj(np.fft.rfft(scene_n, n_fft))
    )[:valid_len]

    cumsum    = np.concatenate([[0], np.cumsum(movie_audio ** 2)])
    local_rms = np.sqrt(np.maximum((cumsum[N:] - cumsum[:valid_len]) / N, 1e-8))
    ncc       = corr / local_rms

    top_1pct = float(np.percentile(ncc, 99))
    gap      = int(gap_s * AUDIO_SR)
    ncc_copy = ncc.copy()
    peaks    = []
    for _ in range(k):
        idx   = int(np.argmax(ncc_copy))
        score = float(ncc_copy[idx])
        conf  = (score - top_1pct) / (abs(top_1pct) + 1e-8)
        peaks.append((idx / AUDIO_SR, conf))
        lo = max(0, idx - gap)
        hi = min(len(ncc_copy), idx + gap + 1)
        ncc_copy[lo:hi] = -np.inf
    return peaks  # sorted by NCC score desc

def find_timestamps_by_audio(shorts_scenes, shorts_audio, movie_audio):
    print("\n🎵 [1/3] Audio NCC matching...")
    times, confs, all_candidates = [], [], []
    for i, (start, end) in enumerate(shorts_scenes):
        scene = shorts_audio[int(start * AUDIO_SR):int(end * AUDIO_SR)]
        if len(scene) < AUDIO_SR // 2:
            print(f"  - Scene {i+1}: too short, skipping")
            times.append(None); confs.append(0.0); all_candidates.append([])
            continue
        candidates = ncc_topk(movie_audio, scene, k=8)
        t, conf = candidates[0] if candidates else (None, 0.0)
        warn = "  ⚠️ low confidence" if conf < AUDIO_CONF_THRESHOLD else ""
        t_str = format_time(t) if t is not None else "—"
        print(f"  - Scene {i+1}: {t_str}  (confidence {conf:.1f}){warn}")
        times.append(t); confs.append(conf); all_candidates.append(candidates)
    return times, confs, all_candidates

# ── Visual matching: DINOv2-Small (GPU) ──────────────────────────────────────
#
# DINOv2 (384-dim L2-normalized, self-supervised):
#   - Purely visual similarity, no text — better suited for scene matching
#   - GPU batch processing handles a 2-hour movie in minutes
#   - Cosine similarity search: (M, 384) @ (384, R) → instant results
#
# Left/center/right crops:
#   - Handles auto-framed shorts where the subject is off-center

class DINOv2Extractor:
    """DINOv2-Small wrapper — encode_image(frames_rgb) → (N, 384) tensor"""
    def __init__(self, model_name='facebook/dinov2-small'):
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).eval().to(DEVICE)

    @torch.no_grad()
    def encode_image(self, frames_rgb):
        pil_imgs = [PILImage.fromarray(f) for f in frames_rgb]
        inputs = self.processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats  = self.model(**inputs).last_hidden_state[:, 0, :]  # CLS token
        return F.normalize(feats, dim=1)

def build_feature_extractor():
    """DINOv2-Small visual encoder → 384-dim features"""
    return DINOv2Extractor()

@torch.no_grad()
def frames_to_features(frames_rgb, model):
    """List of RGB numpy arrays → (N, FEAT_DIM) float32 L2-normalized features"""
    feats = model.encode_image(frames_rgb)
    return feats.cpu().numpy().astype(np.float32)

def crop_frame(frame, sw, sh, h_pos=0.5):
    """Crop movie frame to shorts aspect ratio, then resize to 224×224 for DINOv2"""
    fh, fw = frame.shape[:2]
    cw = int(fh * sw / sh)
    if cw < fw:
        x = int((fw - cw) * h_pos)
        frame = frame[:, x:x + cw]
    return cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

def prepare_scene_features(scenes, shorts_path, sw, sh, model):
    """Extract reference features for each shorts scene (GPU)"""
    print("\n  Extracting reference features from shorts...")
    cap = cv2.VideoCapture(shorts_path)
    scene_feats = []
    for start, end in scenes:
        frames = []
        for pos in FRAME_POSITIONS:
            t = start + (end - start) * pos
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, f = cap.read()
            if ret:
                f = cv2.resize(f, (224, 224), interpolation=cv2.INTER_LINEAR)
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        feats = frames_to_features(frames, model) if frames else np.zeros((0, FEAT_DIM), np.float32)
        scene_feats.append(feats)
    cap.release()
    return scene_feats

def precompute_movie_features(movie_path, sw, sh, model, progress_cb=None):
    """
    Single sequential scan of the full movie → GPU batch feature extraction.
    {crop_pos: {frame_idx: feature_vector (384,)}}

    Optimizations:
    - All 3 crops sent in one GPU call (3× kernel launches → 1×)
    - Frame reading (CPU/I/O) pipelined with GPU via queue
    - Resize and BGR→RGB done in reader thread, not GPU loop
    """
    print(f"\n🖥️  [2/3] Extracting movie features ({DEVICE})...")
    cap   = cv2.VideoCapture(movie_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps / VISUAL_FPS_SAMPLE))
    n_pos = len(CROP_H_POSITIONS)

    movie_feats = {pos: {} for pos in CROP_H_POSITIONS}
    batch_q     = queue.Queue(maxsize=4)

    # ── Reader thread (CPU/I/O) ───────────────────────────────────────────────
    def reader():
        pending_crops = []
        pending_fidxs = []
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if _stop_event.is_set():
                break
            if fidx % step == 0:
                for pos in CROP_H_POSITIONS:
                    c = crop_frame(frame, sw, sh, pos)
                    pending_crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                pending_fidxs.append(fidx)
                if len(pending_fidxs) >= BATCH_SIZE:
                    batch_q.put((list(pending_fidxs), list(pending_crops)))
                    pending_crops.clear()
                    pending_fidxs.clear()
            fidx += 1
        if pending_fidxs:
            batch_q.put((pending_fidxs, pending_crops))
        batch_q.put(_READER_SENTINEL)

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    # ── GPU processing (main thread) ──────────────────────────────────────────
    processed = 0
    try:
        while True:
            item = batch_q.get()
            if item is _READER_SENTINEL:
                break
            _check_stop()
            fidxs, crops = item
            # crops order: [f0_pos0, f0_pos1, f0_pos2, f1_pos0, ...] → single GPU call
            feats = frames_to_features(crops, model)   # (B*n_pos, FEAT_DIM)
            for j, fi in enumerate(fidxs):
                for k, pos in enumerate(CROP_H_POSITIONS):
                    movie_feats[pos][fi] = feats[j * n_pos + k]
            processed += len(fidxs)
            if progress_cb and total > 0:
                progress_cb(processed * step / total)
            pct = processed * step / total * 100 if total > 0 else 0.0
            print(f"  ... {min(processed * step, total)}/{total} ({pct:.1f}%)", end='\r')
    finally:
        # Drain queue so reader thread can unblock on stop
        while True:
            try:
                batch_q.get_nowait()
            except queue.Empty:
                break

    t.join()
    print(f"  ... {total}/{total} (100.0%)")
    cap.release()
    return movie_feats, fps, total / fps

def find_timestamps_by_visual(scene_feats, movie_feats, fps, audio_times, audio_confs,
                               search_window, min_sim=0.0):
    """
    DINOv2 cosine similarity search for best matching frame.
    Scenes with low audio confidence use 2× search window.
    Returns None if best similarity is below min_sim.
    """
    print(f"\n👁️  [3/3] Visual DINOv2 matching (±{search_window}s, 3-crop, min_sim={min_sim})...")
    all_fidxs = sorted(next(iter(movie_feats.values())).keys())

    results = []
    for i, (ref_feats, at, ac) in enumerate(zip(scene_feats, audio_times, audio_confs)):
        _check_stop()
        if len(ref_feats) == 0:
            results.append(None)
            print(f"  - Scene {i+1}: no reference frames")
            continue

        win = search_window * 2 if ac < AUDIO_CONF_THRESHOLD else search_window

        if at is not None:
            lo = int((at - win) * fps)
            hi = int((at + win) * fps)
            cand_fidxs = [f for f in all_fidxs if lo <= f <= hi]
        else:
            cand_fidxs = all_fidxs

        if not cand_fidxs:
            results.append(None)
            print(f"  - Scene {i+1}: no candidates found")
            continue

        ref_mean = ref_feats.mean(axis=0)
        ref_mean /= (np.linalg.norm(ref_mean) + 1e-8)

        best_t, best_sim = None, -1.0
        for pos, ph_map in movie_feats.items():
            cand = [(fi, ph_map[fi]) for fi in cand_fidxs if fi in ph_map]
            if not cand:
                continue
            fidxs_arr = np.array([c[0] for c in cand])
            feats_mat = np.stack([c[1] for c in cand])
            sims      = feats_mat @ ref_mean
            max_i     = int(np.argmax(sims))
            if sims[max_i] > best_sim:
                best_sim = float(sims[max_i])
                best_t   = int(fidxs_arr[max_i]) / fps

        scope = f"±{win:.0f}s" if at is not None else "full scan"
        if best_sim < min_sim:
            results.append(None)
            print(f"  - Scene {i+1}: skipped (sim={best_sim:.4f} < {min_sim}, {scope})")
        else:
            results.append(best_t)
            print(f"  - Scene {i+1}: {format_time(best_t)}  (sim={best_sim:.4f}, {scope})")

    return results

# ── Monotonic constraint ──────────────────────────────────────────────────────

def apply_monotonic_constraint(final_times, scenes, min_gap=5.0, buffer=1.0):
    """
    Sort matched timestamps chronologically and remove overlapping duplicates.
    prev_end includes buffer so rendered clips never overlap.
    min_gap: extra seconds after rendered clip ends before next clip starts.
    """
    print(f"\n⏱️  Applying monotonic constraint... (min gap {min_gap}s)")

    pairs = [(t, i) for i, t in enumerate(final_times) if t is not None]
    pairs.sort(key=lambda x: x[0])

    selected = []  # (scene_idx, t, scene_tuple)
    prev_end = -1e9

    for t, idx in pairs:
        dur = scenes[idx][1] - scenes[idx][0]
        if t >= prev_end + min_gap:
            selected.append((idx, t, scenes[idx]))
            prev_end = t + dur + buffer
            print(f"  ✅ Scene {idx+1}: {format_time(t)}  (render end: {format_time(prev_end)})")
        else:
            print(f"  ⏭️  Scene {idx+1}: {format_time(t)}  → skipped ({prev_end - t:.1f}s until prev clip ends)")

    # Re-sort by original shorts scene order
    selected.sort(key=lambda x: x[0])
    print(f"  → {len(selected)}/{len(pairs)} selected (re-sorted to shorts order)")
    selected_idx = {x[0] for x in selected}
    return [x[1] for x in selected], [x[2] for x in selected], selected_idx

# ── Thumbnail extraction ──────────────────────────────────────────────────────

def extract_thumbnails(movie_file, times, out_dir, prefix, label):
    """Save first frame at each timestamp as JPG, return list of relative paths"""
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    cap = cv2.VideoCapture(movie_file)
    thumbs = []
    for i, t in enumerate(times):
        if t is None:
            thumbs.append(None)
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            fname = f"{prefix}_{label}_{i+1:02d}.jpg"
            cv2.imwrite(os.path.join(img_dir, fname), frame)
            thumbs.append(f"img/{fname}")
        else:
            thumbs.append(None)
    cap.release()
    print(f"  🖼️  {sum(1 for t in thumbs if t)} thumbnails saved → {img_dir}")
    return thumbs

# ── Rendering ─────────────────────────────────────────────────────────────────

def render(label, timestamps, shorts_scenes, movie_clip, output_path, buffer):
    print(f"\n🎞️  [{label}] → {output_path}")
    clips = []
    for i, ((start, end), t) in enumerate(zip(shorts_scenes, timestamps)):
        _check_stop()
        if t is None:
            continue
        t_end = min(t + (end - start) + buffer, movie_clip.duration)
        print(f"  ✂️  Scene {i+1}: {format_time(t)} ~ {format_time(t_end)}")
        clips.append(movie_clip.subclipped(t, t_end))
    if not clips:
        print("  ⚠️  No clips extracted")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = concatenate_videoclips(clips)
    out.write_videofile(output_path, codec="libx264", audio_codec="aac")
    out.close()
    print(f"  ✅ Saved: {output_path}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--shorts',    required=True)
    p.add_argument('-m', '--movie',     nargs='+', required=True)
    p.add_argument('-p', '--prefix',    required=True)
    p.add_argument('-b', '--buffer',    type=float, default=1.0)
    p.add_argument('-t', '--threshold', type=float, default=3.0,
                   help="Cut detection sensitivity (lower = more sensitive, default: 3.0)")
    p.add_argument('-w', '--window',    type=float, default=60.0,
                   help="Visual search window in seconds (default: 60)")
    p.add_argument('--preview', action='store_true',
                   help="Run cut detection only and exit")
    p.add_argument('--monotonic', action='store_true', default=False,
                   help="Enforce chronological order on matched timestamps (prevents duplicate scenes)")
    p.add_argument('--visual-only', action='store_true', default=False,
                   help="Skip audio matching, use full visual scan only (for shorts with BGM)")
    p.add_argument('--min-sim', type=float, default=0.4,
                   help="Minimum visual similarity threshold (default: 0.4, below this is skipped)")
    p.add_argument('--gap', type=float, default=5.0,
                   help="Minimum gap between scenes in seconds for --monotonic (default: 5.0)")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help="Processing device: auto/cuda/cpu (default: auto)")
    args = p.parse_args()

    global DEVICE
    if args.device == 'cpu':
        DEVICE = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️ CUDA not available → switching to CPU")
            DEVICE = torch.device('cpu')
        else:
            DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    shorts_file = args.shorts
    movies      = args.movie

    if not shorts_file.lower().endswith(SUPPORTED_EXTS):
        print(f"❌ Unsupported file extension. Supported: {SUPPORTED_EXTS}"); sys.exit(1)
    if not os.path.exists(shorts_file):
        print("❌ Shorts file not found."); sys.exit(1)

    device_info = f"{DEVICE}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
    print(f"💻 Device: {device_info}")

    prefix = args.prefix.removesuffix('.mp4')

    _set_progress(0.02)
    scenes = get_shorts_scenes(shorts_file, args.threshold)
    if args.preview:
        return
    _set_progress(0.05)

    sw, sh = get_video_size(shorts_file)
    print(f"\n📐 Shorts resolution: {sw}×{sh}")

    _set_progress(0.10)
    if not args.visual_only:
        print("\n📂 Loading shorts audio + DINOv2-Small... (parallel)")
        with ThreadPoolExecutor(max_workers=2) as ex:
            audio_fut = ex.submit(load_audio, shorts_file)
            model_fut = ex.submit(build_feature_extractor)
        shorts_audio = audio_fut.result()
        model        = model_fut.result()
    else:
        print("\n🔧 Loading DINOv2-Small...")
        model = build_feature_extractor()

    scene_feats = prepare_scene_features(scenes, shorts_file, sw, sh, model)
    _set_progress(0.15)

    for mi, movie_file in enumerate(movies):
        if len(movies) > 1:
            print(f"\n{'='*52}")
            print(f"📽️  [{mi+1}/{len(movies)}] {os.path.basename(movie_file)}")
            print(f"{'='*52}")

        if not movie_file.lower().endswith(SUPPORTED_EXTS):
            print(f"❌ Unsupported extension: {movie_file} — skipping"); continue
        if not os.path.exists(movie_file):
            print(f"❌ File not found: {movie_file} — skipping"); continue

        stem    = os.path.splitext(os.path.basename(movie_file))[0]
        out_dir = os.path.join("data/output", f"{prefix}_{stem}")

        # 1. Audio NCC matching
        if args.visual_only:
            print("\n⏭️  Skipping audio matching (--visual-only)")
            n = len(scenes)
            audio_times = [None] * n
            audio_confs = [0.0]  * n
        else:
            print("\n📂 Loading movie audio...")
            movie_audio = load_audio(movie_file)
            audio_times, audio_confs, _ = find_timestamps_by_audio(
                scenes, shorts_audio, movie_audio)

        # 2. Visual DINOv2 matching
        _set_progress(0.15)
        movie_feats, movie_fps, _ = precompute_movie_features(
            movie_file, sw, sh, model,
            progress_cb=lambda p: _set_progress(0.15 + 0.60 * p))
        _set_progress(0.75)
        visual_times = find_timestamps_by_visual(
            scene_feats, movie_feats, movie_fps, audio_times, audio_confs, args.window,
            min_sim=args.min_sim)

        _set_progress(0.80)
        # 3. Final selection: prefer audio match if confident, fallback to visual
        final_times = []
        for at, ac, vt in zip(audio_times, audio_confs, visual_times):
            if not args.visual_only and at is not None and ac >= AUDIO_CONF_THRESHOLD:
                final_times.append(at)
            elif vt is not None:
                final_times.append(vt)
            else:
                final_times.append(None)

        # 4. Monotonic constraint
        if args.monotonic:
            mono_times, mono_scenes, mono_idx = apply_monotonic_constraint(
                final_times, scenes, min_gap=args.gap, buffer=args.buffer)
            # Mask filtered scenes as None for thumbnail consistency
            final_times_for_thumbs = [
                ft if i in mono_idx else None
                for i, ft in enumerate(final_times)
            ]
        else:
            mono_times, mono_scenes = final_times, scenes
            final_times_for_thumbs = final_times

        # Render
        _set_progress(0.84)
        movie_clip = VideoFileClip(movie_file)
        try:
            render("Final", mono_times, mono_scenes, movie_clip,
                   os.path.join(out_dir, f"{prefix}_final.mp4"), args.buffer)
            _set_progress(0.96)
        finally:
            movie_clip.close()

        print("\n🖼️  Extracting thumbnails...")
        shorts_times  = [(s + e) / 2 for s, e in scenes]
        shorts_thumbs = extract_thumbnails(shorts_file, shorts_times,           out_dir, prefix, "shorts")
        final_thumbs  = extract_thumbnails(movie_file,  final_times_for_thumbs, out_dir, prefix, "final")

        _set_progress(0.99)
        generate_report(prefix, shorts_file, out_dir,
                        scenes, audio_times, visual_times, final_times_for_thumbs, args,
                        shorts_thumbs, final_thumbs)
        _set_progress(1.0)

if __name__ == "__main__":
    main()
