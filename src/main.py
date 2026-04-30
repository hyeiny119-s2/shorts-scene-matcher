import cv2
import numpy as np
import argparse
import datetime
import os
import sys
import queue
import threading
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from PIL import Image as PILImage
from scenedetect import detect, AdaptiveDetector
from moviepy import VideoFileClip, concatenate_videoclips
from report import generate_report

SUPPORTED_EXTS    = ('.mp4', '.mkv', '.avi', '.mov', '.ts')
FRAME_POSITIONS   = [0.1, 0.25, 0.5, 0.75, 0.9]
CROP_H_POSITIONS  = [0.25, 0.5, 0.75]
VISUAL_FPS_SAMPLE = 3
BATCH_SIZE        = 64
FEAT_DIM          = 384   # DINOv2-Small

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_stop_event      = threading.Event()
_READER_SENTINEL = object()  # pipeline sentinel (None 충돌 방지)
_progress        = 0.0
_report_paths: list = []

class StopProcessing(Exception):
    pass

def _check_stop():
    if _stop_event.is_set():
        raise StopProcessing("사용자가 처리를 중단했습니다.")

def _set_progress(val):
    global _progress
    _progress = min(max(float(val), 0.0), 1.0)

# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def format_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"

def get_video_size(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def get_shorts_scenes(path, threshold=3.0):
    print(f"\n🎬 컷 분석 중... (threshold={threshold})")
    scenes = []
    for i, sc in enumerate(detect(path, AdaptiveDetector(adaptive_threshold=threshold))):
        s, e = sc[0].get_seconds(), sc[1].get_seconds()
        scenes.append((s, e))
        print(f"  - 컷 {i+1}: {format_time(s)} ~ {format_time(e)}")
    print(f"  → 총 {len(scenes)}개 컷")
    return scenes

# ── 비주얼 매칭: DINOv2-Small (GPU) ─────────────────────────────────────────

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
        feats  = self.model(**inputs).last_hidden_state[:, 0, :]
        return F.normalize(feats, dim=1)

def build_feature_extractor():
    return DINOv2Extractor()

@torch.no_grad()
def frames_to_features(frames_rgb, model):
    feats = model.encode_image(frames_rgb)
    return feats.cpu().numpy().astype(np.float32)

def crop_frame(frame, sw, sh, h_pos=0.5):
    """영화 프레임을 숏츠 비율로 크롭 후 224×224로 리사이즈"""
    fh, fw = frame.shape[:2]
    cw = int(fh * sw / sh)
    if cw < fw:
        x = int((fw - cw) * h_pos)
        frame = frame[:, x:x + cw]
    return cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

def prepare_scene_features(scenes, shorts_path, sw, sh, model):
    print("\n  숏츠 레퍼런스 feature 추출 중...")
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
    print(f"\n🖥️  [1/2] 영화 feature 추출 중 ({DEVICE})...")
    cap   = cv2.VideoCapture(movie_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps / VISUAL_FPS_SAMPLE))
    n_pos = len(CROP_H_POSITIONS)

    movie_feats = {pos: {} for pos in CROP_H_POSITIONS}
    batch_q     = queue.Queue(maxsize=4)

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

    processed = 0
    try:
        while True:
            item = batch_q.get()
            if item is _READER_SENTINEL:
                break
            _check_stop()
            fidxs, crops = item
            feats = frames_to_features(crops, model)
            for j, fi in enumerate(fidxs):
                for k, pos in enumerate(CROP_H_POSITIONS):
                    movie_feats[pos][fi] = feats[j * n_pos + k]
            processed += len(fidxs)
            if progress_cb and total > 0:
                progress_cb(processed * step / total)
            pct = processed * step / total * 100 if total > 0 else 0.0
            print(f"  ... {min(processed * step, total)}/{total} ({pct:.1f}%)", end='\r')
    finally:
        while True:
            try:
                batch_q.get_nowait()
            except queue.Empty:
                break

    t.join()
    print(f"  ... {total}/{total} (100.0%)")
    cap.release()
    return movie_feats, fps, total / fps

def find_timestamps_by_visual(scene_feats, movie_feats, fps, min_sim=0.0):
    print(f"\n👁️  [2/2] 비주얼 DINOv2 매칭 중 (전체 스캔, 3방향 크롭, min_sim={min_sim})...")
    all_fidxs = sorted(next(iter(movie_feats.values())).keys())

    results = []
    for i, ref_feats in enumerate(scene_feats):
        _check_stop()
        if len(ref_feats) == 0:
            results.append(None)
            print(f"  - 씬 {i+1}: 레퍼런스 없음")
            continue

        ref_mean = ref_feats.mean(axis=0)
        ref_mean /= (np.linalg.norm(ref_mean) + 1e-8)

        best_t, best_sim = None, -1.0
        for pos, ph_map in movie_feats.items():
            cand = [(fi, ph_map[fi]) for fi in all_fidxs if fi in ph_map]
            if not cand:
                continue
            fidxs_arr = np.array([c[0] for c in cand])
            feats_mat = np.stack([c[1] for c in cand])
            sims      = feats_mat @ ref_mean
            max_i     = int(np.argmax(sims))
            if sims[max_i] > best_sim:
                best_sim = float(sims[max_i])
                best_t   = int(fidxs_arr[max_i]) / fps

        if best_sim < min_sim:
            results.append(None)
            print(f"  - 씬 {i+1}: 스킵 (sim={best_sim:.4f} < {min_sim})")
        else:
            results.append(best_t)
            print(f"  - 씬 {i+1}: {format_time(best_t)}  (sim={best_sim:.4f})")

    return results

# ── 단조 시간순 제약 ──────────────────────────────────────────────────────────

def apply_monotonic_constraint(final_times, scenes, min_gap=5.0, buffer=1.0):
    print(f"\n⏱️  단조 시간순 정렬 및 중복 제거 중... (최소 간격 {min_gap}s)")

    pairs = [(t, i) for i, t in enumerate(final_times) if t is not None]
    pairs.sort(key=lambda x: x[0])

    selected = []
    prev_end = -1e9

    for t, idx in pairs:
        dur = scenes[idx][1] - scenes[idx][0]
        if t >= prev_end + min_gap:
            selected.append((idx, t, scenes[idx]))
            prev_end = t + dur + buffer
            print(f"  ✅ 씬 {idx+1}: {format_time(t)}  (렌더 끝: {format_time(prev_end)})")
        else:
            print(f"  ⏭️  씬 {idx+1}: {format_time(t)}  → 스킵 (이전 렌더 끝까지 {prev_end - t:.1f}s 남음)")

    selected.sort(key=lambda x: x[0])
    print(f"  → {len(selected)}/{len(pairs)}개 선택 (숏츠 순서로 재정렬)")
    selected_idx = {x[0] for x in selected}
    return [x[1] for x in selected], [x[2] for x in selected], selected_idx

# ── 개별 클립 저장 ────────────────────────────────────────────────────────────

def export_clips(timestamps, shorts_scenes, movie_path, out_dir, buffer):
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)
    print(f"\n📂 개별 클립 저장 중 → {clips_dir}")
    for i, ((start, end), t) in enumerate(zip(shorts_scenes, timestamps)):
        _check_stop()
        if t is None:
            continue
        path = os.path.join(clips_dir, f"clip_{i+1:02d}.mp4")
        print(f"  ✂️  clip_{i+1:02d}: {format_time(t)} ~ {format_time(t + (end - start) + buffer)}")
        mc = VideoFileClip(movie_path)
        try:
            t_end = min(t + (end - start) + buffer, mc.duration)
            mc.subclipped(t, t_end).write_videofile(path, codec="libx264", audio_codec="aac")
        finally:
            mc.close()
    print(f"  ✅ 클립 저장 완료 → {clips_dir}")

# ── 썸네일 추출 ───────────────────────────────────────────────────────────────

def extract_thumbnails(video_file, times, out_dir, prefix, label):
    img_dir = os.path.join(out_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_file)
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
    print(f"  🖼️  썸네일 {sum(1 for t in thumbs if t)}개 저장 → {img_dir}")
    return thumbs

# ── 렌더링 ────────────────────────────────────────────────────────────────────

def render(label, timestamps, shorts_scenes, movie_clip, output_path, buffer):
    print(f"\n🎞️  [{label}] → {output_path}")
    clips = []
    for i, ((start, end), t) in enumerate(zip(shorts_scenes, timestamps)):
        _check_stop()
        if t is None:
            continue
        t_end = min(t + (end - start) + buffer, movie_clip.duration)
        print(f"  ✂️  씬 {i+1}: {format_time(t)} ~ {format_time(t_end)}")
        clips.append(movie_clip.subclipped(t, t_end))
    if not clips:
        print("  ⚠️  추출된 클립 없음")
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = concatenate_videoclips(clips)
    out.write_videofile(output_path, codec="libx264", audio_codec="aac")
    out.close()
    print(f"  ✅ 저장: {output_path}")

# ── 진입점 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-s', '--shorts',    required=True)
    p.add_argument('-m', '--movie',     nargs='+', required=True)
    p.add_argument('-b', '--buffer',    type=float, default=1.0)
    p.add_argument('-t', '--threshold', type=float, default=3.0,
                   help="컷 감지 민감도 (낮을수록 민감, 기본값: 3.0)")
    p.add_argument('--preview', action='store_true',
                   help="컷 감지만 확인하고 종료")
    p.add_argument('--monotonic', action='store_true', default=False,
                   help="매칭 타임스탬프를 영화 시간순으로 강제 정렬 (반복 장면 방지)")
    p.add_argument('--min-sim', type=float, default=0.4,
                   help="비주얼 최소 유사도 threshold (기본값: 0.4, 미만이면 스킵)")
    p.add_argument('--gap', type=float, default=5.0,
                   help="--monotonic 최소 씬 간격 초 (기본값: 5.0)")
    p.add_argument('--export-clips', action='store_true', default=False,
                   help="컷을 clips/ 폴더에 개별 파일로도 저장")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'],
                   help="처리 장치: auto/cuda/cpu (기본값: auto)")
    args = p.parse_args()

    global DEVICE
    if args.device == 'cpu':
        DEVICE = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠️ CUDA 없음 → CPU로 전환")
            DEVICE = torch.device('cpu')
        else:
            DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _report_paths.clear()

    shorts_file = args.shorts
    movies      = args.movie

    if not shorts_file.lower().endswith(SUPPORTED_EXTS):
        print(f"❌ 지원하지 않는 확장자. 지원: {SUPPORTED_EXTS}"); sys.exit(1)
    if not os.path.exists(shorts_file):
        print("❌ 숏츠 파일을 찾을 수 없습니다."); sys.exit(1)

    device_info = f"{DEVICE}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
    print(f"💻 디바이스: {device_info}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    _set_progress(0.02)
    scenes = get_shorts_scenes(shorts_file, args.threshold)
    if args.preview:
        return
    _set_progress(0.05)

    sw, sh = get_video_size(shorts_file)
    print(f"\n📐 숏츠 해상도: {sw}×{sh}")

    _set_progress(0.10)
    print("\n🔧 DINOv2-Small 로딩 중...")
    model = build_feature_extractor()

    scene_feats = prepare_scene_features(scenes, shorts_file, sw, sh, model)
    _set_progress(0.15)

    for mi, movie_file in enumerate(movies):
        if len(movies) > 1:
            print(f"\n{'='*52}")
            print(f"📽️  [{mi+1}/{len(movies)}] {os.path.basename(movie_file)}")
            print(f"{'='*52}")

        if not movie_file.lower().endswith(SUPPORTED_EXTS):
            print(f"❌ 지원하지 않는 확장자: {movie_file} — 건너뜁니다"); continue
        if not os.path.exists(movie_file):
            print(f"❌ 파일 없음: {movie_file} — 건너뜁니다"); continue

        stem    = os.path.splitext(os.path.basename(movie_file))[0]
        out_dir = os.path.join("data", f"{stem}_{timestamp}")

        # 1. 비주얼 DINOv2 매칭
        _set_progress(0.15)
        movie_feats, movie_fps, _ = precompute_movie_features(
            movie_file, sw, sh, model,
            progress_cb=lambda p: _set_progress(0.15 + 0.65 * p))
        _set_progress(0.80)
        final_times = find_timestamps_by_visual(
            scene_feats, movie_feats, movie_fps,
            min_sim=args.min_sim)

        # 2. 단조 시간순 제약
        _set_progress(0.84)
        if args.monotonic:
            mono_times, mono_scenes, mono_idx = apply_monotonic_constraint(
                final_times, scenes, min_gap=args.gap, buffer=args.buffer)
            final_times_for_thumbs = [
                ft if i in mono_idx else None
                for i, ft in enumerate(final_times)
            ]
        else:
            mono_times, mono_scenes = final_times, scenes
            final_times_for_thumbs = final_times

        # 3. 렌더링
        movie_clip = VideoFileClip(movie_file)
        try:
            render("Final", mono_times, mono_scenes, movie_clip,
                   os.path.join(out_dir, f"{stem}_final.mp4"), args.buffer)
            if args.export_clips:
                export_clips(mono_times, mono_scenes, movie_file, out_dir, args.buffer)
            _set_progress(0.96)
        finally:
            movie_clip.close()

        print("\n🖼️  썸네일 추출 중...")
        shorts_times  = [(s + e) / 2 for s, e in scenes]
        shorts_thumbs = extract_thumbnails(shorts_file, shorts_times,            out_dir, stem, "shorts")
        final_thumbs  = extract_thumbnails(movie_file,  final_times_for_thumbs,  out_dir, stem, "final")

        _set_progress(0.99)
        generate_report(stem, shorts_file, out_dir,
                        scenes, final_times_for_thumbs, args,
                        shorts_thumbs, final_thumbs)
        _report_paths.append(os.path.join(out_dir, f"{stem}_report.html"))
        _set_progress(1.0)

if __name__ == "__main__":
    main()
