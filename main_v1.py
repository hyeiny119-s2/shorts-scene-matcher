import cv2
import numpy as np
import librosa
import argparse
import os
import sys
import subprocess
import tempfile
import torch
import torch.nn.functional as F
import torchvision.models as models
from scenedetect import detect, AdaptiveDetector
from moviepy import VideoFileClip, concatenate_videoclips
from report import generate_report

SUPPORTED_EXTS       = ('.mp4', '.mkv', '.avi', '.mov', '.ts')
AUDIO_SR             = 11025
AUDIO_CONF_THRESHOLD = 5.0           # 이 이하면 오디오 불확실 → 비주얼 윈도우 2배
FRAME_POSITIONS      = [0.1, 0.25, 0.5, 0.75, 0.9]
CROP_H_POSITIONS     = [0.25, 0.5, 0.75]   # 좌/중/우 크롭 (auto-framing 대응)
VISUAL_FPS_SAMPLE    = 3             # 영화 샘플링 fps
BATCH_SIZE           = 64            # GPU forward pass 배치 크기
FEAT_DIM             = 2048          # ResNet50 출력 차원

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet 정규화 상수 (GPU 상주)
_IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(3, 1, 1)
_IMG_STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(3, 1, 1)

# ── 공통 유틸 ────────────────────────────────────────────────────────────────

def format_time(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"

def get_video_size(path):
    cap = cv2.VideoCapture(path)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def get_shorts_scenes(path, threshold=3.0):
    print(f"🎬 컷 분석 중... (threshold={threshold})")
    scenes = []
    for i, sc in enumerate(detect(path, AdaptiveDetector(adaptive_threshold=threshold))):
        s, e = sc[0].get_seconds(), sc[1].get_seconds()
        scenes.append((s, e))
        print(f"  - 컷 {i+1}: {format_time(s)} ~ {format_time(e)}")
    print(f"  → 총 {len(scenes)}개 컷")
    return scenes

# ── 오디오 로딩 ──────────────────────────────────────────────────────────────

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

# ── 오디오 매칭: NCC ─────────────────────────────────────────────────────────

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

def ncc_match(movie_audio, scene_audio):
    peaks = ncc_topk(movie_audio, scene_audio, k=1)
    if not peaks:
        return None, 0.0
    return peaks[0]

def find_timestamps_by_audio(shorts_scenes, shorts_audio, movie_audio):
    print("\n🎵 [1/3] 오디오 NCC 매칭 중...")
    times, confs, all_candidates = [], [], []
    for i, (start, end) in enumerate(shorts_scenes):
        scene = shorts_audio[int(start * AUDIO_SR):int(end * AUDIO_SR)]
        if len(scene) < AUDIO_SR // 2:
            print(f"  - 씬 {i+1}: 너무 짧아 스킵")
            times.append(None); confs.append(0.0); all_candidates.append([])
            continue
        candidates = ncc_topk(movie_audio, scene, k=8)
        t, conf = candidates[0] if candidates else (None, 0.0)
        warn = "  ⚠️ 낮은 신뢰도" if conf < AUDIO_CONF_THRESHOLD else ""
        print(f"  - 씬 {i+1}: {format_time(t)}  (신뢰도 {conf:.1f}){warn}")
        times.append(t); confs.append(conf); all_candidates.append(candidates)
    return times, confs, all_candidates

# ── 비주얼 매칭: ResNet50 CNN Features (GPU) ─────────────────────────────────
#
# pHash(64비트) 대비 ResNet50(2048-dim L2-normalized):
#   → 시맨틱 장면 이해 수준의 feature → 구별력 압도적 향상
#   → GPU 배치 처리로 2시간 영화도 수십 초 내 완료
#   → 코사인 유사도 search: (M, 2048) @ (2048, R) → 즉시 결과
#
# 좌/중/우 크롭 유지:
#   → auto-framing 숏츠 (피사체가 중앙이 아닐 때) 대응

def build_feature_extractor():
    """ResNet50 마지막 FC 제거 → 2048-dim avgpool feature"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    return model.eval().to(DEVICE)

@torch.no_grad()
def frames_to_features(frames_bgr, model):
    """
    list of BGR numpy arrays → (N, FEAT_DIM) float32 L2-normalized features
    GPU 배치 forward pass
    """
    tensors = []
    for f in frames_bgr:
        rgb  = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t    = torch.from_numpy(
            cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        ).permute(2, 0, 1).to(DEVICE)
        tensors.append((t - _IMG_MEAN) / _IMG_STD)

    batch = torch.stack(tensors)                         # (N, 3, 224, 224)
    feats = model(batch).squeeze(-1).squeeze(-1)         # (N, 2048)
    feats = F.normalize(feats, dim=1)
    return feats.cpu().numpy().astype(np.float32)

def crop_frame(frame, sw, sh, h_pos=0.5):
    """영화 프레임을 숏츠 비율로 크롭 (resize 없이, features 추출 전에 수행)"""
    fh, fw = frame.shape[:2]
    cw = int(fh * sw / sh)
    if cw < fw:
        x = int((fw - cw) * h_pos)
        frame = frame[:, x:x + cw]
    return frame

def prepare_scene_features(scenes, shorts_path, sw, sh, model):
    """숏츠 씬별 레퍼런스 프레임 feature 추출 (GPU)"""
    print("  숏츠 레퍼런스 feature 추출 중...")
    cap = cv2.VideoCapture(shorts_path)
    scene_feats = []
    for start, end in scenes:
        frames = []
        for pos in FRAME_POSITIONS:
            t = start + (end - start) * pos
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, f = cap.read()
            if ret:
                frames.append(f)   # 숏츠는 이미 세로 비율 → 그대로 사용
        feats = frames_to_features(frames, model) if frames else np.zeros((0, FEAT_DIM), np.float32)
        scene_feats.append(feats)
    cap.release()
    return scene_feats

def precompute_movie_features(movie_path, sw, sh, model):
    """
    영화 전체 1회 순차 스캔 → GPU 배치로 feature 추출
    {crop_pos: {frame_idx: feature_vector (2048,)}}
    """
    print(f"\n🖥️  [2/3] 영화 CNN feature 추출 중 ({DEVICE})...")
    cap   = cv2.VideoCapture(movie_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, int(fps / VISUAL_FPS_SAMPLE))

    movie_feats    = {pos: {} for pos in CROP_H_POSITIONS}
    pending_frames = {pos: [] for pos in CROP_H_POSITIONS}
    pending_fidxs  = []
    fidx           = 0

    def flush():
        for pos in CROP_H_POSITIONS:
            feats = frames_to_features(pending_frames[pos], model)
            for j, fi in enumerate(pending_fidxs):
                movie_feats[pos][fi] = feats[j]
            pending_frames[pos].clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if fidx % step == 0:
            for pos in CROP_H_POSITIONS:
                pending_frames[pos].append(crop_frame(frame, sw, sh, pos))
            pending_fidxs.append(fidx)
            if len(pending_fidxs) >= BATCH_SIZE:
                flush()
                pending_fidxs.clear()
        if fidx % (step * 300) == 0:
            print(f"  ... {fidx}/{total} ({fidx/total*100:.1f}%)", end='\r')
        fidx += 1

    if pending_fidxs:
        flush()
    print(f"  ... {total}/{total} (100.0%)")
    cap.release()
    movie_duration = total / fps
    return movie_feats, fps, movie_duration

def find_timestamps_by_visual(scene_feats, movie_feats, fps, audio_times, audio_confs, search_window):
    """
    ResNet50 코사인 유사도 기반 최적 프레임 탐색
    오디오 신뢰도 낮은 씬은 search_window 2배 적용
    """
    print(f"\n👁️  [3/3] 비주얼 CNN 매칭 중 (±{search_window}s, 3방향 크롭)...")
    all_fidxs = sorted(next(iter(movie_feats.values())).keys())

    results = []
    for i, (ref_feats, at, ac) in enumerate(zip(scene_feats, audio_times, audio_confs)):
        if len(ref_feats) == 0:
            results.append(None)
            print(f"  - 씬 {i+1}: 레퍼런스 없음")
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
            print(f"  - 씬 {i+1}: 후보 없음")
            continue

        # 씬 레퍼런스 feature 평균 → 하나의 쿼리 벡터
        ref_mean = ref_feats.mean(axis=0)
        ref_mean /= (np.linalg.norm(ref_mean) + 1e-8)  # (2048,)

        best_t, best_sim = None, -1.0
        for pos, ph_map in movie_feats.items():
            cand = [(fi, ph_map[fi]) for fi in cand_fidxs if fi in ph_map]
            if not cand:
                continue
            fidxs_arr = np.array([c[0] for c in cand])
            feats_mat = np.stack([c[1] for c in cand])   # (M, 2048)
            sims      = feats_mat @ ref_mean              # (M,) cosine sim
            max_i     = int(np.argmax(sims))
            if sims[max_i] > best_sim:
                best_sim = float(sims[max_i])
                best_t   = int(fidxs_arr[max_i]) / fps

        results.append(best_t)
        scope = f"±{win:.0f}s" if at is not None else "전체"
        print(f"  - 씬 {i+1}: {format_time(best_t)}  (sim={best_sim:.4f}, {scope})")

    return results

# ── 단조 시간순 제약 ──────────────────────────────────────────────────────────


def apply_monotonic_constraint(final_times, scenes, min_gap=5.0):
    """
    Sort matched timestamps chronologically and remove overlapping duplicates.
    Returns (sorted_times, sorted_scenes) — ready to pass directly to render().
    min_gap: minimum seconds between start of successive clips.
    """
    print(f"\n⏱️  단조 시간순 정렬 및 중복 제거 중... (최소 간격 {min_gap}s)")

    # Collect valid matches with original scene index
    pairs = [(t, i) for i, t in enumerate(final_times) if t is not None]
    pairs.sort(key=lambda x: x[0])  # sort by movie timestamp

    selected_times  = []
    selected_scenes = []
    prev_end        = -1e9

    for t, idx in pairs:
        dur = scenes[idx][1] - scenes[idx][0]
        if t >= prev_end + min_gap:
            selected_times.append(t)
            selected_scenes.append(scenes[idx])
            prev_end = t + dur
            print(f"  ✅ 씬 {idx+1}: {format_time(t)}")
        else:
            print(f"  ⏭️  씬 {idx+1}: {format_time(t)}  → 스킵 (이전 씬과 {t - (prev_end - dur):.1f}s 차이)")

    print(f"  → {len(selected_times)}/{len(pairs)}개 선택")
    return selected_times, selected_scenes

# ── 렌더링 ────────────────────────────────────────────────────────────────────

def render(label, timestamps, shorts_scenes, movie_clip, output_path, buffer):
    print(f"\n🎞️  [{label}] → {output_path}")
    clips = []
    for i, ((start, end), t) in enumerate(zip(shorts_scenes, timestamps)):
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
    p.add_argument('-m', '--movie',     required=True)
    p.add_argument('-p', '--prefix',    required=True)
    p.add_argument('-b', '--buffer',    type=float, default=1.0)
    p.add_argument('-t', '--threshold', type=float, default=3.0,
                   help="컷 감지 민감도 (낮을수록 민감, 기본값: 3.0)")
    p.add_argument('-w', '--window',    type=float, default=60.0,
                   help="비주얼 검색 윈도우 초 (기본값: 60)")
    p.add_argument('--preview', action='store_true',
                   help="컷 감지만 확인하고 종료")
    p.add_argument('--monotonic', action='store_true', default=False,
                   help="매칭 타임스탬프를 영화 시간순으로 강제 정렬 (반복 장면 방지)")
    p.add_argument('--visual-only', action='store_true', default=False,
                   help="오디오 매칭 생략, 비주얼 전체 스캔만 사용 (BGM 있는 숏츠용)")
    args = p.parse_args()

    shorts_file, movie_file = args.shorts, args.movie
    if not shorts_file.lower().endswith(SUPPORTED_EXTS) or \
       not movie_file.lower().endswith(SUPPORTED_EXTS):
        print(f"❌ 지원하지 않는 확장자. 지원: {SUPPORTED_EXTS}"); sys.exit(1)
    if not os.path.exists(shorts_file) or not os.path.exists(movie_file):
        print("❌ 파일을 찾을 수 없습니다."); sys.exit(1)

    device_info = f"{DEVICE}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
    print(f"💻 디바이스: {device_info}")

    prefix  = args.prefix.removesuffix('.mp4')
    out_dir = os.path.join("data/output", prefix)

    scenes = get_shorts_scenes(shorts_file, args.threshold)
    if args.preview:
        return

    sw, sh = get_video_size(shorts_file)
    print(f"📐 숏츠 해상도: {sw}×{sh}")

    # 1. 오디오 NCC (BGM 있는 숏츠는 --visual-only로 생략)
    if args.visual_only:
        print("\n⏭️  오디오 매칭 생략 (--visual-only)")
        n = len(scenes)
        audio_times     = [None] * n
        audio_confs     = [0.0]  * n
        audio_candidates = [[]]  * n
    else:
        print("\n📂 오디오 로딩 중...")
        shorts_audio = load_audio(shorts_file)
        movie_audio  = load_audio(movie_file)
        audio_times, audio_confs, audio_candidates = find_timestamps_by_audio(
            scenes, shorts_audio, movie_audio)

    # 2. ResNet50 CNN 비주얼 매칭 (GPU 배치)
    #    --visual-only 시 audio_times 전부 None → 전체 스캔
    print("\n🔧 ResNet50 로딩 중...")
    model       = build_feature_extractor()
    scene_feats = prepare_scene_features(scenes, shorts_file, sw, sh, model)
    movie_feats, movie_fps, movie_duration = precompute_movie_features(movie_file, sw, sh, model)
    visual_times = find_timestamps_by_visual(
        scene_feats, movie_feats, movie_fps, audio_times, audio_confs, args.window)

    # 3. Final: 오디오 신뢰도 기반 선택
    #    --visual-only 시 무조건 비주얼 사용
    final_times = []
    for at, ac, vt in zip(audio_times, audio_confs, visual_times):
        if not args.visual_only and at is not None and ac >= AUDIO_CONF_THRESHOLD:
            final_times.append(at)
        elif vt is not None:
            final_times.append(vt)
        else:
            final_times.append(at)

    # 4. 단조 시간순 제약 (옵션): 정렬+중복제거 후 별도 scenes 리스트 생성
    if args.monotonic:
        mono_times, mono_scenes = apply_monotonic_constraint(final_times, scenes)
    else:
        mono_times, mono_scenes = final_times, scenes

    # 렌더링
    movie_clip = VideoFileClip(movie_file)
    try:
        render("비주얼 CNN",  visual_times, scenes,      movie_clip,
               os.path.join(out_dir, f"{prefix}_visual.mp4"), args.buffer)
        render("Final",       mono_times,   mono_scenes, movie_clip,
               os.path.join(out_dir, f"{prefix}_final.mp4"),  args.buffer)
    finally:
        movie_clip.close()

    generate_report(prefix, shorts_file, out_dir,
                    scenes, audio_times, visual_times, final_times, args)

if __name__ == "__main__":
    main()
