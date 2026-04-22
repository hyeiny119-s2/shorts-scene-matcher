"""
오디오 매칭 진단 스크립트
씬 1의 오디오가 원본 영상 어디에 있는지 raw cross-correlation으로 찾습니다.
"""
import numpy as np
import subprocess, tempfile, os, sys
import librosa
from scenedetect import detect, AdaptiveDetector

SHORTS = "data/input/thatsummer/Thatsummer_shorts.mp4"
MOVIE  = "data/input/thatsummer/Thatsummer_full.mp4"
SR     = 11025

def load_audio(path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        tmp = f.name
    subprocess.run(['ffmpeg','-y','-i',path,'-ac','1','-ar',str(SR),'-vn',tmp],
                   check=True, capture_output=True)
    audio, _ = librosa.load(tmp, sr=SR, mono=True)
    os.unlink(tmp)
    return audio

def fmt(s):
    return f"{int(s//3600):02d}:{int((s%3600)//60):02d}:{s%60:05.2f}"

print("=" * 60)
print("1. 숏츠 씬 감지")
scenes = detect(SHORTS, AdaptiveDetector(adaptive_threshold=3.0))
print(f"   감지된 씬 수: {len(scenes)}")
for i, sc in enumerate(scenes):
    print(f"   씬 {i+1}: {fmt(sc[0].get_seconds())} ~ {fmt(sc[1].get_seconds())}")

if not scenes:
    print("   ❌ 씬이 감지되지 않았습니다. threshold를 낮춰보세요.")
    sys.exit(1)

print("\n2. 오디오 로딩")
print(f"   숏츠 로딩 중...")
shorts_audio = load_audio(SHORTS)
print(f"   숏츠 길이: {len(shorts_audio)/SR:.1f}초")
print(f"   원본 로딩 중...")
movie_audio  = load_audio(MOVIE)
print(f"   원본 길이: {len(movie_audio)/SR:.1f}초")

def ncc(movie, scene):
    """로컬 RMS 정규화 cross-correlation — 에너지 편향 제거"""
    N = len(scene)
    valid_len = len(movie) - N + 1
    if valid_len <= 0:
        return None
    scene_n = scene / (np.linalg.norm(scene) + 1e-8)
    n_fft = 2 ** int(np.ceil(np.log2(len(movie) + N - 1)))
    corr = np.fft.irfft(
        np.fft.rfft(movie, n_fft) * np.conj(np.fft.rfft(scene_n, n_fft))
    )[:valid_len]
    # 로컬 RMS로 나눔
    cumsum = np.concatenate([[0], np.cumsum(movie ** 2)])
    local_rms = np.sqrt(np.maximum((cumsum[N:] - cumsum[:valid_len]) / N, 1e-8))
    return corr / local_rms

print("\n3. 전체 씬 NCC(로컬 정규화) 매칭 결과")
print(f"   {'씬':>3}  {'숏츠 구간':>22}  {'원본 매칭 1위':>14}  {'2위':>14}  {'3위':>14}")
print("   " + "-" * 75)

matched = []
for i, sc in enumerate(scenes):
    s = sc[0].get_seconds()
    e = sc[1].get_seconds()
    seg = shorts_audio[int(s*SR):int(e*SR)]
    if len(seg) < SR // 2:
        matched.append(None)
        print(f"   {i+1:>3}  {fmt(s)} ~ {fmt(e)}  (너무 짧아 스킵)")
        continue

    corr = ncc(movie_audio, seg)
    if corr is None:
        matched.append(None)
        continue

    top3 = np.argsort(corr)[::-1][:3]
    best_t = top3[0] / SR
    matched.append(best_t)
    tops = "  ".join(fmt(t/SR) for t in top3)
    print(f"   {i+1:>3}  {fmt(s)} ~ {fmt(e)}  →  {tops}")

print("\n4. 단조 증가 여부 확인 (씬 순서대로 시간이 증가해야 정상)")
prev = -1
for i, t in enumerate(matched):
    if t is None:
        continue
    ok = "✅" if t > prev else "❌ 역순!"
    print(f"   씬 {i+1:>2}: {fmt(t)}  {ok}")
    prev = t

print("=" * 60)
