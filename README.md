# Shorts Scene Matcher

Automatically finds and cuts matching scenes from a full movie based on a reference short-form video, using CLIP visual features and audio NCC matching.

---

## 어떤 프로그램인가요?

숏츠(짧은 영상)를 레퍼런스로 넣으면, 원본 풀영상에서 해당 장면들을 자동으로 찾아 잘라주는 프로그램입니다.

- CLIP ViT-B/32 기반 시각적 매칭
- 오디오 NCC 매칭 (BGM 있는 숏츠는 Visual Only 옵션 사용)
- GPU 자동 감지 (없으면 CPU로 자동 전환)
- 결과 HTML 리포트 + 썸네일 생성

---

## 설치 (Windows)

### 1. 이 레포지토리 다운로드

오른쪽 상단 **Code → Download ZIP** → 압축 풀기

### 2. setup.bat 실행

폴더 안의 `setup.bat` 을 더블클릭합니다.

- Python 자동 설치
- GPU 감지 후 CUDA / CPU 버전 PyTorch 설치
- CLIP 모델 다운로드 (~350MB)
- ffmpeg 설치
- 바탕화면 바로가기 생성

> 인터넷 연결 필요 / 약 15~20분 소요 (최초 1회)

### 3. 실행

바탕화면의 **Shorts Auto Editor** 바로가기 더블클릭
(바로가기가 없으면 폴더 안 `run.bat` 더블클릭)

---

## 사용 방법 (GUI)

1. **숏츠** 파일을 왼쪽에 드래그하거나 클릭해서 선택
2. **풀영상** 파일을 오른쪽에 드래그하거나 클릭해서 선택
3. 출력 이름(prefix) 입력
4. 옵션 선택
   - **Visual Only** — 숏츠에 BGM이 있을 때 체크
   - **시간순 정렬** — 중복 클립 방지 (권장)
5. **▶ 실행** 클릭
6. 완료 후 **📁 출력 폴더 열기** 로 결과 확인

결과물은 `data/output/{출력이름}/` 폴더에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `{prefix}_visual.mp4` | 비주얼 매칭 결과 영상 |
| `{prefix}_final.mp4` | 최종 결과 영상 |
| `{prefix}_report.html` | 매칭 결과 리포트 (썸네일 포함) |

---

## CLI 사용법 (고급)

```bash
python main.py \
  -s data/input/shorts.mp4 \
  -m data/input/movie.mp4 \
  -p output_name \
  --visual-only \
  --monotonic
```

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-s` | 숏츠 파일 경로 | 필수 |
| `-m` | 풀영상 파일 경로 | 필수 |
| `-p` | 출력 prefix | 필수 |
| `-b` | 클립 버퍼 (초) | 1.0 |
| `-t` | 컷 감지 민감도 | 3.0 |
| `-w` | 비주얼 검색 윈도우 (초) | 60.0 |
| `--visual-only` | 오디오 매칭 생략 (BGM 있는 숏츠) | off |
| `--monotonic` | 시간순 정렬 + 중복 제거 | off |
| `--min-sim` | 최소 유사도 | 0.4 |
| `--gap` | monotonic 최소 간격 (초) | 5.0 |
| `--device` | auto / cuda / cpu | auto |

---

## Docker (Linux/GPU 서버)

```bash
docker build -t shorts-auto-editor-image .
docker run --gpus all --rm \
  -v $(pwd)/data:/app/data \
  shorts-auto-editor-image \
  python main.py -s data/input/shorts.mp4 -m data/input/movie.mp4 -p result --visual-only --monotonic
```

---

## 처리 시간 (2시간 영화 기준)

| 환경 | 소요 시간 |
|------|-----------|
| NVIDIA GPU | 5 ~ 10분 |
| CPU only | 30 ~ 60분 |
