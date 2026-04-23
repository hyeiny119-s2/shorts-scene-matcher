# Shorts Auto Editor

숏츠 영상을 넣으면, 원본 영화에서 해당 장면들을 자동으로 찾아서 잘라주는 프로그램입니다.

DINOv2-Base 비주얼 매칭 + 오디오 NCC 매칭을 조합해 정확한 타임스탬프를 찾습니다.

---

## 동작 방식

1. 숏츠의 컷을 자동으로 분석
2. 각 컷을 오디오 NCC + DINOv2 비주얼 유사도로 원본 영화에서 탐색
3. 매칭된 구간을 잘라서 합친 영상 + 리포트 HTML 생성

---

## 실행 방법

### Docker (권장)

```bash
# 이미지 빌드 (최초 1회)
docker build -t shorts-auto-editor-image .

# 실행
docker run --rm --gpus all \
  -v /path/to/data:/app/data \
  -v ~/.cache:/root/.cache \
  shorts-auto-editor-image \
  -s data/input/shorts.mp4 \
  -m data/input/movie.mp4 \
  -p output_name \
  --visual-only --monotonic
```

> HuggingFace 모델 캐시 재사용을 위해 `-v ~/.cache:/root/.cache` 마운트를 권장합니다.

### Python 직접 실행

```bash
pip install -r requirements.txt
pip install -r requirements_gui.txt  # GUI 사용 시

# CLI
python main.py -s shorts.mp4 -m movie.mp4 -p output_name

# GUI
python gui.py
```

---

## 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `-s` | 숏츠 파일 경로 | 필수 |
| `-m` | 원본 영화 파일 (여러 개 가능) | 필수 |
| `-p` | 출력 폴더명 prefix | 필수 |
| `-b` | 클립 버퍼 (초) | `1.0` |
| `-t` | 컷 감지 민감도 (낮을수록 민감) | `3.0` |
| `-w` | 비주얼 탐색 윈도우 (초) | `60` |
| `--visual-only` | 오디오 매칭 생략 (BGM 있는 숏츠용) | off |
| `--monotonic` | 시간순 정렬 + 중복 제거 | off |
| `--min-sim` | 비주얼 최소 유사도 (미만이면 스킵) | `0.4` |
| `--gap` | `--monotonic` 최소 씬 간격 (초) | `5.0` |
| `--device` | `auto` / `cuda` / `cpu` | `auto` |

---

## 출력

결과는 `data/output/{prefix}_{영화파일명}/` 에 저장됩니다.

| 파일 | 내용 |
|------|------|
| `{prefix}_final.mp4` | 최종 결과 영상 |
| `{prefix}_report.html` | 씬별 매칭 결과 리포트 (브라우저에서 열기) |
| `img/` | 씬별 썸네일 이미지 |

---

## 지원 포맷

`.mp4` `.mkv` `.avi` `.mov` `.ts`
