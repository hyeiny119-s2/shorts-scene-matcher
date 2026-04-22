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

폴더 안의 `setup.bat` 을 더블클릭하고 완료될 때까지 기다립니다.

> 인터넷 연결 필요 / 약 15~20분 소요 (최초 1회)

### 3. 실행

바탕화면 또는 폴더 안의 **Shorts Auto Editor** 바로가기를 더블클릭합니다.
(바로가기가 없으면 폴더 안 `run.bat` 더블클릭)

---

## 사용 방법 (GUI)

1. **숏츠** 파일을 왼쪽에 드래그하거나 클릭해서 선택
2. **풀영상** 파일을 오른쪽에 드래그하거나 클릭해서 선택
3. **저장 폴더명** 입력 (결과가 저장될 폴더 이름)
4. 옵션 선택
   - **Visual Only** — 숏츠에 BGM이 있을 때 체크
   - **시간순 정렬** — 중복 클립 방지 (권장)
5. **▶ 실행** 클릭 → 진행률 바로 진행 상황 확인
6. 완료 후 **📁 출력 폴더 열기** 로 결과 확인
7. 처리를 멈추려면 **■ 중단** 클릭
8. 새로 시작하려면 **↺ 초기화** 클릭

결과물은 `data/output/{출력이름}/` 폴더에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `{prefix}_visual.mp4` | 비주얼 매칭 결과 영상 |
| `{prefix}_final.mp4` | 최종 결과 영상 |
| `{prefix}_report.html` | 매칭 결과 리포트 (썸네일 포함) |

---

## 처리 시간 (2시간 영화 기준)

| 환경 | 소요 시간 |
|------|-----------|
| NVIDIA GPU | 5 ~ 10분 |
| CPU only | 30 ~ 60분 |
