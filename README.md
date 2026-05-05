# CCTV 인간 감지 시스템

YOLO11 + DeepSORT + OpenCV 기반 실시간 인간 감지·추적 프로그램.  
포즈 모델 선택 시 COCO 17관절 스켈레톤까지 표시하는 PyQt5 GUI를 제공합니다.

## 주요 기능

- **인간 감지** — YOLO11 n/s/m/l/x 모델, 신뢰도 임계값 실시간 조절
- **다중 객체 추적** — DeepSORT로 각 인원에 고유 트랙 ID 부여
- **포즈 추정** — YOLO11 pose 모델 선택 시 17개 관절 키포인트·스켈레톤 표시
- **영상 소스** — 동영상 파일(mp4/avi/mkv 등) 및 웹캠 입력 지원
- **커스텀 모델** — 직접 학습한 `.pt` 파일 로드 가능

## 기술 스택

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| ultralytics | ≥ 8.0.0 | YOLO11 감지 / 포즈 추정 |
| deep-sort-realtime | ≥ 1.3.2 | 다중 객체 추적 |
| opencv-python | ≥ 4.8.0 | 영상 입출력·프레임 처리 |
| PyQt5 | ≥ 5.15.0 | GUI 프레임워크 |
| torch | ≥ 2.0.0 | YOLO 추론 백엔드 (CUDA 권장) |

## 프로젝트 구조

```
proto-cctv/
├── main.py                  # 진입점
├── run.bat                  # 더블클릭 실행용 배치 파일
├── requirements.txt
├── core/
│   ├── detector.py          # YOLO11 감지기
│   ├── tracker.py           # DeepSORT 트래커 래퍼
│   └── video_thread.py      # QThread 기반 백그라운드 처리
└── gui/
    └── main_window.py       # PyQt5 메인 윈도우
```

## 설치 및 실행

### 1. 가상환경 생성 및 의존성 설치

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> CUDA GPU 환경이라면 PyTorch를 먼저 별도 설치하세요.  
> [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)

### 2. 실행

```powershell
.\.venv\Scripts\python.exe main.py
```

또는 `run.bat` 더블클릭.

## 사용 방법

1. **영상 소스 선택** — `파일 열기` 또는 `웹캠 시작`
2. **모델 선택** — 콤보박스에서 YOLO11 사전학습 모델 선택, 또는 커스텀 `.pt` 직접 지정
3. **포즈 감지** — `yolo11n-pose.pt` 등 pose 모델 선택 → "관절 키포인트" 체크박스 활성화
4. **표시 옵션** — 바운딩 박스 / 트랙 ID / 관절 키포인트 개별 on/off

> 모델 파일은 최초 실행 시 자동으로 다운로드됩니다 (수십 MB).

## 스크린샷

<!-- 스크린샷을 추가하려면 이 아래에 이미지를 삽입하세요 -->

## 라이선스

본 프로젝트는 학술 목적으로 제작되었습니다.
