# CCTV 이상 현상 감지 시스템

YOLO11 + DeepSORT + OpenCV 기반 실시간 **이상행동 감지** 프로그램.
인물을 감지·추적하면서 17관절 키포인트 시퀀스를 KeypointLSTM(`kpt_behavior.pth`)에 입력해
**인물(트랙 ID)별로** 이상행동을 분류한다. 단일 "이상행동 감지" 모드로 통합되어
바운딩 박스·트랙 ID·키포인트·인물별 행동 라벨·감지 인원·FPS가 한 화면에서 동시에 표시된다.

## 파이프라인

```
프레임 → YOLO pose (yolo11n/s/m-pose) → bbox + 17관절
       → DeepSORT (n_init=2)           → track_id
       → KeypointLSTM (kpt_behavior.pth) → 인물별 (행동, 신뢰도)
       → 박스·키포인트·라벨 그리기 + 인원수·FPS·대표 이상행동 표시
```

`kpt_behavior.pth` 를 지정하지 않아도 감지·추적·키포인트 표시는 정상 동작한다(행동 라벨만 생략).

## 주요 기능

| 기능 | 설명 |
|---|---|
| **인간 감지 + 포즈** | YOLO11 pose(n/s/m) 모델로 인물 bbox + COCO 17관절 스켈레톤 |
| **다중 객체 추적** | DeepSORT로 각 인원에 고유 트랙 ID + 색상 부여 |
| **인물별 이상행동 분류** | KeypointLSTM — 트랙 ID별 8종 행동 분류, 이상행동 시 빨간 박스 + `"ID:1 \| ASSAULT 92%"` 라벨 |
| **실시간 통계** | 감지 인원 수, 처리 FPS, 화면 대표 이상행동 |
| **IP 카메라** | RTSP 스트림 연결 + 끊김 시 자동 재연결(최대 5회) |
| **경량화 옵션** | 감지 모델 크기(n/s/m) · 처리 해상도(384/480/640) · CUDA FP16 자동 |

### 이상행동 8종 클래스
`정상 / 전도(쓰러짐) / 파손 / 방화 / 흡연 / 유기 / 절도 / 폭행`

## 기술 스택

| 라이브러리 | 버전 | 용도 |
|---|---|---|
| ultralytics | ≥ 8.0.0 | YOLO11 포즈 추정 |
| deep-sort-realtime | ≥ 1.3.2 | 다중 객체 추적 |
| opencv-python | ≥ 4.8.0 | 영상 입출력·프레임 처리 |
| PyQt5 | ≥ 5.15.0 | GUI 프레임워크 |
| torch + torchvision | ≥ 2.0.0 | 추론 백엔드 (CUDA 권장) |

## 프로젝트 구조

```
proto-cctv/
├── main.py                         # 진입점 (Qt 플러그인 경로 설정)
├── run.bat                         # 더블클릭 실행
├── requirements.txt
├── core/
│   ├── detector.py                 # YOLO11 포즈 감지기 (imgsz·FP16 경량화)
│   ├── tracker.py                  # DeepSORT 트래커 (n_init=2, GPU 임베더)
│   ├── kpt_behavior_classifier.py  # KeypointLSTM 인물별 행동 분류 + 트랙 버퍼
│   ├── labels.py                   # 이상행동 8종 라벨 (한글/영문)
│   └── video_thread.py             # QThread 백그라운드 처리 (통합 파이프라인)
├── gui/
│   └── main_window.py              # PyQt5 메인 윈도우
├── model/                          # 사전학습 가중치 (참조용)
└── train/
    └── 1_behavior/                 # KeypointLSTM 학습 (재추출·재학습 스크립트)
```

## 설치 및 실행

### 1. 가상환경 및 의존성 설치

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

> CUDA GPU 환경이라면 PyTorch를 먼저 별도 설치하세요.
> https://pytorch.org/get-started/locally

### 2. 실행

```powershell
.\.venv\Scripts\python.exe main.py
```

또는 `run.bat` 더블클릭.

## 사용 방법

1. **모델 설정**에서 감지 모델(`yolo11n-pose` 권장), 이상행동 분류 모델(`kpt_behavior.pth` 경로,
   기본 자동 입력), 처리 해상도(경량 PC는 384/480 권장)를 확인한다.
2. **파일 열기** / **웹캠 시작** / **RTSP 연결** 중 하나로 영상 소스를 시작한다.
3. 화면에 인물 박스 · 트랙 ID · 키포인트 · `"ID:n | 행동 %"` 라벨이 표시되고,
   우측 패널에 감지 인원 · FPS · 대표 이상행동이 갱신된다. 이상행동 인물은 박스가 빨간색으로 바뀐다.
4. **표시 옵션** 체크박스로 바운딩 박스 / 트랙 ID / 키포인트 / 행동 라벨을 실시간 on/off 할 수 있다.

### 경량 PC 최적화 팁
- 감지 모델 `yolo11n-pose`, 처리 해상도 `384` 조합이 가장 가볍다.
- CUDA GPU에서는 FP16(half) 추론과 GPU 임베더가 자동 적용된다.

## 학습 (재훈련)

KeypointLSTM 인물별 행동 분류 모델(`kpt_behavior.pth`) 재학습 절차.
`train/1_behavior/split/data/` 하위 클래스 폴더(`07.전도` ~ `13.폭행`)의 mp4 영상과
CSV 이미지(`split/train.csv`, `split/val.csv`)를 사용한다.

```powershell
cd train\1_behavior

# 0) 기존 추출 결과 삭제 (extract_kpts_video 는 npz에 병합하므로 필수)
Remove-Item split\kpt_train.npz, split\kpt_val.npz -ErrorAction SilentlyContinue

# 1) CSV 이미지에서 키포인트 추출 (정상 클래스 위주)
python extract_keypoints.py --split train
python extract_keypoints.py --split val

# 2) 클래스별 mp4 영상에서 추출 후 npz 병합
#    - 영상 "파일 단위" 8:2 train/val 분할 (윈도우 누수 제거)
#    - 인물 검출률 50% 미만 윈도우는 자동 폐기 (품질 필터)
python extract_kpts_video.py

# 3) 학습 (약 60 에포크) → best 체크포인트를 kpt_behavior.pth 로 자동 배포
python train_kpt.py
```

## 라이선스

본 프로젝트는 학술 목적으로 제작되었습니다.
