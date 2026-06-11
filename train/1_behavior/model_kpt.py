"""키포인트 행동 분류 모델 정의 (학습용).

core/kpt_behavior_classifier.py 의 KeypointLSTM 과 동일 구조.
학습 스크립트에서 직접 import 하도록 독립 파일로 분리.

저장 포맷: plain state_dict() → torch.save(model.state_dict(), 'kpt_behavior.pth')
앱 로드 : core/kpt_behavior_classifier.py 의 TrackBehaviorBuffer._load_weights()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointLSTM(nn.Module):
    """v1 — 구버전 체크포인트 호환용 (신규 학습은 KeypointLSTMv2 사용).

    입력 : (B, T, 34)  — T 프레임, 17관절 × (x, y) bbox 상대 정규화 좌표
    출력 : (B, num_classes)
    """

    def __init__(self, input_size: int = 34, hidden: int = 128,
                 num_layers: int = 2, num_classes: int = 8, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


class KeypointLSTMv2(nn.Module):
    """v2 — 행동 인식 정확도 개선 버전.

    v1 대비 개선점:
      1. 속도 특징 추가 — 프레임 간 좌표 차분을 입력에 결합 (34 → 68차원).
         전도·폭행처럼 "움직임"이 본질인 행동에 정적 좌표만으로는 부족.
      2. BiLSTM — 30프레임 윈도우 전체를 양방향으로 인코딩.
      3. attention pooling — 마지막 hidden 대신 중요 프레임 가중 평균.
         이상행동이 윈도우 어느 구간에 있어도 포착 가능 (약지도 라벨 완화).

    입력 : (B, T, 34)  — v1과 동일 (npz 재추출 불필요)
    출력 : (B, num_classes)
    """

    def __init__(self, input_size: int = 34, hidden: int = 128,
                 num_layers: int = 2, num_classes: int = 8, dropout: float = 0.5):
        super().__init__()
        feat = input_size * 2  # 위치 + 속도
        self.norm = nn.LayerNorm(feat)
        self.lstm = nn.LSTM(
            input_size=feat,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 속도 특징 — 미검출(0,0) 관절은 차분이 튀므로 양쪽 프레임 모두 유효할 때만 사용
        nz = (x != 0).float()
        vel = (x[:, 1:] - x[:, :-1]) * nz[:, 1:] * nz[:, :-1]
        vel = F.pad(vel, (0, 0, 1, 0))            # 첫 프레임 속도 = 0
        f = self.norm(torch.cat([x, vel], dim=2))  # (B, T, 68)
        out, _ = self.lstm(f)                      # (B, T, 2H)
        w = torch.softmax(self.attn(out), dim=1)   # (B, T, 1) 프레임 중요도
        return self.head((w * out).sum(dim=1))


if __name__ == "__main__":
    for cls in (KeypointLSTM, KeypointLSTMv2):
        model = cls()
        x = torch.rand(4, 30, 34)
        n = sum(p.numel() for p in model.parameters())
        print(f"{cls.__name__}: input {x.shape} → output {model(x).shape}, 파라미터 {n:,}")
