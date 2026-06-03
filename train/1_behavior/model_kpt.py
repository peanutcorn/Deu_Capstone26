"""키포인트 행동 분류 모델 정의 (학습용).

core/kpt_behavior_classifier.py 의 KeypointLSTM 과 동일 구조.
학습 스크립트에서 직접 import 하도록 독립 파일로 분리.

저장 포맷: plain state_dict() → torch.save(model.state_dict(), 'kpt_behavior.pth')
앱 로드 : core/kpt_behavior_classifier.py 의 TrackBehaviorBuffer._load_weights()
"""

import torch
import torch.nn as nn


class KeypointLSTM(nn.Module):
    """
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


if __name__ == "__main__":
    model = KeypointLSTM()
    x = torch.rand(4, 30, 34)
    print(f"input {x.shape} → output {model(x).shape}")
    n = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n:,}")
