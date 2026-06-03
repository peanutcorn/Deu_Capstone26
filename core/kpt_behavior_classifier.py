"""키포인트 기반 이상행동 분류 — KeypointLSTM + 트랙별 슬라이딩 윈도우 버퍼.

YOLO pose 가 추출한 17관절 좌표 시퀀스를 입력으로 받아
인물별로 이상행동을 분류한다.

파이프라인:
    YOLO pose → (17, 3) keypoints per person
    → _encode : bbox 상대 정규화 → (34,) 벡터
    → 30프레임 deque (track_id별)
    → KeypointLSTM → (class_id, confidence)

학습된 가중치 포맷: plain state_dict() (module. 접두사 없음)
    torch.save(model.state_dict(), 'kpt_behavior.pth')
"""

from collections import deque, OrderedDict

import numpy as np
import torch
import torch.nn as nn

_KPT_CONF_THR = 0.3   # 이 값 미만 관절은 (0, 0) 처리


class KeypointLSTM(nn.Module):
    """경량 키포인트 시퀀스 → 행동 분류기.

    입력 : (B, T, 34)  — T 프레임, 17관절 × (x, y) 정규화 좌표
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
        return self.head(h[-1])  # 마지막 레이어의 hidden state


def _encode(kpts, bbox, conf_thr: float = _KPT_CONF_THR) -> np.ndarray:
    """(17, 3) keypoints → (34,) bbox-상대 정규화 벡터.

    x' = (kx - x1) / bbox_w,  y' = (ky - y1) / bbox_h
    신뢰도 미달 관절은 (0, 0).
    """
    x1, y1, x2, y2 = bbox
    bw, bh = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
    vec = np.zeros(34, dtype=np.float32)
    if kpts is not None:
        for i, (kx, ky, kc) in enumerate(kpts):
            if float(kc) >= conf_thr:
                vec[i * 2]     = (float(kx) - x1) / bw
                vec[i * 2 + 1] = (float(ky) - y1) / bh
    return vec


class TrackBehaviorBuffer:
    """트랙 ID별 키포인트 시퀀스를 관리하고 행동 추론을 실행한다.

    매 프레임 update(tracks, kpt_map) 를 호출하면
    {track_id: (class_id, confidence)} 딕셔너리를 반환한다.
    버퍼가 아직 seq_len 을 채우지 못한 트랙은 결과 없음.
    """

    def __init__(self, model_path: str, seq_len: int = 30, stride: int = 10,
                 device=None):
        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model = KeypointLSTM()
        self._load_weights(model_path)
        self.model.to(self.device).eval()

        # track_id → deque[(34,)]
        self._bufs: dict[int, deque] = {}
        # track_id → frame 카운터
        self._cnts: dict[int, int] = {}
        # track_id → (class_id, conf)
        self._results: dict[int, tuple] = {}

    def _load_weights(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        # plain state_dict 또는 wrapper dict 둘 다 처리
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        # DataParallel 키 접두사 제거
        state = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state.items()
        )
        self.model.load_state_dict(state)

    def update(self, tracks: list, kpt_map: dict) -> dict:
        """tracks, kpt_map 을 받아 결과 딕셔너리를 반환·갱신한다."""
        active_ids = set()
        for t in tracks:
            tid = t["track_id"]
            active_ids.add(tid)
            vec = _encode(kpt_map.get(tid), t["bbox"])

            if tid not in self._bufs:
                self._bufs[tid] = deque(maxlen=self.seq_len)
                self._cnts[tid] = 0

            self._bufs[tid].append(vec)
            self._cnts[tid] += 1

            # 버퍼가 충분히 쌓이고 stride 조건을 만족하면 추론
            if (len(self._bufs[tid]) == self.seq_len
                    and self._cnts[tid] % self.stride == 0):
                self._results[tid] = self._infer(self._bufs[tid])

        # 사라진 트랙 정리
        for tid in list(self._bufs):
            if tid not in active_ids:
                self._bufs.pop(tid)
                self._cnts.pop(tid)
                self._results.pop(tid, None)

        return dict(self._results)

    @torch.no_grad()
    def _infer(self, buf: deque) -> tuple:
        seq = torch.from_numpy(np.stack(list(buf))).unsqueeze(0).to(self.device)
        prob = torch.softmax(self.model(seq), dim=1)[0]
        conf, cls = torch.max(prob, 0)
        return int(cls.item()), float(conf.item())
