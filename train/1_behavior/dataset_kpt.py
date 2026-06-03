"""키포인트 시퀀스 데이터셋 (extract_keypoints.py 출력 .npz 로드)."""

import numpy as np
import torch
from torch.utils.data import Dataset

LABEL_NAMES = ["정상", "전도", "파손", "방화", "흡연", "유기", "절도", "폭행"]


class KptBehaviorDataset(Dataset):
    """extract_keypoints.py 가 생성한 .npz 파일을 로드한다.

    .npz 구조:
        sequences : (N, T, 34)  float32
        labels    : (N,)        int64
    """

    def __init__(self, npz_path: str, augment: bool = False):
        data = np.load(npz_path)
        self.sequences = data["sequences"].astype(np.float32)  # (N, T, 34)
        self.labels = data["labels"].astype(np.int64)          # (N,)
        self.augment = augment
        assert len(self.sequences) == len(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].copy()  # (T, 34)

        if self.augment:
            # 좌우 반전: 홀수 인덱스(x) 를 1에서 뺌, 짝수(y) 는 그대로
            if np.random.rand() < 0.5:
                seq[:, 0::2] = 1.0 - seq[:, 0::2]
                # 좌우 대칭 관절 쌍 교환 (COCO 기준)
                # 0↔0(코), 1↔2(눈), 3↔4(귀), 5↔6(어깨), 7↔8(팔꿈치),
                # 9↔10(손목), 11↔12(골반), 13↔14(무릎), 15↔16(발목)
                _FLIP_PAIRS = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
                for l, r in _FLIP_PAIRS:
                    li, ri = l*2, r*2
                    seq[:, [li, li+1, ri, ri+1]] = seq[:, [ri, ri+1, li, li+1]]

            # 가우시안 노이즈
            noise = np.random.randn(*seq.shape).astype(np.float32) * 0.01
            seq = np.clip(seq + noise, 0.0, 1.0)

        return torch.from_numpy(seq), int(self.labels[idx])

    def class_weights(self) -> torch.Tensor:
        counts = np.bincount(self.labels, minlength=len(LABEL_NAMES))
        total = counts.sum()
        weights = [(1.0 - c / total) if total > 0 else 1.0 for c in counts]
        return torch.FloatTensor(weights)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python dataset_kpt.py split/kpt_train.npz")
        sys.exit(1)
    ds = KptBehaviorDataset(sys.argv[1], augment=True)
    print(f"샘플 수: {len(ds)}")
    seq, lbl = ds[0]
    print(f"seq shape: {seq.shape}, label: {lbl} ({LABEL_NAMES[lbl]})")
    print(f"클래스 가중치: {ds.class_weights().tolist()}")
