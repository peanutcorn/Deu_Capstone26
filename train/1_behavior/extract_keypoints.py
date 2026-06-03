"""NIA 이미지 시퀀스에서 YOLO pose 키포인트를 추출해 학습 데이터를 생성한다.

입력:
    split/train.csv  (기존 소스 CSV 그대로 사용 가능)
    형식: frame1_path,frame2_path,frame3_path,label

출력:
    split/kpt_train.npz  — {'sequences': (N, T, 34), 'labels': (N,)}
    split/kpt_val.npz

처리 흐름:
    1. CSV 읽기 → 연속 동일 레이블 클립을 하나의 에피소드로 병합
    2. 각 프레임에 YOLO pose 실행 → (17, 3) keypoints
    3. bbox 상대 정규화 → (34,) 벡터
    4. 슬라이딩 윈도우 (SEQ_LEN, STRIDE) 로 샘플 생성
    5. .npz 저장

사용법:
    python extract_keypoints.py --data-root C:/data/NIA --model yolo11n-pose.pt
    python extract_keypoints.py --data-root C:/data/NIA --model yolo11n-pose.pt --split val
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.normpath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _PROJECT)

SEQ_LEN = 30    # 슬라이딩 윈도우 길이 (프레임 수)
STRIDE = 10     # 슬라이딩 윈도우 이동 간격
KPT_CONF_THR = 0.3  # 이 값 미만 관절 → (0, 0)
EPISODE_GAP = 3     # 연속 3클립 이상의 공백이 있으면 에피소드 분리
MIN_VALID_FRAC = 0.5  # 윈도우 내 인물 검출(비-제로) 프레임 비율 최소값


def encode_kpts(kpts_xy, bbox):
    """(17, 2) float + bbox [x1,y1,x2,y2] → (34,) 정규화 벡터."""
    x1, y1, x2, y2 = bbox
    bw, bh = max(float(x2 - x1), 1), max(float(y2 - y1), 1)
    vec = np.zeros(34, dtype=np.float32)
    for i, (kx, ky) in enumerate(kpts_xy):
        vec[i * 2]     = (kx - x1) / bw
        vec[i * 2 + 1] = (ky - y1) / bh
    return vec


def run_yolo_on_frame(model, img_bgr):
    """YOLO pose 추론 → 가장 넓은 bbox 의 (keypoints_xy, bbox) 반환. 없으면 None."""
    results = model(img_bgr, verbose=False)
    best_area, best = -1, None
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                kpts = r.keypoints[i].data[0].cpu().numpy()   # (17, 3)
                # 신뢰도 미달 관절 (0, 0) 처리
                kpts_xy = np.array([
                    [k[0], k[1]] if k[2] >= KPT_CONF_THR else [0.0, 0.0]
                    for k in kpts
                ], dtype=np.float32)
                best_area = area
                best = (kpts_xy, [x1, y1, x2, y2])
    return best


def load_csv(csv_path, data_root):
    """CSV → (full_paths_per_clip, label) 리스트."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.reader(f):
            row = [r.strip() for r in row]
            if len(row) < 4:
                continue
            paths = []
            for p in row[:3]:
                p = p.replace("\\", os.sep)
                fp = p if os.path.isabs(p) else os.path.join(data_root, p)
                paths.append(fp)
            label = int(row[3])
            if os.path.exists(paths[0]):
                rows.append((paths, label))
    return rows


def build_episodes(rows):
    """연속 동일 레이블 클립을 하나의 에피소드로 묶는다."""
    if not rows:
        return []
    episodes = []
    cur_label = rows[0][1]
    cur_frames = list(rows[0][0])

    for paths, label in rows[1:]:
        if label == cur_label:
            cur_frames.extend(paths)
        else:
            if cur_frames:
                episodes.append((cur_frames, cur_label))
            cur_label = label
            cur_frames = list(paths)

    if cur_frames:
        episodes.append((cur_frames, cur_label))
    return episodes


def extract(data_root, csv_path, yolo_model, output_path,
            seq_len=SEQ_LEN, stride=STRIDE):
    from ultralytics import YOLO
    model = YOLO(yolo_model)

    rows = load_csv(csv_path, data_root)
    episodes = build_episodes(rows)

    sequences, labels = [], []
    processed, skipped = 0, 0

    for frame_paths, label in episodes:
        vecs = []
        for fp in frame_paths:
            img = cv2.imread(fp)
            if img is None:
                vecs.append(np.zeros(34, dtype=np.float32))
                continue
            result = run_yolo_on_frame(model, img)
            if result is None:
                vecs.append(np.zeros(34, dtype=np.float32))
            else:
                kpts_xy, bbox = result
                vecs.append(encode_kpts(kpts_xy, bbox))

        # 슬라이딩 윈도우 (인물 검출률 낮은 윈도우는 폐기)
        for start in range(0, len(vecs) - seq_len + 1, stride):
            seq = np.stack(vecs[start:start + seq_len])
            valid = np.mean([not np.all(f == 0) for f in seq])
            if valid < MIN_VALID_FRAC:
                skipped += 1
                continue
            sequences.append(seq)
            labels.append(label)
            processed += 1

        if len(vecs) < seq_len:
            skipped += 1

    print(f"\n총 {processed}개 샘플 생성 ({skipped}개 에피소드 길이 부족으로 스킵)")
    if not sequences:
        print("샘플이 없습니다. 데이터 경로 및 CSV를 확인하세요.")
        return

    np.savez_compressed(
        output_path,
        sequences=np.stack(sequences).astype(np.float32),
        labels=np.array(labels, dtype=np.int64),
    )
    print(f"저장: {output_path}")
    print(f"  sequences shape: {np.stack(sequences).shape}")

    # 클래스 분포
    label_names = ["정상", "전도", "파손", "방화", "흡연", "유기", "절도", "폭행"]
    from collections import Counter
    cnt = Counter(labels)
    for i, n in enumerate(label_names):
        print(f"  {n}: {cnt.get(i, 0)}")


def main():
    _default_data_root = os.path.join(_HERE, "split", "data")
    parser = argparse.ArgumentParser(description="NIA 키포인트 추출")
    parser.add_argument("--data-root", default=_default_data_root)
    parser.add_argument("--split-dir", default="split")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--stride", type=int, default=STRIDE)
    args = parser.parse_args()

    split_dir = args.split_dir if os.path.isabs(args.split_dir) \
        else os.path.join(_HERE, args.split_dir)

    csv_path = os.path.join(split_dir, f"{args.split}.csv")
    output_path = os.path.join(split_dir, f"kpt_{args.split}.npz")

    assert os.path.isfile(csv_path), f"CSV 없음: {csv_path}"
    print(f"CSV: {csv_path}")
    print(f"YOLO 모델: {args.model}")

    extract(args.data_root, csv_path, args.model, output_path,
            seq_len=args.seq_len, stride=args.stride)


if __name__ == "__main__":
    main()
