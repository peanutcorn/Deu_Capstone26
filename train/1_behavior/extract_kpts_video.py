"""mp4 영상에서 직접 키포인트를 추출해 기존 kpt_*.npz에 병합한다.

split/data/ 하위 폴더 구조:
    07.전도  → label 1
    08.파손  → label 2
    09.방화  → label 3
    10.흡연  → label 4
    11.유기  → label 5
    12.절도  → label 6
    13.폭행  → label 7
    14.교통약자 → 스킵 (8클래스 모델에 없음)

사용법:
    python extract_kpts_video.py
    python extract_kpts_video.py --train-ratio 0.8 --model yolo11n-pose.pt
"""

import argparse
import os
import sys
import random
import cv2
import numpy as np
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))

SEQ_LEN = 30
STRIDE  = 10
KPT_CONF_THR = 0.3
MIN_VALID_FRAC = 0.5   # 윈도우 내 인물 검출(비-제로) 프레임 비율이 이 값 미만이면 폐기
SPLIT_SEED = 42        # 영상 단위 train/val 분할 재현성

FOLDER_LABEL_MAP = {
    "07": 1,
    "08": 2,
    "09": 3,
    "10": 4,
    "11": 5,
    "12": 6,
    "13": 7,
}
LABEL_NAMES = ["정상", "전도", "파손", "방화", "흡연", "유기", "절도", "폭행"]


def encode_kpts(kpts_xy, bbox):
    x1, y1, x2, y2 = bbox
    bw = max(float(x2 - x1), 1.0)
    bh = max(float(y2 - y1), 1.0)
    vec = np.zeros(34, dtype=np.float32)
    for i, (kx, ky) in enumerate(kpts_xy):
        vec[i * 2]     = (kx - x1) / bw
        vec[i * 2 + 1] = (ky - y1) / bh
    return vec


def run_yolo_on_frame(model, img_bgr):
    results = model(img_bgr, verbose=False)
    best_area, best = -1, None
    for r in results:
        if r.boxes is None or r.keypoints is None:
            continue
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                kpts = r.keypoints[i].data[0].cpu().numpy()
                kpts_xy = np.array(
                    [[k[0], k[1]] if k[2] >= KPT_CONF_THR else [0.0, 0.0]
                     for k in kpts],
                    dtype=np.float32,
                )
                best_area = area
                best = (kpts_xy, [x1, y1, x2, y2])
    return best


def extract_from_video(video_path, label, yolo_model, seq_len, stride):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [오류] 열기 실패: {video_path}")
        return [], []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"  프레임 수: {total}  FPS: {fps:.1f}")

    vecs = []
    idx  = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = run_yolo_on_frame(yolo_model, frame)
        vecs.append(encode_kpts(*result) if result else np.zeros(34, dtype=np.float32))
        idx += 1
        if idx % 200 == 0:
            print(f"    {idx}/{total} 프레임 처리 중...", end="\r", flush=True)
    cap.release()
    print(f"    {idx} 프레임 처리 완료            ")

    seqs, labels = [], []
    kept, dropped = 0, 0
    for start in range(0, len(vecs) - seq_len + 1, stride):
        window = np.stack(vecs[start : start + seq_len])
        # 인물이 검출된(비-제로) 프레임 비율 — 너무 낮으면 라벨이 무의미하므로 폐기
        valid = np.mean([not np.all(f == 0) for f in window])
        if valid < MIN_VALID_FRAC:
            dropped += 1
            continue
        seqs.append(window)
        labels.append(label)
        kept += 1
    if dropped:
        print(f"    품질 필터: {kept}개 유지 / {dropped}개 폐기 (검출률<{MIN_VALID_FRAC:.0%})")
    return seqs, labels


def merge_npz(path, new_seqs, new_labels):
    if len(new_seqs) == 0:
        if os.path.exists(path):
            d = np.load(path)
            return len(d["sequences"])
        return 0

    new_arr = np.stack(new_seqs).astype(np.float32)
    new_lbl = np.array(new_labels, dtype=np.int64)

    if os.path.exists(path):
        d = np.load(path)
        merged_arr = np.concatenate([d["sequences"], new_arr], axis=0)
        merged_lbl = np.concatenate([d["labels"],    new_lbl], axis=0)
    else:
        merged_arr, merged_lbl = new_arr, new_lbl

    np.savez_compressed(path, sequences=merged_arr, labels=merged_lbl)
    return len(merged_arr)


def main():
    default_data_dir  = os.path.join(_HERE, "split", "data")
    default_split_dir = os.path.join(_HERE, "split")

    parser = argparse.ArgumentParser(description="mp4 영상 키포인트 추출 및 npz 병합")
    parser.add_argument("--data-dir",    default=default_data_dir)
    parser.add_argument("--split-dir",   default=default_split_dir)
    parser.add_argument("--model",       default="yolo11n-pose.pt")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seq-len",     type=int,   default=SEQ_LEN)
    parser.add_argument("--stride",      type=int,   default=STRIDE)
    args = parser.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(args.model)

    train_seqs, train_labels = [], []
    val_seqs,   val_labels   = [], []

    data_dir  = args.data_dir  if os.path.isabs(args.data_dir)  else os.path.join(_HERE, args.data_dir)
    split_dir = args.split_dir if os.path.isabs(args.split_dir) else os.path.join(_HERE, args.split_dir)

    for folder in sorted(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        prefix = folder.split(".")[0]
        if prefix not in FOLDER_LABEL_MAP:
            print(f"스킵: {folder}")
            continue
        label      = FOLDER_LABEL_MAP[prefix]
        label_name = LABEL_NAMES[label]

        mp4s = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".mp4"))
        if not mp4s:
            print(f"스킵: {folder} (mp4 없음)")
            continue

        # 영상 파일 단위로 train/val 분할 (윈도우 단위 분할의 train/val 누수 제거)
        rng = random.Random(SPLIT_SEED + label)
        rng.shuffle(mp4s)
        n_train_vid = max(1, int(len(mp4s) * args.train_ratio))
        val_files = set(mp4s[n_train_vid:])

        print(f"\n[{folder}] → label {label} ({label_name}), "
              f"영상 {len(mp4s)}개 (train {n_train_vid} / val {len(mp4s)-n_train_vid})")
        for mp4_name in mp4s:
            video_path = os.path.join(folder_path, mp4_name)
            split = "val" if mp4_name in val_files else "train"
            print(f"  [{split}] 영상: {mp4_name}")
            seqs, labels = extract_from_video(video_path, label, yolo,
                                               seq_len=args.seq_len, stride=args.stride)
            if not seqs:
                print("  시퀀스 생성 없음 (영상 너무 짧거나 사람 미감지)")
                continue
            print(f"  시퀀스 {len(seqs)}개 생성")

            if split == "val":
                val_seqs.extend(seqs);   val_labels.extend(labels)
            else:
                train_seqs.extend(seqs); train_labels.extend(labels)

    if not train_seqs:
        print("\n추출된 시퀀스 없음. 데이터 디렉터리와 영상 파일을 확인하세요.")
        return

    train_npz = os.path.join(split_dir, "kpt_train.npz")
    val_npz   = os.path.join(split_dir, "kpt_val.npz")

    total_train = merge_npz(train_npz, train_seqs, train_labels)
    total_val   = merge_npz(val_npz,   val_seqs,   val_labels)

    print(f"\n저장 완료: train {total_train}개 / val {total_val}개")
    print("\n클래스 분포:")
    for split_name, npz_path in [("train", train_npz), ("val", val_npz)]:
        lbl = np.load(npz_path)["labels"]
        c   = Counter(lbl.tolist())
        dist = "  ".join(f"{LABEL_NAMES[i]}:{c.get(i, 0)}" for i in range(8) if c.get(i, 0) > 0)
        print(f"  {split_name}: {dist}")


if __name__ == "__main__":
    main()
