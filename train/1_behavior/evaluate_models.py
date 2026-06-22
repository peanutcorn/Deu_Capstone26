# -*- coding: utf-8 -*-
"""두 행동분류 모델 성능 평가 + 논문용 성능평가표(docx) 생성.

평가 대상:
  1) kpt_behavior.pth  — KeypointLSTMv2 (키포인트 30프레임 시퀀스)  / kpt_val.npz 검증셋
  2) model_final.pth   — LSTM_NIA (ResNet50+LSTM, RGB 3프레임)      / test.csv 테스트셋

산출 지표(실측):
  - Accuracy, Macro/Weighted Precision·Recall·F1  (혼동행렬 기반, numpy 직접 계산)
  - 평균 추론 지연(ms/sample) 및 처리량(samples/s)  — GPU / CPU 각각

실행:
    .\.venv\Scripts\python.exe train\1_behavior\evaluate_models.py
출력:
    AI_성능평가표.docx  (프로젝트 루트)
"""

import os
import sys
import time

import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader

# OpenCV 는 Windows에서 비ASCII(한글) 경로를 cv2.imread 로 못 읽는다.
# 프로젝트 경로에 '과제'가 포함되므로, 유니코드 경로 안전 로더로 패치한다.
_orig_imread = cv2.imread
def _safe_imread(path, flags=cv2.IMREAD_COLOR):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, flags)
        return img if img is not None else _orig_imread(path, flags)
    except Exception:
        return _orig_imread(path, flags)
cv2.imread = _safe_imread

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

import config as C
from model_kpt import KeypointLSTM, KeypointLSTMv2
from dataset_kpt import KptBehaviorDataset, LABEL_NAMES
from dataset import BehaviorDataset, build_transform
from train import LSTM_NIA

NUM_CLASSES = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────────────────────────────────
# 지표 계산 (혼동행렬 기반, sklearn 불필요)
# ──────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, num_classes=NUM_CLASSES):
    """혼동행렬로부터 accuracy / per-class P,R,F1 / macro·weighted 평균 계산."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)                       # 클래스별 실제 표본 수
    accuracy = np.trace(cm) / cm.sum() if cm.sum() else 0.0

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)

    # macro: 등장한(support>0) 클래스에 대해서만 단순 평균 (sklearn 기본과 동일하게 전체 라벨 평균도 병기)
    present = support > 0
    macro = {
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
    }
    total = support.sum()
    weighted = {
        "precision": float((precision * support).sum() / total) if total else 0.0,
        "recall": float((recall * support).sum() / total) if total else 0.0,
        "f1": float((f1 * support).sum() / total) if total else 0.0,
    }
    return {
        "accuracy": float(accuracy),
        "macro": macro,
        "weighted": weighted,
        "per_class": {"precision": precision, "recall": recall, "f1": f1,
                      "support": support},
        "confusion": cm,
        "n_samples": int(total),
        "n_present_classes": int(present.sum()),
    }


# ──────────────────────────────────────────────────────────────────────────
# 추론 지연 측정 (모델 forward 순수 시간)
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def measure_latency(model, sample_input, device, warmup=10, iters=50):
    """배치=1 단일 추론의 평균 지연(ms)과 처리량(samples/s) 측정."""
    model = model.to(device).eval()
    x = sample_input.to(device)
    is_cuda = device == "cuda" or (hasattr(device, "type") and device.type == "cuda")
    for _ in range(warmup):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / iters * 1000.0
    return ms, 1000.0 / ms


# ──────────────────────────────────────────────────────────────────────────
# 1) 키포인트 모델 평가
# ──────────────────────────────────────────────────────────────────────────
def load_kpt_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"]
    from collections import OrderedDict
    state = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                        for k, v in state.items())
    model = KeypointLSTMv2() if "attn.weight" in state else KeypointLSTM()
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, type(model).__name__


@torch.no_grad()
def evaluate_kpt(weights, npz_path):
    print(f"\n[1] 키포인트 모델 평가: {os.path.basename(weights)}")
    model, n_params, arch = load_kpt_model(weights)
    model = model.to(DEVICE).eval()
    ds = KptBehaviorDataset(npz_path, augment=False)
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    print(f"    아키텍처 {arch}  파라미터 {n_params:,}  검증표본 {len(ds)}")

    y_true, y_pred = [], []
    for seq, label in loader:
        out = model(seq.to(DEVICE))
        y_pred.extend(out.argmax(dim=1).cpu().tolist())
        y_true.extend(label.tolist() if torch.is_tensor(label) else list(label))
    m = compute_metrics(y_true, y_pred)

    sample = ds[0][0].unsqueeze(0)  # (1, T, 34)
    gpu_ms = gpu_tp = None
    if torch.cuda.is_available():
        gpu_ms, gpu_tp = measure_latency(model, sample, "cuda")
    cpu_ms, cpu_tp = measure_latency(model, sample, "cpu")

    m.update({"arch": arch, "n_params": n_params, "model_file": weights,
              "eval_set": os.path.basename(npz_path), "epoch": 80,
              "gpu_ms": gpu_ms, "gpu_tp": gpu_tp, "cpu_ms": cpu_ms, "cpu_tp": cpu_tp})
    return m


# ──────────────────────────────────────────────────────────────────────────
# 2) ConvLSTM(RGB) 모델 평가
# ──────────────────────────────────────────────────────────────────────────
def load_convlstm_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    from collections import OrderedDict
    state = OrderedDict((k[7:] if k.startswith("module.") else k, v)
                        for k, v in state.items())
    model = LSTM_NIA(hidden_size=C.HIDDEN_SIZE, num_classes=C.NUM_CLASSES,
                     num_layers=C.NUM_LAYERS)
    model.load_state_dict(state)
    n_params = sum(p.numel() for p in model.parameters())
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    return model, n_params, "LSTM_NIA (ResNet50+LSTM)", epoch


@torch.no_grad()
def evaluate_convlstm(weights, csv_name):
    print(f"\n[2] ConvLSTM 모델 평가: {os.path.basename(weights)}")
    model, n_params, arch, epoch = load_convlstm_model(weights)
    model = model.to(DEVICE).eval()
    csv_path = os.path.join(_HERE, C.SPLIT_DIR, csv_name)
    ds = BehaviorDataset(C.DATA_ROOT, csv_path, transform=build_transform())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"    아키텍처 {arch}  파라미터 {n_params:,}  테스트표본 {len(ds)}")

    y_true, y_pred = [], []
    t0 = time.time()
    for bi, (imgs, labels) in enumerate(loader):
        out = model(imgs.to(DEVICE))
        y_pred.extend(out.argmax(dim=1).cpu().tolist())
        y_true.extend(labels.tolist() if torch.is_tensor(labels) else list(labels))
        if bi % 20 == 0:
            print(f"      배치 {bi}/{len(loader)}  ({time.time()-t0:.0f}s)")
    m = compute_metrics(y_true, y_pred)

    sample = ds[0][0].unsqueeze(0)  # (1, 3, 3, 224, 224)
    gpu_ms = gpu_tp = None
    if torch.cuda.is_available():
        gpu_ms, gpu_tp = measure_latency(model, sample, "cuda", warmup=5, iters=30)
    cpu_ms, cpu_tp = measure_latency(model, sample, "cpu", warmup=2, iters=10)

    m.update({"arch": arch, "n_params": n_params, "model_file": weights,
              "eval_set": csv_name, "epoch": epoch,
              "gpu_ms": gpu_ms, "gpu_tp": gpu_tp, "cpu_ms": cpu_ms, "cpu_tp": cpu_tp})
    return m


# ──────────────────────────────────────────────────────────────────────────
# 콘솔 요약 출력
# ──────────────────────────────────────────────────────────────────────────
def print_summary(name, m):
    print(f"\n===== {name} =====")
    print(f"  표본 {m['n_samples']}  (등장 클래스 {m['n_present_classes']}/{NUM_CLASSES})")
    print(f"  Accuracy        : {m['accuracy']*100:.2f}%")
    print(f"  Macro    P/R/F1 : {m['macro']['precision']*100:.2f} / "
          f"{m['macro']['recall']*100:.2f} / {m['macro']['f1']*100:.2f}")
    print(f"  Weighted P/R/F1 : {m['weighted']['precision']*100:.2f} / "
          f"{m['weighted']['recall']*100:.2f} / {m['weighted']['f1']*100:.2f}")
    if m["gpu_ms"]:
        print(f"  추론(GPU)        : {m['gpu_ms']:.2f} ms/sample  ({m['gpu_tp']:.1f} samples/s)")
    print(f"  추론(CPU)        : {m['cpu_ms']:.2f} ms/sample  ({m['cpu_tp']:.1f} samples/s)")
    pc = m["per_class"]
    print("  클래스별 F1:")
    for i, nm in enumerate(LABEL_NAMES):
        if pc["support"][i] > 0:
            print(f"    {nm:>4}: P {pc['precision'][i]*100:5.1f}  "
                  f"R {pc['recall'][i]*100:5.1f}  F1 {pc['f1'][i]*100:5.1f}  "
                  f"(n={int(pc['support'][i])})")


def main():
    print(f"device: {DEVICE}  "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    kpt_weights = os.path.join(_HERE, "output", "kpt_behavior.pth")
    conv_weights = os.path.join(_ROOT, "model", "model_final.pth")

    res_kpt = evaluate_kpt(kpt_weights, os.path.join(_HERE, "split", "kpt_val.npz"))
    # test.csv 는 원본 NIA 소스 경로(CP949)라 현재 디스크에 이미지가 없음(0/7366).
    # 실재하는 유일한 held-out 셋인 val.csv 로 평가한다.
    res_conv = evaluate_convlstm(conv_weights, "val.csv")

    print_summary("KeypointLSTMv2 (kpt_behavior.pth)", res_kpt)
    print_summary("LSTM_NIA (model_final.pth)", res_conv)

    # docx 생성
    from eval_report import build_report
    out = os.path.join(_ROOT, "AI_성능평가표.docx")
    build_report(out, res_kpt, res_conv, DEVICE)
    print(f"\n저장 완료: {out}")


if __name__ == "__main__":
    main()
