import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from core.detector import HumanDetector
from core.tracker import HumanTracker

_COLOR_PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]

# COCO 17 관절 연결 쌍 (skeleton)
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),    # 얼굴
    (5, 6),                              # 양 어깨
    (5, 7), (7, 9),                      # 왼쪽 팔
    (6, 8), (8, 10),                     # 오른쪽 팔
    (5, 11), (6, 12),                    # 몸통 측면
    (11, 12),                            # 골반
    (11, 13), (13, 15),                  # 왼쪽 다리
    (12, 14), (14, 16),                  # 오른쪽 다리
]

_KPT_CONF_THRESHOLD = 0.3


def _track_color(track_id) -> tuple:
    return _COLOR_PALETTE[int(track_id) % len(_COLOR_PALETTE)]


def _iou(a: list, b: list) -> float:
    """두 bbox [x1,y1,x2,y2] 의 IoU."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _match_keypoints(tracks: list, detections: list) -> dict:
    """track_id → keypoints(np.ndarray|None) 매핑. IoU 최대 detection 선택."""
    result = {}
    for t in tracks:
        best_iou, best_kpts = 0.0, None
        for d in detections:
            iou = _iou(t["bbox"], d["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_kpts = d.get("keypoints")
        # IoU 0.1 미만이면 박스가 거의 겹치지 않으므로 잘못된 매핑으로 간주
        result[t["track_id"]] = best_kpts if best_iou > 0.1 else None
    return result


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)
    stats_updated = pyqtSignal(int, float)
    error_occurred = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, source, model_path: str, conf: float, max_age: int):
        super().__init__()
        self.source = source
        self.model_path = model_path
        self.conf = conf
        self.max_age = max_age
        self._running = False
        self._paused = False
        self.show_bbox = True
        self.show_track_id = True
        self.show_keypoints = True

    def run(self):
        self._running = True
        cap = None
        try:
            detector = HumanDetector(self.model_path, self.conf)
            tracker = HumanTracker(self.max_age)
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.error_occurred.emit(f"영상 소스를 열 수 없습니다: {self.source}")
                return

            src_fps = cap.get(cv2.CAP_PROP_FPS)
            # 30 fps 상한: 처리 속도가 원본 fps를 초과하면 Qt 이벤트 큐가 과부하됨
            target_interval = 1.0 / min(src_fps if src_fps > 0 else 30.0, 30.0)

            prev_time = time.time()
            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue

                loop_start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                detector.set_confidence(self.conf)
                detections = detector.detect(frame)
                tracks = tracker.update(detections, frame)

                kpt_map = {}
                if self.show_keypoints and detector.is_pose:
                    kpt_map = _match_keypoints(tracks, detections)

                self._draw(frame, tracks, kpt_map)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                self.stats_updated.emit(len(tracks), fps)
                self.frame_ready.emit(self._to_qimage(frame))

                elapsed = time.time() - loop_start
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)

        except Exception:
            import traceback
            self.error_occurred.emit(traceback.format_exc())
        finally:
            if cap is not None:
                cap.release()
            self.finished_signal.emit()

    # ------------------------------------------------------------------ 그리기
    def _draw(self, frame: np.ndarray, tracks: list, kpt_map: dict):
        for t in tracks:
            tid = t["track_id"]
            color = _track_color(tid)
            x1, y1, x2, y2 = t["bbox"]

            if self.show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if self.show_track_id:
                    label = f"ID:{tid}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            kpts = kpt_map.get(tid)
            if kpts is not None:
                self._draw_pose(frame, kpts, color)

    def _draw_pose(self, frame: np.ndarray, kpts: np.ndarray, color: tuple):
        """17개 관절 점과 skeleton 연결선을 그린다."""
        h, w = frame.shape[:2]
        points = []
        for kx, ky, kc in kpts:
            if kc >= _KPT_CONF_THRESHOLD and 0 <= int(kx) < w and 0 <= int(ky) < h:
                points.append((int(kx), int(ky)))
            else:
                points.append(None)

        # 연결선
        for a, b in _SKELETON:
            if points[a] and points[b]:
                cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)

        # 관절 점
        for pt in points:
            if pt:
                cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------ 유틸
    @staticmethod
    def _to_qimage(frame: np.ndarray) -> QImage:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        # .copy() 필수: 없으면 QImage가 numpy 버퍼 포인터를 보유한 채 GC 발생 → dangling pointer crash (exit code 5)
        return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

    def set_conf(self, value: float):
        self.conf = value

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def stop(self):
        self._running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)
