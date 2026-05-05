from ultralytics import YOLO
import numpy as np


class HumanDetector:
    """YOLO11 기반 인간 감지기. 포즈 모델이면 키포인트도 함께 반환."""

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # 포즈 모델 여부 — task 속성으로 판별
        self.is_pose = getattr(self.model, "task", None) == "pose"

    def set_confidence(self, value: float):
        self.conf_threshold = value

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Returns list of dicts:
          { 'bbox': [x1,y1,x2,y2], 'conf': float, 'keypoints': np.ndarray | None }
        keypoints shape: (17, 3) — (x, y, conf) per joint. None for non-pose models.
        """
        kwargs = dict(conf=self.conf_threshold, verbose=False)
        if not self.is_pose:
            kwargs["classes"] = [0]  # 포즈 모델은 person 전용이라 classes 불필요

        results = self.model(frame, **kwargs)
        detections = []
        for r in results:
            kpts_data = r.keypoints  # None if not pose model
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                kpts = None
                if kpts_data is not None and i < len(kpts_data):
                    # .data[0]: YOLO가 각 결과에 배치 차원을 감싸므로 [0]으로 벗겨냄
                    kpts = kpts_data[i].data[0].cpu().numpy()  # (17, 3)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "keypoints": kpts,
                })
        return detections
