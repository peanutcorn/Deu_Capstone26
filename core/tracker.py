from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np


class HumanTracker:
    """DeepSORT 래퍼. YOLO 감지 결과를 받아 트랙 ID를 부여."""

    def __init__(self, max_age: int = 30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections: list[dict], frame: np.ndarray) -> list[dict]:
        """
        detections: HumanDetector.detect() 결과
        Returns list of dicts:
          { 'track_id': int, 'bbox': [x1,y1,x2,y2], 'conf': float }
        """
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        # deep_sort_realtime expects: [([x1,y1,w,h], conf, class_id), ...]
        raw = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2 - x1, y2 - y1
            raw.append(([x1, y1, w, h], d["conf"], 0))

        tracks = self.tracker.update_tracks(raw, frame=frame)
        results = []
        for t in tracks:
            # 연속 프레임에서 충분히 관측되지 않은 잠정 트랙은 제외
            if not t.is_confirmed():
                continue
            ltrb = t.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            results.append({
                # DeepSORT가 track_id를 문자열로 반환하므로 명시적 변환 필요 (f-string 외 % 연산 오류 방지)
                "track_id": int(t.track_id),
                "bbox": [x1, y1, x2, y2],
            })
        return results
