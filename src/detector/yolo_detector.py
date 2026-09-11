import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import config

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class YOLOHumanDetector:
    """
    Free YOLO Human Bounding Box Detector.
    Isolates person bounding boxes (COCO class 0) for ROI extraction and pre-recording framing checks.
    """

    def __init__(self, model_name: str = config.YOLO_MODEL_NAME, conf_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD):
        self.conf_threshold = conf_threshold
        self.model = None
        
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(model_name)
            except Exception as e:
                print(f"[YOLO Warning] Could not load YOLO model '{model_name}': {e}. Falling back to OpenCV ROI mode.")
        else:
            print("[YOLO Warning] Ultralytics package not available. Falling back to OpenCV ROI mode.")

    def detect_person(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], float, List[Dict]]:
        """
        Detects primary human subject in the frame.
        Returns:
            best_bbox: (x1, y1, x2, y2) of the highest-confidence/largest person box, or None.
            max_conf: Confidence score of the detection.
            all_persons: List of all detected person bounding boxes.
        """
        if frame is None:
            return None, 0.0, []

        h, w, _ = frame.shape

        if self.model is not None:
            try:
                results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
                all_persons = []
                best_bbox = None
                best_area_conf = 0.0

                for box in results.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    
                    if cls_id == config.PERSON_CLASS_ID:
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                        
                        # Clip to bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        area = (x2 - x1) * (y2 - y1)

                        person_dict = {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": conf,
                            "area": area
                        }
                        all_persons.append(person_dict)

                        # Primary subject heuristic: product of area and confidence
                        area_conf = area * conf
                        if area_conf > best_area_conf:
                            best_area_conf = area_conf
                            best_bbox = (x1, y1, x2, y2)
                            max_conf = conf

                if best_bbox is not None:
                    return best_bbox, max_conf, all_persons

            except Exception as e:
                print(f"[YOLO Error] Detection failed: {e}")

        # Fallback: treat entire frame as human ROI
        return (0, 0, w, h), 1.0, [{"bbox": (0, 0, w, h), "confidence": 1.0, "area": w * h}]

    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float, label: str = "Human Subject") -> np.ndarray:
        """Draws stylized bounding box overlay on frame."""
        output = frame.copy()
        if bbox is None:
            return output

        x1, y1, x2, y2 = bbox
        color = (0, 255, 128)  # Glowing emerald green
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text = f"{label} ({int(confidence * 100)}%)"
        cv2.putText(output, text, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return output
