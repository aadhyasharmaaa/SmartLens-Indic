from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from config.settings import MIN_BOX_AREA, TEXT_DETECTION_MODEL_NAME


class Detector:
    def __init__(
        self,
        detector_model_name: str | None = None,
        recognizer_model_dir: str | None = None,
        min_box_area: int = MIN_BOX_AREA,
        scene_mode: bool = False,
        merge_box_gap: int = 0,
    ):
        self.detector_model_name = detector_model_name or TEXT_DETECTION_MODEL_NAME
        self.recognizer_model_dir = recognizer_model_dir
        self.min_box_area = min_box_area
        self.scene_mode = False
        self.merge_box_gap = 0

    def _merge_near_boxes(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        return sorted(boxes, key=lambda b: (b[1], b[0]))

    def detect(self, image: Any) -> List[Tuple[int, int, int, int]]:
        try:
            from paddleocr import TextDetection
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PaddleOCR is not installed in this workspace. Install the exact OCR dependency before running the PP-OCRv5 detector."
            ) from exc

        detector = TextDetection(model_name=self.detector_model_name)
        results = detector.predict(image)

        boxes: List[Tuple[int, int, int, int]] = []
        for item in results:
            polys = item.get("dt_polys", [])
            if polys is None:
                continue
            for poly in polys:
                arr = np.asarray(poly, dtype=np.float32)
                if arr.size == 0:
                    continue
                xs = arr[:, 0]
                ys = arr[:, 1]
                x1 = int(np.min(xs))
                y1 = int(np.min(ys))
                x2 = int(np.max(xs))
                y2 = int(np.max(ys))
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                if width * height < self.min_box_area:
                    continue
                boxes.append((x1, y1, x2, y2))

        return self._merge_near_boxes(boxes)
