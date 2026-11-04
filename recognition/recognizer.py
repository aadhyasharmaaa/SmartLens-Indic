from __future__ import annotations

from typing import Any, List, Tuple

from config.settings import TEXT_RECOGNITION_MODEL_NAME


class DevanagariRecognizer:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or TEXT_RECOGNITION_MODEL_NAME

    def recognize(self, image: Any, boxes: List[Tuple[int, int, int, int]] | None = None) -> List[dict]:
        try:
            from paddleocr import TextRecognition
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PaddleOCR is not installed. This prototype is waiting for the exact Devanagari recognizer model and dependency set."
            ) from exc

        recognizer = TextRecognition(model_name=self.model_name)

        if boxes is None or len(boxes) == 0:
            results = recognizer.predict(image)
            return [
                {
                    "box": (0, 0, image.shape[1], image.shape[0]),
                    "text": str(item.get("rec_text", "")).strip(),
                    "confidence": float(item.get("rec_score", 0.0)),
                }
                for item in results
            ]

        outputs: List[dict] = []
        for x1, y1, x2, y2 in boxes:
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            results = recognizer.predict(crop)
            for item in results:
                text = str(item.get("rec_text", "")).strip()
                conf = float(item.get("rec_score", 0.0))
                outputs.append({"box": (x1, y1, x2, y2), "text": text, "confidence": conf})

        return outputs
