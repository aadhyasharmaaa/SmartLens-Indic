from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR
from postprocessing.normalize_unicode import normalize_unicode_text

MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"


class PaddleDevanagariRecognizer:
    """Local recognition adapter for detected Devanagari text regions."""

    def __init__(
        self,
        model_dir: str | Path = PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR,
        *,
        device: str = "cpu",
        model: Any | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "inference.yml").is_file():
            raise FileNotFoundError(
                f"Devanagari PP-OCRv5 recognition model was not found in {self.model_dir}."
            )
        self.device = device
        if model is None:
            from paddleocr import TextRecognition

            model = TextRecognition(
                model_name=MODEL_NAME,
                model_dir=str(self.model_dir),
                device=device,
            )
        self._model = model

    def recognize(self, image: np.ndarray) -> dict[str, Any]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Recognizer input must be a non-empty NumPy image.")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Recognizer input must be a three-channel uint8 BGR image.")

        predictions = self._model.predict(input=image)
        if len(predictions) != 1:
            raise RuntimeError(
                "PaddleOCR recognition returned an unexpected number of results "
                f"for one image: {len(predictions)}."
            )
        payload = getattr(predictions[0], "json", None)
        if not isinstance(payload, dict) or not isinstance(payload.get("res"), dict):
            raise RuntimeError("PaddleOCR returned an unsupported recognition result format.")
        result = payload["res"]
        text = result.get("rec_text")
        score = result.get("rec_score")
        if not isinstance(text, str):
            raise RuntimeError("PaddleOCR recognition result did not contain rec_text.")
        try:
            confidence = float(score) if score is not None else None
        except (TypeError, ValueError) as exc:
            raise RuntimeError("PaddleOCR recognition score was not numeric.") from exc

        return {
            "text": normalize_unicode_text(text),
            "confidence": confidence,
            "confidence_type": "PaddleOCR recognition score; not calibrated probability",
        }


@lru_cache(maxsize=1)
def get_paddle_devanagari_recognizer() -> PaddleDevanagariRecognizer:
    return PaddleDevanagariRecognizer(device="cpu")
