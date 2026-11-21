from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pytesseract.pytesseract import TesseractNotFoundError

from config.settings import (
    TESSDATA_DIR,
    TESSERACT_CMD,
    TESSERACT_LANGUAGE,
    TESSERACT_PAGE_SEGMENTATION_MODE,
)


class TesseractOCR:
    def __init__(
        self,
        executable_path: str | Path = TESSERACT_CMD,
        tessdata_dir: str | Path = TESSDATA_DIR,
        language: str = TESSERACT_LANGUAGE,
        page_segmentation_mode: int = TESSERACT_PAGE_SEGMENTATION_MODE,
    ):
        self.executable_path = Path(executable_path)
        self.tessdata_dir = Path(tessdata_dir)
        self.language = language
        if not 0 <= page_segmentation_mode <= 13:
            raise ValueError("Tesseract page segmentation mode must be between 0 and 13.")
        self.page_segmentation_mode = page_segmentation_mode
        pytesseract.pytesseract.tesseract_cmd = str(self.executable_path)
        os.environ["TESSDATA_PREFIX"] = str(self.tessdata_dir)

    def check_available(self) -> dict[str, Any]:
        if not self.executable_path.is_file():
            raise FileNotFoundError(f"Tesseract executable was not found: {self.executable_path}")
        if not self.tessdata_dir.is_dir():
            raise FileNotFoundError(f"Tesseract language-data directory was not found: {self.tessdata_dir}")

        language_file = self.tessdata_dir / f"{self.language}.traineddata"
        if not language_file.is_file():
            raise FileNotFoundError(f"Required Tesseract language data was not found: {language_file}")

        try:
            version = str(pytesseract.get_tesseract_version())
            languages = pytesseract.get_languages()
        except TesseractNotFoundError as exc:
            raise RuntimeError(f"Could not start Tesseract at {self.executable_path}") from exc

        if self.language not in languages:
            raise RuntimeError(
                f"Tesseract does not report language '{self.language}' in {self.tessdata_dir}. "
                f"Available languages: {', '.join(languages)}"
            )

        return {"version": version, "languages": languages}

    def recognize(self, image: np.ndarray) -> dict[str, Any]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Tesseract input must be a non-empty NumPy image.")
        if image.dtype != np.uint8:
            raise ValueError(f"Tesseract input must use uint8 pixels, got {image.dtype}.")

        if image.ndim == 2:
            rgb_image = image
        elif image.ndim == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError(f"Unsupported Tesseract image shape: {image.shape}")

        config = f"--psm {self.page_segmentation_mode}"
        try:
            data = pytesseract.image_to_data(
                rgb_image,
                lang=self.language,
                config=config,
                output_type=Output.DICT,
            )
        except TesseractNotFoundError as exc:
            raise RuntimeError(f"Could not start Tesseract at {self.executable_path}") from exc

        lines: OrderedDict[tuple[str, str, str], list[str]] = OrderedDict()
        confidences: list[float] = []
        word_details: list[dict[str, Any]] = []
        for index, raw_text in enumerate(data.get("text", [])):
            text = str(raw_text).strip()
            if not text:
                continue

            try:
                confidence = float(data["conf"][index])
                if confidence < 0:
                    confidence = None
            except (TypeError, ValueError, IndexError):
                confidence = None

            key = (
                str(data["block_num"][index]),
                str(data["par_num"][index]),
                str(data["line_num"][index]),
            )
            lines.setdefault(key, []).append(text)

            if confidence is not None:
                confidences.append(confidence)
            word_details.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "left": int(data["left"][index]),
                    "top": int(data["top"][index]),
                    "width": int(data["width"][index]),
                    "height": int(data["height"][index]),
                    "block_num": str(data["block_num"][index]),
                    "par_num": str(data["par_num"][index]),
                    "line_num": str(data["line_num"][index]),
                    "uncertain": confidence is not None and confidence < 50,
                }
            )

        recognized_text = "\n".join(" ".join(words) for words in lines.values())
        return {
            "text": recognized_text,
            "confidence": mean(confidences) if confidences else None,
            "word_confidences": confidences,
            "word_count": sum(len(words) for words in lines.values()),
            "word_details": word_details,
        }
