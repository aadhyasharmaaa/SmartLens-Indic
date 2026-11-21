from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PP_OCR_DETECTOR_MODEL_DIR, PP_OCR_DETECTOR_MODEL_NAME


class PaddleTextDetector:
    def __init__(
        self,
        model_dir: str | Path = PP_OCR_DETECTOR_MODEL_DIR,
        model_name: str = PP_OCR_DETECTOR_MODEL_NAME,
    ) -> None:
        self.model_dir = Path(model_dir)
        if not (self.model_dir / "inference.yml").is_file():
            raise FileNotFoundError(
                f"PP-OCRv5 detector model files were not found in {self.model_dir}."
            )

        from paddleocr import TextDetection

        self._model = TextDetection(model_name=model_name, model_dir=str(self.model_dir))

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Text detector input must be a non-empty NumPy image.")
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError("Text detector input must be a uint8 grayscale or color image.")
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported text detector image shape: {image.shape}")

        height, width = image.shape[:2]
        predictions = self._model.predict(image, batch_size=1)
        regions: list[dict[str, Any]] = []
        for prediction in predictions:
            payload = getattr(prediction, "json", None)
            if not isinstance(payload, dict) or not isinstance(payload.get("res"), dict):
                raise RuntimeError("PP-OCRv5 returned an unsupported detection result format.")

            result = payload["res"]
            polygons = result.get("dt_polys")
            scores = result.get("dt_scores")
            if polygons is None:
                polygons = []
            elif isinstance(polygons, np.ndarray):
                polygons = polygons.tolist()
            if scores is None:
                scores = []
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()
            if len(polygons) != len(scores):
                raise RuntimeError("PP-OCRv5 returned mismatched text polygons and scores.")

            for polygon, score in zip(polygons, scores):
                points = np.asarray(polygon, dtype=np.float32)
                if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 4:
                    continue

                left = max(0, math.floor(float(points[:, 0].min())))
                top = max(0, math.floor(float(points[:, 1].min())))
                right = min(width, math.ceil(float(points[:, 0].max())) + 1)
                bottom = min(height, math.ceil(float(points[:, 1].max())) + 1)
                if left >= right or top >= bottom:
                    continue

                regions.append(
                    {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "text": "",
                        "confidence": float(score),
                        "source": PP_OCR_DETECTOR_MODEL_NAME,
                        "polygon": [
                            (float(point[0]), float(point[1])) for point in points
                        ],
                    }
                )

        regions.sort(key=lambda region: (region["top"], region["left"]))
        return _merge_same_line_regions(regions)


def _merge_same_line_regions(
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    merged_members: list[list[dict[str, Any]]] = []
    for region in regions:
        matching = next(
            (
                index
                for index, members in enumerate(merged_members)
                if all(_regions_share_line(region, member) for member in members)
            ),
            None,
        )
        if matching is None:
            merged.append(region)
            merged_members.append([_region_bounds(region)])
            continue

        previous = merged[matching]
        previous["left"] = min(previous["left"], region["left"])
        previous["top"] = min(previous["top"], region["top"])
        previous["right"] = max(previous["right"], region["right"])
        previous["bottom"] = max(previous["bottom"], region["bottom"])
        previous["confidence"] = min(previous["confidence"], region["confidence"])
        previous["polygons"] = previous.get("polygons", [previous["polygon"]]) + [
            region["polygon"]
        ]
        previous["polygon"] = None
        merged_members[matching].append(_region_bounds(region))

    merged.sort(key=lambda region: (region["top"], region["left"]))
    return merged


def _region_bounds(region: dict[str, Any]) -> dict[str, int]:
    return {
        "left": region["left"],
        "top": region["top"],
        "right": region["right"],
        "bottom": region["bottom"],
    }


def _regions_share_line(
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    first_height = max(1, first["bottom"] - first["top"])
    second_height = max(1, second["bottom"] - second["top"])
    vertical_overlap = max(
        0,
        min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]),
    )
    overlap_ratio = vertical_overlap / min(first_height, second_height)
    first_center = (first["top"] + first["bottom"]) / 2
    second_center = (second["top"] + second["bottom"]) / 2
    center_distance = abs(first_center - second_center)
    horizontal_gap = max(
        0,
        max(first["left"], second["left"]) - min(first["right"], second["right"]),
    )
    line_height = min(first_height, second_height)
    max_gap = max(8, round(line_height * 1.5))
    return (
        overlap_ratio >= 0.45
        and center_distance <= line_height * 0.5
        and horizontal_gap <= max_gap
    )


@lru_cache(maxsize=1)
def get_text_detector() -> PaddleTextDetector:
    return PaddleTextDetector()
