from __future__ import annotations

import cv2
import numpy as np

from evaluation import compare_recognizers


class FakeDetector:
    def detect(self, image):
        return [
            {"left": 0, "top": 0, "right": 20, "bottom": 20},
            {"left": 20, "top": 0, "right": 40, "bottom": 20},
        ]


def test_scene_comparison_keeps_image_name_and_region_order(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "scene.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((20, 40, 3), dtype=np.uint8))
    monkeypatch.setattr(compare_recognizers, "get_text_detector", lambda: FakeDetector())

    calls = []

    def measure(**kwargs):
        calls.append((kwargs["image_name"], kwargs["region_index"], kwargs["recognizer_name"]))
        return {
            "image": kwargs["image_name"],
            "region_index": kwargs["region_index"],
            "recognized_text": f"{kwargs['recognizer_name']}-{kwargs['region_index']}",
            "scope": f"same_detector_crop:{kwargs['recognizer_name']}",
            "track": "scene-images",
        }

    monkeypatch.setattr(compare_recognizers, "_measure", measure)
    rows = compare_recognizers._run_scene_track(
        [image_path],
        tesseract=object(),
        paddle=object(),
    )

    assert calls == [
        (str(image_path), 1, "Tesseract"),
        (str(image_path), 1, "PaddleOCR-Devanagari"),
        (str(image_path), 2, "Tesseract"),
        (str(image_path), 2, "PaddleOCR-Devanagari"),
    ]
    assert rows[-2]["recognized_text"] == "Tesseract-1\nTesseract-2"
    assert rows[-1]["recognized_text"] == "PaddleOCR-Devanagari-1\nPaddleOCR-Devanagari-2"
    assert all(row["image"] == str(image_path) for row in rows)
