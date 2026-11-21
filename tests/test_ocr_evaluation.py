from __future__ import annotations

import numpy as np

from evaluation.evaluate_ocr import (
    character_error_rate,
    evaluate_image,
    word_error_rate,
)


def test_evaluation_metrics_normalize_unicode_before_comparison() -> None:
    assert character_error_rate("कहाँ", "कहाँ") == 0
    assert word_error_rate("हिंदी पाठ", "हिंदी  पाठ") == 0
    assert character_error_rate("भारत", "भार") == 0.25
    assert word_error_rate("पहला दूसरा", "पहला") == 0.5


def test_evaluation_reports_full_frame_and_each_crop_independently() -> None:
    image = np.full((80, 120, 3), 255, dtype=np.uint8)

    class FakeDetector:
        def detect(self, candidate):
            return [
                {"left": 2, "top": 3, "right": 40, "bottom": 20},
                {"left": 2, "top": 30, "right": 50, "bottom": 50},
            ]

    calls = []

    def fake_recognizer(candidate, **kwargs):
        calls.append((candidate.shape, kwargs["scope"], kwargs["method"], kwargs["psm"]))
        return {
            "track": kwargs["track"],
            "image": kwargs["image_name"],
            "scope": kwargs["scope"],
            "region_index": kwargs.get("region_index"),
            "reference_text": kwargs["reference"],
            "recognized_text": "नमस्ते",
            "cer": 0.0 if kwargs["reference"] else None,
            "wer": 0.0 if kwargs["reference"] else None,
            "confidence": 80.0,
            "preprocessing_method": kwargs["method"],
            "page_segmentation_mode": kwargs["psm"],
            "ocr_ms": 1.0,
            "detection_ms": kwargs.get("detection_ms"),
        }

    rows = evaluate_image(
        image,
        track="word-images",
        image_name="sample.png",
        reference="नमस्ते",
        detector=FakeDetector(),
        recognizer=fake_recognizer,
    )

    assert rows[0]["scope"] == "full_frame"
    assert rows[0]["reference_text"] == "नमस्ते"
    crop_rows = [row for row in rows if row["scope"].startswith("detector_crop_")]
    assert len(crop_rows) == 8
    assert {row["region_index"] for row in crop_rows} == {1, 2}
    assert all(row["reference_text"] is None for row in crop_rows)
    assert len(calls) == 7 + 2 * 4
    assert calls[-1][0][0] > calls[-5][0][0]
    assert all(row["page_segmentation_mode"] == 6 for row in crop_rows)
