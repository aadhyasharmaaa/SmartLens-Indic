from __future__ import annotations

import cv2
import numpy as np

from app import _decode_image
from database.repository import (
    get_recent_readings,
    initialize_database,
    record_reading,
)
from services import reading_service


def test_decoded_image_flows_through_reading_and_history(
    tmp_path,
    monkeypatch,
) -> None:
    source_image = np.full((96, 144, 3), 245, dtype=np.uint8)
    source_image[20:54, 18:120] = (70, 120, 180)
    ok, encoded = cv2.imencode(".png", source_image)
    assert ok

    image = _decode_image(encoded.tobytes())
    assert image.shape == source_image.shape
    assert np.array_equal(image, source_image)

    uploads_dir = tmp_path / "uploads"
    readings_dir = tmp_path / "readings"
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", uploads_dir)
    monkeypatch.setattr(reading_service, "READINGS_DIR", readings_dir)
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2GRAY,
        ),
    )

    class FixedOCR:
        def check_available(self):
            return {"version": "test-double"}

        def recognize(self, candidate):
            assert candidate.ndim == 2
            return {
                "text": "नमस्ते, दुनिया!",
                "confidence": 88.0,
                "word_confidences": [88.0],
                "word_count": 2,
                "word_details": [],
            }

    class FixedDetector:
        def detect(self, candidate):
            assert candidate.shape == image.shape
            return [
                {
                    "left": 12,
                    "top": 14,
                    "right": 126,
                    "bottom": 62,
                    "text": "",
                    "confidence": 0.9,
                    "source": "test-detector",
                }
            ]

    class FixedDevanagariRecognizer:
        def recognize(self, candidate):
            assert candidate.ndim == 3
            return {"text": "नमस्ते दुनिया", "confidence": 0.95}

    spoken: list[str] = []

    def fake_synthesis(text, output_path):
        spoken.append(text)
        output_path.write_bytes(b"test-only audio placeholder")
        return True, output_path

    monkeypatch.setattr(reading_service, "generate_hindi_audio", fake_synthesis)

    result = reading_service.process_image(
        image,
        input_kind="upload",
        generate_audio=True,
        ocr=FixedOCR(),
        text_detector=FixedDetector(),
        paddle_recognizer=FixedDevanagariRecognizer(),
    )

    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    record_reading(result, database_path=database_path)
    history = get_recent_readings(database_path=database_path)

    assert result["status"] == "success"
    assert result["recognized_text"] == "नमस्ते दुनिया"
    assert result["recognition_engine"] == "PaddleOCR-Devanagari with Tesseract fallback"
    assert result["speech_candidates_validated"] is True
    assert spoken == ["नमस्ते दुनिया"]
    assert result["original_image_path"].is_file()
    assert result["preprocessed_image_path"].is_file()
    assert result["audio_path"].read_bytes() == b"test-only audio placeholder"
    assert len(history) == 1
    assert history[0]["recognized_text"] == "नमस्ते दुनिया"
    assert history[0]["recognition_engine"] == result["recognition_engine"]
    assert history[0]["audio_path"] == str(result["audio_path"])
    assert history[0]["input_kind"] == "upload"


def test_detector_failure_keeps_tesseract_fallback_in_history(
    tmp_path,
    monkeypatch,
) -> None:
    image = np.full((88, 132, 3), 230, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(
            candidate,
            cv2.COLOR_BGR2GRAY,
        ),
    )

    class FixedOCR:
        def check_available(self):
            return {"version": "test-double"}

        def recognize(self, candidate):
            word = {
                "text": "सुरक्षित",
                "confidence": 82.0,
                "left": 12,
                "top": 18,
                "width": 58,
                "height": 24,
                "block_num": "1",
                "par_num": "1",
                "line_num": "1",
            }
            return {
                "text": "सुरक्षित",
                "confidence": 82.0,
                "word_confidences": [82.0],
                "word_count": 1,
                "word_details": [word],
            }

    class BrokenDetector:
        def detect(self, candidate):
            raise RuntimeError("simulated local detector failure")

    class NoTextRecognizer:
        def recognize(self, candidate):
            return {"text": "", "confidence": None}

    result = reading_service.process_image(
        image,
        ocr=FixedOCR(),
        text_detector=BrokenDetector(),
        paddle_recognizer=NoTextRecognizer(),
    )

    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    record_reading(result, database_path=database_path)
    saved = get_recent_readings(database_path=database_path)[0]

    assert result["recognized_text"] == "सुरक्षित"
    assert result["recognition_engine"] == "Tesseract"
    assert result["detection_source"] == "Tesseract OCR fallback"
    assert "simulated local detector failure" in result["detection_warning"]
    assert len(result["region_crops"]) == 1
    assert saved["recognized_text"] == result["recognized_text"]
    assert saved["recognition_engine"] == "Tesseract"
