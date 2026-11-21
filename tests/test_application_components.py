from __future__ import annotations

import sqlite3
import inspect
import wave
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from app import _decode_image
from app import _is_low_information_camera_frame
from app import _run_reading
from database.repository import get_reading_count, get_recent_readings, initialize_database, record_reading
from postprocessing.text_cleanup import clean_ocr_text
from preprocessing.preprocess import (
    preprocess_for_tesseract,
    resolve_tesseract_preprocessing_method,
)
from recognition.tesseract_ocr import TesseractOCR
from recognition.paddle_text_detector import PaddleTextDetector
from recognition.text_regions import group_word_regions
from services import reading_service
from tts.piper_tts import generate_hindi_audio


class EmptyDetector:
    def detect(self, image):
        return []


@pytest.fixture(autouse=True)
def avoid_loading_detector_model_in_unit_tests(monkeypatch) -> None:
    monkeypatch.setattr(reading_service, "get_text_detector", lambda: EmptyDetector())
    monkeypatch.setattr(reading_service, "get_paddle_devanagari_recognizer", lambda: None)


def test_text_cleanup_preserves_hindi_and_removes_empty_lines() -> None:
    assert clean_ocr_text("  नमस्ते   दुनिया  \r\n \n") == "नमस्ते दुनिया"


def test_history_starts_empty_and_persists_unicode(tmp_path) -> None:
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)

    assert get_recent_readings(database_path=database_path) == []
    assert get_reading_count(database_path) == 0

    record_reading(
        {
            "id": "test-reading",
            "status": "success",
            "input_kind": "upload",
            "recognized_text": "हिंदी पाठ",
            "total_ms": 12.5,
            "text_detection_ms": 8.5,
            "region_preprocess_ms": 3.2,
        },
        database_path=database_path,
    )

    readings = get_recent_readings(database_path=database_path)
    assert len(readings) == 1
    assert readings[0]["recognized_text"] == "हिंदी पाठ"
    assert readings[0]["text_detection_ms"] == 8.5
    assert readings[0]["region_preprocess_ms"] == 3.2
    assert readings[0]["recognition_engine"] == "Tesseract"
    assert get_reading_count(database_path) == 1


def test_history_initialization_migrates_existing_database(tmp_path) -> None:
    database_path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE reading_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                input_kind TEXT NOT NULL,
                original_image_path TEXT,
                preprocessed_image_path TEXT,
                recognized_text TEXT,
                confidence_mean REAL,
                preprocess_ms REAL,
                ocr_ms REAL,
                tts_ms REAL,
                total_ms REAL,
                audio_path TEXT,
                error_code TEXT,
                error_message TEXT
            )
            """
        )

    initialize_database(database_path)
    record_reading(
        {
            "id": "migrated-reading",
            "status": "success",
            "input_kind": "upload",
            "recognized_text": "सुरक्षित",
            "text_detection_ms": 14.0,
            "region_preprocess_ms": 4.0,
        },
        database_path=database_path,
    )

    reading = get_recent_readings(database_path=database_path)[0]
    assert reading["text_detection_ms"] == 14.0
    assert reading["region_preprocess_ms"] == 4.0


def test_history_initialization_migrates_recognition_engine_column(tmp_path) -> None:
    database_path = tmp_path / "existing.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE reading_sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                input_kind TEXT NOT NULL,
                original_image_path TEXT,
                preprocessed_image_path TEXT,
                recognized_text TEXT,
                confidence_mean REAL,
                preprocess_ms REAL,
                text_detection_ms REAL,
                region_preprocess_ms REAL,
                ocr_ms REAL,
                tts_ms REAL,
                total_ms REAL,
                audio_path TEXT,
                error_code TEXT,
                error_message TEXT
            )
            """
        )

    initialize_database(database_path)
    record_reading(
        {
            "id": "paddle-reading",
            "status": "success",
            "input_kind": "upload",
            "recognized_text": "हिंदी",
            "recognition_engine": "PaddleOCR-Devanagari",
        },
        database_path=database_path,
    )

    reading = get_recent_readings(database_path=database_path)[0]
    assert reading["recognition_engine"] == "PaddleOCR-Devanagari"


def test_history_surfaces_database_errors(tmp_path) -> None:
    database_path = tmp_path / "invalid.sqlite3"
    database_path.write_text("not a database", encoding="utf-8")

    with pytest.raises(sqlite3.DatabaseError):
        get_recent_readings(database_path=database_path)


def test_tesseract_reports_missing_executable(tmp_path) -> None:
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tessdata / "hin.traineddata").write_bytes(b"test")
    engine = TesseractOCR(tmp_path / "missing-tesseract.exe", tessdata)

    with pytest.raises(FileNotFoundError, match="Tesseract executable"):
        engine.check_available()


def test_tesseract_reports_missing_hindi_data(tmp_path) -> None:
    tessdata = tmp_path / "tessdata"
    tessdata.mkdir()
    (tmp_path / "tesseract.exe").write_bytes(b"test executable placeholder")
    engine = TesseractOCR(tmp_path / "tesseract.exe", tessdata)

    with pytest.raises(FileNotFoundError, match="language data"):
        engine.check_available()


def test_piper_reports_missing_model_without_install_or_download(tmp_path) -> None:
    ok, message = generate_hindi_audio("नमस्ते", tmp_path / "result.wav", tmp_path, "missing")

    assert not ok
    assert "voice model was not found" in message
    assert not (tmp_path / "result.wav").exists()


def test_piper_reports_synthesis_error(tmp_path, monkeypatch) -> None:
    from config.settings import PIPER_VOICE_PATH
    from piper import PiperVoice

    if not PIPER_VOICE_PATH.is_file() or not PIPER_VOICE_PATH.with_suffix(".onnx.json").is_file():
        pytest.skip("Configured Piper voice files are not installed.")

    def fail_synthesis(cls, *args, **kwargs):
        raise RuntimeError("simulated Piper failure")

    monkeypatch.setattr(PiperVoice, "load", classmethod(fail_synthesis))
    ok, message = generate_hindi_audio("नमस्ते", tmp_path / "result.wav")

    assert not ok
    assert "simulated Piper failure" in message
    assert not (tmp_path / "result.wav").exists()


def test_piper_skips_synthesis_when_hindi_has_no_phonemes(
    tmp_path,
    monkeypatch,
) -> None:
    from piper import PiperVoice

    class EmptyPhonemizerVoice:
        def phonemize(self, text):
            return []

        def synthesize_wav(self, text, audio_file):
            pytest.fail("Piper must not synthesize text without phonemes")

    voice = EmptyPhonemizerVoice()
    monkeypatch.setattr(PiperVoice, "load", classmethod(lambda cls, *args, **kwargs: voice))

    ok, message = generate_hindi_audio("नमस्ते", tmp_path / "result.wav")

    assert not ok
    assert "could not produce Hindi phonemes" in message
    assert not (tmp_path / "result.wav").exists()


def test_empty_ocr_skips_tts_and_preserves_input(tmp_path, monkeypatch) -> None:
    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda prepared, method="auto": prepared[:, :, 0],
    )

    def fail_if_synthesized(*args, **kwargs):
        pytest.fail("Piper must not be called for empty OCR text")

    monkeypatch.setattr(reading_service, "generate_hindi_audio", fail_if_synthesized)

    class EmptyOCR:
        def check_available(self):
            return {"version": "test"}

        def recognize(self, prepared):
            return {"text": " \n ", "confidence": None, "word_confidences": [], "word_count": 0}

    result = reading_service.process_image(image, generate_audio=True, ocr=EmptyOCR())

    assert result["status"] == "empty_text"
    assert result["recognized_text"] == ""
    assert result["audio_path"] is None
    assert result["original_image_path"].is_file()
    assert np.array_equal(result["original_image"], image)


def test_unavailable_webcam_placeholder_skips_ocr_and_speech(monkeypatch) -> None:
    import cv2
    import app

    image = np.full((275, 336, 3), 36, dtype=np.uint8)
    cv2.rectangle(image, (147, 118), (188, 151), (240, 240, 240), 3)
    cv2.line(image, (153, 145), (181, 121), (240, 240, 240), 4)
    warnings = []
    monkeypatch.setattr(app.st, "warning", warnings.append)
    monkeypatch.setattr(
        app,
        "process_image",
        lambda *args, **kwargs: pytest.fail("A blank camera placeholder must not reach OCR"),
    )

    assert _is_low_information_camera_frame(image)
    _run_reading(image, "webcam", True)

    assert len(warnings) == 1
    assert "almost blank" in warnings[0]


def test_zero_confidence_ocr_does_not_generate_speech(tmp_path, monkeypatch) -> None:
    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda prepared, method="auto": prepared[:, :, 0],
    )

    class ZeroConfidenceOCR:
        def check_available(self):
            return {"version": "test"}

        def recognize(self, prepared):
            return {
                "text": "रॉ",
                "confidence": 0.0,
                "word_confidences": [0.0],
                "word_count": 1,
                "word_details": [],
            }

    def fail_if_synthesized(*args, **kwargs):
        pytest.fail("Piper must not speak text reported with zero Tesseract confidence")

    monkeypatch.setattr(reading_service, "generate_hindi_audio", fail_if_synthesized)

    result = reading_service.process_image(
        image,
        generate_audio=True,
        ocr=ZeroConfidenceOCR(),
        text_detector=EmptyDetector(),
    )

    assert result["recognized_text"] == "रॉ"
    assert result["audio_path"] is None
    assert result["tts_ms"] is None
    assert "confidence was zero" in result["tts_skipped_reason"]


@pytest.mark.parametrize(
    ("text", "confidence", "reason_fragment"),
    [
        ("रॉ", None, "confidence was zero, invalid"),
        ("रॉ", float("nan"), "confidence was zero, invalid"),
        ("||| ...", 82.0, "no Devanagari letters or numerals"),
        ("१२३", 40.0, None),
    ],
)
def test_speech_gate_rejects_unusable_text_without_confidence_threshold(
    tmp_path,
    monkeypatch,
    text,
    confidence,
    reason_fragment,
) -> None:
    import cv2

    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda prepared, method="auto": cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY),
    )
    synthesized: list[str] = []

    class FixedOCR:
        def check_available(self):
            return {"version": "test"}

        def recognize(self, prepared):
            return {
                "text": text,
                "confidence": confidence,
                "word_confidences": [confidence] if confidence is not None else [],
                "word_count": 1 if text else 0,
                "word_details": [],
            }

    def fake_synthesis(value, output_path):
        synthesized.append(value)
        return True, output_path

    monkeypatch.setattr(reading_service, "generate_hindi_audio", fake_synthesis)

    result = reading_service.process_image(
        image,
        generate_audio=True,
        ocr=FixedOCR(),
        text_detector=EmptyDetector(),
    )

    if reason_fragment is None:
        assert synthesized == [text]
        assert result["audio_path"] is not None
        assert result["tts_skipped_reason"] is None
    else:
        assert synthesized == []
        assert result["audio_path"] is None
        assert reason_fragment in result["tts_skipped_reason"]


def test_invalid_image_is_rejected_before_processing() -> None:
    with pytest.raises(ValueError, match="empty or invalid"):
        reading_service.process_image(np.empty((0, 0, 3), dtype=np.uint8))


def test_automatic_preprocessing_keeps_larger_images_grayscale() -> None:
    image = np.tile(np.arange(120, dtype=np.uint8), (80, 1))

    assert resolve_tesseract_preprocessing_method(image, "auto") == "clahe_grayscale"
    prepared = preprocess_for_tesseract(image, "auto")
    assert prepared.ndim == 2
    assert np.unique(prepared).size > 2


def test_automatic_preprocessing_uses_denoised_grayscale_for_tiny_word_images() -> None:
    image = np.full((19, 32, 3), 255, dtype=np.uint8)

    assert resolve_tesseract_preprocessing_method(image, "auto") == "denoised_grayscale"
    assert np.array_equal(
        preprocess_for_tesseract(image, "auto"),
        preprocess_for_tesseract(image, "denoised_grayscale"),
    )
    assert preprocess_for_tesseract(image, "legacy_otsu").ndim == 2


def test_manual_crop_is_sent_to_ocr_without_changing_original(tmp_path, monkeypatch) -> None:
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    observed_shape = None
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")

    def prepare(region, method="auto"):
        nonlocal observed_shape
        observed_shape = region.shape
        return cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    import cv2

    monkeypatch.setattr(reading_service, "preprocess_for_tesseract", prepare)

    class FixedOCR:
        def check_available(self):
            return {"version": "test"}

        def recognize(self, prepared):
            return {
                "text": "नमस्ते",
                "confidence": 80.0,
                "word_confidences": [80.0],
                "word_count": 1,
                "word_details": [],
            }

    original = image.copy()
    result = reading_service.process_image(
        image,
        crop_box=(10, 15, 70, 55),
        preprocessing_method="grayscale",
        page_segmentation_mode=6,
        ocr=FixedOCR(),
    )

    assert observed_shape == (40, 60, 3)
    assert result["selected_region_image"].shape == (40, 60, 3)
    assert result["crop_box"] == (10, 15, 70, 55)
    assert np.array_equal(image, original)
    assert np.array_equal(result["original_image"], original)


def test_invalid_crop_bounds_are_rejected() -> None:
    image = np.full((30, 40, 3), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="Crop box"):
        reading_service.process_image(image, crop_box=(10, 10, 5, 20))


def test_crop_keyword_is_supported_by_app_and_service_callers() -> None:
    assert "crop_box" in inspect.signature(reading_service.process_image).parameters
    assert "crop_box" in inspect.signature(_run_reading).parameters


def test_word_regions_are_sorted_and_overlapping_duplicates_are_removed() -> None:
    words = [
        {
            "text": "नीचे",
            "confidence": 70.0,
            "left": 10,
            "top": 100,
            "width": 40,
            "height": 20,
            "block_num": "1",
            "par_num": "1",
            "line_num": "2",
        },
        {
            "text": "ऊपर",
            "confidence": 75.0,
            "left": 10,
            "top": 20,
            "width": 40,
            "height": 20,
            "block_num": "1",
            "par_num": "1",
            "line_num": "1",
        },
        {
            "text": "ऊपर",
            "confidence": 90.0,
            "left": 11,
            "top": 21,
            "width": 39,
            "height": 19,
            "block_num": "2",
            "par_num": "1",
            "line_num": "1",
        },
    ]

    regions = group_word_regions(words, (160, 100, 3))

    assert [region["text"] for region in regions] == ["ऊपर", "नीचे"]
    assert regions[0]["confidence"] == 90.0


def test_paddle_detector_converts_polygons_and_clips_bounds() -> None:
    class FakeModel:
        def predict(self, image, batch_size):
            assert batch_size == 1
            assert image.shape == (40, 60, 3)
            return [
                type(
                    "FakePrediction",
                    (),
                    {
                        "json": {
                            "res": {
                                "dt_polys": np.array(
                                    [[[-5, 4], [30, 4], [30, 25], [-5, 25]]],
                                    dtype=np.int16,
                                ),
                                "dt_scores": np.array([0.91], dtype=np.float32),
                            }
                        }
                    },
                )()
            ]

    detector = object.__new__(PaddleTextDetector)
    detector._model = FakeModel()

    regions = detector.detect(np.zeros((40, 60, 3), dtype=np.uint8))

    assert len(regions) == 1
    assert regions[0]["source"] == "PP-OCRv5_mobile_det"
    assert regions[0]["confidence"] == pytest.approx(0.91)
    assert (regions[0]["left"], regions[0]["top"]) == (0, 4)
    assert (regions[0]["right"], regions[0]["bottom"]) == (31, 26)


def test_paddle_detector_reports_missing_local_model(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="detector model files"):
        PaddleTextDetector(model_dir=tmp_path)


def test_paddle_detector_merges_neighboring_boxes_on_the_same_line() -> None:
    class FakeModel:
        def predict(self, image, batch_size):
            return [
                type(
                    "FakePrediction",
                    (),
                    {
                        "json": {
                            "res": {
                                "dt_polys": [
                                    [[5, 10], [35, 10], [35, 40], [5, 40]],
                                    [[40, 11], [70, 11], [70, 39], [40, 39]],
                                    [[5, 70], [40, 70], [40, 100], [5, 100]],
                                ],
                                "dt_scores": [0.9, 0.8, 0.85],
                            }
                        }
                    },
                )()
            ]

    detector = object.__new__(PaddleTextDetector)
    detector._model = FakeModel()

    regions = detector.detect(np.zeros((120, 100, 3), dtype=np.uint8))

    assert len(regions) == 2
    assert (regions[0]["left"], regions[0]["right"]) == (5, 71)
    assert regions[0]["confidence"] == pytest.approx(0.8)
    assert len(regions[0]["polygons"]) == 2
    assert regions[1]["top"] == 70


def test_detector_failure_is_reported_and_uses_tesseract_box_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    import cv2

    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )

    class FixedOCR:
        def __init__(self, page_segmentation_mode=11):
            self.page_segmentation_mode = page_segmentation_mode

        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            return {
                "text": "नमस्ते",
                "confidence": 80.0,
                "word_confidences": [80.0],
                "word_count": 1,
                "word_details": [
                    {
                        "text": "नमस्ते",
                        "confidence": 80.0,
                        "left": 10,
                        "top": 10,
                        "width": 40,
                        "height": 20,
                        "block_num": "1",
                        "par_num": "1",
                        "line_num": "1",
                        "uncertain": False,
                    }
                ],
            }

    def fail_detector():
        raise FileNotFoundError("detector files missing")

    monkeypatch.setattr(reading_service, "TesseractOCR", FixedOCR)
    monkeypatch.setattr(reading_service, "get_text_detector", fail_detector)

    result = reading_service.process_image(image, ocr=FixedOCR())

    assert result["recognized_text"] == "नमस्ते"
    assert result["detection_source"] == "Tesseract OCR fallback"
    assert "detector files missing" in result["detection_warning"]
    assert len(result["detected_regions"]) == 1


def test_streamlit_result_renders_audio_player(tmp_path) -> None:
    audio_path = tmp_path / "hindi.wav"
    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(22050)
        audio_file.writeframes(b"\x00\x00" * 220)

    image = np.full((32, 48, 3), 255, dtype=np.uint8)
    result = {
        "id": "audio-ui-test",
        "status": "success",
        "recognized_text": "नमस्ते",
        "confidence_mean": 80.0,
        "word_details": [],
        "original_image": image,
        "selected_region_image": image,
        "preprocessed_image": image[:, :, 0],
        "preprocessing_method": "grayscale",
        "page_segmentation_mode": 6,
        "detected_regions": [],
        "detection_source": "PP-OCRv5_mobile_det",
        "detection_warning": None,
        "region_crops": [],
        "regions_not_retried": 0,
        "preprocess_ms": 1.0,
        "text_detection_ms": 2.0,
        "region_preprocess_ms": 0.0,
        "ocr_ms": 3.0,
        "tts_ms": 4.0,
        "total_ms": 10.0,
        "audio_path": audio_path,
    }

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(str(app_path), default_timeout=30).run()
    app_test.session_state["last_reading"] = result
    app_test.run()

    assert not app_test.exception
    assert any("नमस्ते" in area.value for area in app_test.text_area)
    assert len(app_test.get("audio")) == 1


def test_full_frame_ocr_automatically_crops_and_processes_multiple_regions(
    tmp_path,
    monkeypatch,
) -> None:
    import cv2

    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )
    calls: list[tuple[int, int, int]] = []
    primary_word_details = [
        {
            "text": "पहली",
            "confidence": 90.0,
            "left": 20,
            "top": 20,
            "width": 40,
            "height": 15,
            "block_num": "1",
            "par_num": "1",
            "line_num": "1",
            "uncertain": False,
        },
        {
            "text": "दूसरी",
            "confidence": 80.0,
            "left": 20,
            "top": 80,
            "width": 45,
            "height": 15,
            "block_num": "1",
            "par_num": "1",
            "line_num": "2",
            "uncertain": False,
        },
    ]

    class FakeOCR:
        def __init__(self, page_segmentation_mode=11):
            self.page_segmentation_mode = page_segmentation_mode

        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            calls.append((candidate.shape[0], candidate.shape[1], self.page_segmentation_mode))
            if len(calls) == 1:
                return {
                    "text": "दूसरी\nपहली",
                    "confidence": 85.0,
                    "word_confidences": [80.0, 90.0],
                    "word_count": 2,
                    "word_details": primary_word_details,
                }
            return {
                "text": "क्षेत्र",
                "confidence": 70.0,
                "word_confidences": [70.0],
                "word_count": 1,
                "word_details": [],
            }

    monkeypatch.setattr(reading_service, "TesseractOCR", FakeOCR)

    class FakeDetector:
        def detect(self, candidate):
            assert candidate.shape == image.shape
            return [
                {
                    "left": 15,
                    "top": 15,
                    "right": 70,
                    "bottom": 45,
                    "text": "",
                    "confidence": 0.9,
                    "source": "PP-OCRv5_mobile_det",
                    "polygon": [(15, 15), (70, 15), (70, 45), (15, 45)],
                },
                {
                    "left": 15,
                    "top": 75,
                    "right": 75,
                    "bottom": 105,
                    "text": "",
                    "confidence": 0.8,
                    "source": "PP-OCRv5_mobile_det",
                    "polygon": [(15, 75), (75, 75), (75, 105), (15, 105)],
                },
            ]

    result = reading_service.process_image(
        image,
        text_detector=FakeDetector(),
        preprocessing_method="grayscale",
    )

    assert calls[0] == (120, 160, 11)
    assert len(calls) == 3
    assert all(call[2] == 6 for call in calls[1:])
    assert result["recognized_text"] == "पहली\nदूसरी"
    assert len(result["detected_regions"]) == 2
    assert len(result["region_crops"]) == 2
    assert result["crop_box"] == (0, 0, 160, 120)
    assert result["detection_source"] == "PP-OCRv5_mobile_det"
    assert result["region_crops"][0]["region_ocr_text"] == "क्षेत्र"


def test_hybrid_recognition_combines_ordered_regions_and_validates_speech(
    tmp_path,
    monkeypatch,
) -> None:
    import cv2

    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    image[20:40, 10:70] = 100
    image[80:100, 10:70] = 200
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )

    class FixedOCR:
        def __init__(self, page_segmentation_mode=11):
            self.page_segmentation_mode = page_segmentation_mode

        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            if self.page_segmentation_mode == 11:
                text = "पूर्ण पाठ"
            else:
                text = "पहली पंक्ति" if candidate.mean() < 180 else "दूसरी पंक्ति"
            return {
                "text": text,
                "confidence": 80.0,
                "word_confidences": [80.0],
                "word_count": 1,
                "word_details": [],
            }

    class FakeDetector:
        def detect(self, candidate):
            return [
                {"left": 10, "top": 80, "right": 70, "bottom": 100, "confidence": 0.9},
                {"left": 10, "top": 20, "right": 70, "bottom": 40, "confidence": 0.9},
            ]

    class FakePaddle:
        device = "cpu"

        def recognize(self, candidate):
            text = "पहली पंक्ति" if candidate.mean() < 180 else "दूसरी पंक्ति"
            return {"text": text, "confidence": 0.95}

    spoken = []

    def fake_synthesis(text, output_path):
        spoken.append(text)
        return True, output_path

    monkeypatch.setattr(reading_service, "TesseractOCR", FixedOCR)
    monkeypatch.setattr(reading_service, "generate_hindi_audio", fake_synthesis)

    result = reading_service.process_image(
        image,
        input_kind="webcam",
        generate_audio=True,
        text_detector=FakeDetector(),
        paddle_recognizer=FakePaddle(),
    )

    assert result["recognized_text"] == "पहली पंक्ति\nदूसरी पंक्ति"
    assert result["recognition_engine"] == "PaddleOCR-Devanagari with Tesseract fallback"
    assert result["speech_candidates_validated"] is True
    assert result["audio_path"] is not None
    assert spoken == [result["recognized_text"]]
    assert [region["top"] for region in result["detected_regions"]] == [20, 80]
    assert all(
        region["selected_recognizer"] == "PaddleOCR-Devanagari"
        for region in result["region_crops"]
    )


def test_mismatched_recognizer_candidates_are_not_spoken(tmp_path, monkeypatch) -> None:
    import cv2

    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )

    class FixedOCR:
        def __init__(self, page_segmentation_mode=11):
            self.page_segmentation_mode = page_segmentation_mode

        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            return {
                "text": "तंगी कहाँ?",
                "confidence": 80.0,
                "word_confidences": [80.0],
                "word_count": 1,
                "word_details": [],
            }

    class FakeDetector:
        def detect(self, candidate):
            return [
                {
                    "left": 5,
                    "top": 5,
                    "right": 100,
                    "bottom": 50,
                    "confidence": 0.9,
                }
            ]

    class FakePaddle:
        def recognize(self, candidate):
            return {"text": "गलत पाठ", "confidence": 0.99}

    def fail_if_synthesized(*args, **kwargs):
        pytest.fail("Unconfirmed recognizer output must not be sent to Piper")

    monkeypatch.setattr(reading_service, "TesseractOCR", FixedOCR)
    monkeypatch.setattr(reading_service, "generate_hindi_audio", fail_if_synthesized)

    result = reading_service.process_image(
        image,
        generate_audio=True,
        text_detector=FakeDetector(),
        paddle_recognizer=FakePaddle(),
    )

    assert result["recognized_text"] == "गलत पाठ"
    assert result["speech_candidates_validated"] is False
    assert result["audio_path"] is None
    assert "did not independently confirm" in result["tts_skipped_reason"]


def test_full_frame_tesseract_line_can_confirm_a_paddle_region(
    tmp_path,
    monkeypatch,
) -> None:
    import cv2

    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )

    full_frame_words = [
        {
            "text": "तंगी",
            "confidence": 94.0,
            "left": 20,
            "top": 25,
            "width": 35,
            "height": 20,
            "block_num": "1",
            "par_num": "1",
            "line_num": "1",
        },
        {
            "text": "कहाँ",
            "confidence": 94.0,
            "left": 60,
            "top": 25,
            "width": 35,
            "height": 20,
            "block_num": "1",
            "par_num": "1",
            "line_num": "1",
        },
    ]

    class FixedOCR:
        def __init__(self, page_segmentation_mode=11):
            self.page_segmentation_mode = page_segmentation_mode

        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            if self.page_segmentation_mode == 11:
                return {
                    "text": "तंगी कहाँ",
                    "confidence": 94.0,
                    "word_confidences": [94.0, 94.0],
                    "word_count": 2,
                    "word_details": full_frame_words,
                }
            return {
                "text": "",
                "confidence": None,
                "word_confidences": [],
                "word_count": 0,
                "word_details": [],
            }

    class FakeDetector:
        def detect(self, candidate):
            return [
                {
                    "left": 15,
                    "top": 20,
                    "right": 100,
                    "bottom": 55,
                    "confidence": 0.9,
                }
            ]

    class FakePaddle:
        def recognize(self, candidate):
            return {"text": "तंगी कहाँ?", "confidence": 0.92}

    spoken = []

    def fake_synthesis(text, output_path):
        spoken.append(text)
        return True, output_path

    monkeypatch.setattr(reading_service, "TesseractOCR", FixedOCR)
    monkeypatch.setattr(reading_service, "generate_hindi_audio", fake_synthesis)

    result = reading_service.process_image(
        image,
        generate_audio=True,
        text_detector=FakeDetector(),
        paddle_recognizer=FakePaddle(),
    )

    assert result["recognized_text"] == "तंगी कहाँ?"
    assert result["region_crops"][0]["speech_validation_source"] == "full-image Tesseract line"
    assert result["speech_candidates_validated"] is True
    assert spoken == ["तंगी कहाँ?"]


def test_streamlit_read_page_does_not_offer_manual_cropping() -> None:
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app_test = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app_test.exception
    assert not any("crop" in checkbox.label.lower() for checkbox in app_test.checkbox)


def test_large_image_is_downscaled_for_ocr_but_original_is_preserved(
    tmp_path,
    monkeypatch,
) -> None:
    import cv2

    image = np.full((2200, 3100, 3), 127, dtype=np.uint8)
    original = image.copy()
    monkeypatch.setattr(reading_service, "UPLOADS_DIR", tmp_path / "uploads")
    monkeypatch.setattr(
        reading_service,
        "preprocess_for_tesseract",
        lambda candidate, method="auto": cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )

    class EmptyOCR:
        def check_available(self):
            return {"version": "test"}

        def recognize(self, candidate):
            assert max(candidate.shape[:2]) <= reading_service.MAX_OCR_IMAGE_SIDE
            return {
                "text": "",
                "confidence": None,
                "word_confidences": [],
                "word_count": 0,
                "word_details": [],
            }

    result = reading_service.process_image(image, ocr=EmptyOCR())

    assert max(result["ocr_input_shape"][:2]) == reading_service.MAX_OCR_IMAGE_SIDE
    assert result["original_image"].shape == original.shape
    assert np.array_equal(result["original_image"], original)
    assert np.array_equal(image, original)


def test_tesseract_returns_word_level_confidence_and_boxes() -> None:
    import cv2

    from config.settings import ROOT_DIR

    image_path = (
        ROOT_DIR
        / "sample_data"
        / "val"
        / "images"
        / "13042023_L_GH012834_image_000102_15.png"
    )
    result = TesseractOCR(page_segmentation_mode=6).recognize(
        cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    )

    assert "word_details" in result
    assert result["word_details"]
    assert {"text", "confidence", "left", "top", "width", "height", "uncertain"} <= set(
        result["word_details"][0]
    )


def test_unreadable_upload_is_rejected() -> None:
    with pytest.raises(ValueError, match="not a readable image"):
        _decode_image(b"not an image")


def test_generated_piper_audio_is_readable(tmp_path) -> None:
    from config.settings import PIPER_VOICE_PATH

    if not PIPER_VOICE_PATH.is_file() or not PIPER_VOICE_PATH.with_suffix(".onnx.json").is_file():
        pytest.skip("Configured Piper voice files are not installed.")

    output = tmp_path / "spoken_hindi.wav"
    ok, value = generate_hindi_audio("नमस्ते।", output)

    assert ok, value
    with wave.open(value, "rb") as audio:
        assert audio.getnframes() > 0
        assert audio.getframerate() > 0
