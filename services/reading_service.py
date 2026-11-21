from __future__ import annotations

import logging
import math
import time
import unicodedata
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config.settings import (
    PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR,
    READINGS_DIR,
    UPLOADS_DIR,
)
from postprocessing.text_cleanup import clean_ocr_text
from preprocessing.preprocess import (
    preprocess_for_tesseract,
    resolve_tesseract_preprocessing_method,
    resize_to_max_side,
)
from recognition.paddle_devanagari_recognizer import (
    PaddleDevanagariRecognizer,
    get_paddle_devanagari_recognizer,
)
from recognition.tesseract_ocr import TesseractOCR
from recognition.paddle_text_detector import PaddleTextDetector, get_text_detector
from recognition.text_regions import group_word_regions
from tts.piper_tts import generate_hindi_audio

MAX_REGION_OCR_RETRIES = 24
MAX_OCR_IMAGE_SIDE = 2000
MIN_RECOGNITION_CROP_HEIGHT = 48
MAX_RECOGNITION_CROP_SCALE = 2.0
logger = logging.getLogger(__name__)


def _has_devanagari_text(text: str) -> bool:
    return any(
        "\u0900" <= character <= "\u097f"
        and unicodedata.category(character)[0] in {"L", "N"}
        for character in text
    )


def _recognition_match_key(text: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFC", text)
        if unicodedata.category(character)[0] in {"L", "N"}
    )


def _overlap_ratio(first: dict[str, Any], second: dict[str, Any]) -> float:
    intersection_width = max(
        0,
        min(first["right"], second["right"]) - max(first["left"], second["left"]),
    )
    intersection_height = max(
        0,
        min(first["bottom"], second["bottom"]) - max(first["top"], second["top"]),
    )
    intersection = intersection_width * intersection_height
    first_area = max(0, first["right"] - first["left"]) * max(
        0, first["bottom"] - first["top"]
    )
    second_area = max(0, second["right"] - second["left"]) * max(
        0, second["bottom"] - second["top"]
    )
    smaller_area = min(first_area, second_area)
    return intersection / smaller_area if smaller_area else 0.0


def _matching_tesseract_line(
    detected_region: dict[str, Any],
    ocr_word_regions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matches = [
        (candidate, _overlap_ratio(detected_region, candidate))
        for candidate in ocr_word_regions
    ]
    matches = [(candidate, overlap) for candidate, overlap in matches if overlap >= 0.3]
    return max(matches, key=lambda match: match[1])[0] if matches else None


def _prepare_region_crop(
    image: np.ndarray,
    region: dict[str, Any],
) -> np.ndarray:
    height, width = image.shape[:2]
    left = max(0, int(region["left"]))
    top = max(0, int(region["top"]))
    right = min(width, int(region["right"]))
    bottom = min(height, int(region["bottom"]))
    if left >= right or top >= bottom:
        raise ValueError(f"Detected text region has invalid bounds: {region!r}")

    crop = image[top:bottom, left:right]
    padding = max(2, round((bottom - top) * 0.1))
    crop = cv2.copyMakeBorder(
        crop,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    scale = min(
        MAX_RECOGNITION_CROP_SCALE,
        max(1.0, MIN_RECOGNITION_CROP_HEIGHT / crop.shape[0]),
    )
    if scale > 1:
        crop = cv2.resize(
            crop,
            (round(crop.shape[1] * scale), round(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def process_image(
    image: np.ndarray,
    *,
    input_kind: str = "upload",
    generate_audio: bool = False,
    ocr: TesseractOCR | None = None,
    text_detector: PaddleTextDetector | None = None,
    paddle_recognizer: PaddleDevanagariRecognizer | None = None,
    crop_box: tuple[int, int, int, int] | None = None,
    preprocessing_method: str = "auto",
    page_segmentation_mode: int | None = None,
) -> dict[str, Any]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("The supplied image is empty or invalid.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel uint8 BGR image.")

    height, width = image.shape[:2]
    if crop_box is not None:
        left, top, right, bottom = crop_box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(
                f"Crop box {crop_box} must be inside the image dimensions {width}x{height}."
            )
        selected_region = image[top:bottom, left:right].copy()
    else:
        selected_region = image.copy()
        crop_box = (0, 0, width, height)

    actual_preprocessing_method = resolve_tesseract_preprocessing_method(
        selected_region, preprocessing_method
    )
    ocr_input_image = resize_to_max_side(selected_region, max_side=MAX_OCR_IMAGE_SIDE)
    scale_x = selected_region.shape[1] / ocr_input_image.shape[1]
    scale_y = selected_region.shape[0] / ocr_input_image.shape[0]
    if page_segmentation_mode is None:
        actual_page_segmentation_mode = 6 if max(selected_region.shape[:2]) <= 64 else 11
    else:
        if page_segmentation_mode not in (6, 11):
            raise ValueError("Choose Tesseract page segmentation mode 6 or 11.")
        actual_page_segmentation_mode = page_segmentation_mode

    session_id = uuid.uuid4().hex
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    original_path = UPLOADS_DIR / f"{session_id}.png"
    if not cv2.imwrite(str(original_path), image):
        raise OSError(f"Could not save the original image copy: {original_path}")

    total_start = time.perf_counter()
    preprocess_start = time.perf_counter()
    tesseract_image = preprocess_for_tesseract(
        ocr_input_image, method=actual_preprocessing_method
    )
    preprocess_ms = (time.perf_counter() - preprocess_start) * 1000

    preprocessed_path = UPLOADS_DIR / f"{session_id}_preprocessed.png"
    if not cv2.imwrite(str(preprocessed_path), tesseract_image):
        raise OSError(f"Could not save the preprocessed image: {preprocessed_path}")

    engine = ocr or TesseractOCR(page_segmentation_mode=actual_page_segmentation_mode)
    availability = engine.check_available()
    ocr_start = time.perf_counter()
    recognition = engine.recognize(tesseract_image)
    full_frame_ocr_ms = (time.perf_counter() - ocr_start) * 1000
    region_ocr_ms = 0.0
    region_preprocess_ms = 0.0
    text = clean_ocr_text(recognition["text"])
    ocr_word_regions = group_word_regions(
        recognition.get("word_details", []),
        ocr_input_image.shape,
    )
    if ocr_word_regions:
        text = clean_ocr_text("\n".join(region["text"] for region in ocr_word_regions))

    detection_start = time.perf_counter()
    detection_warning: str | None = None
    try:
        detector = text_detector or get_text_detector()
        detected_regions = detector.detect(ocr_input_image)
    except Exception as exc:
        logger.exception("PP-OCRv5 text detection failed; using Tesseract OCR boxes")
        detected_regions = []
        detection_warning = (
            f"PP-OCRv5 text detection was unavailable ({exc}); showing Tesseract OCR "
            "word boxes instead."
        )
    text_detection_ms = (time.perf_counter() - detection_start) * 1000
    if not detected_regions:
        if detection_warning is None:
            detection_warning = (
                "PP-OCRv5 found no text regions; showing Tesseract OCR word boxes instead."
            )
        detected_regions = ocr_word_regions
        detection_source = "Tesseract OCR fallback"
    else:
        detection_source = "PP-OCRv5_mobile_det"
    detected_regions = sorted(
        detected_regions,
        key=lambda region: (region["top"], region["left"]),
    )

    paddle_engine = paddle_recognizer
    paddle_warning: str | None = None
    if (
        paddle_engine is None
        and detected_regions
        and len(detected_regions) <= MAX_REGION_OCR_RETRIES
        and (PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR / "inference.yml").is_file()
    ):
        try:
            paddle_engine = get_paddle_devanagari_recognizer()
        except Exception as exc:
            logger.exception("PaddleOCR Devanagari recognizer could not be initialized")
            paddle_warning = (
                f"The optional Devanagari recognizer could not be loaded ({exc}); "
                "using Tesseract Hindi OCR."
            )
    elif paddle_engine is None:
        paddle_warning = (
            "The local Devanagari recognizer model is unavailable; using Tesseract Hindi OCR."
        )

    region_crops: list[dict[str, Any]] = []
    paddle_region_text_used = False
    paddle_speech_validated = True
    selected_region_texts: list[str] = []
    for index, region in enumerate(detected_regions):
        if index >= MAX_REGION_OCR_RETRIES:
            region["region_ocr_ms"] = 0.0
            region["region_ocr_text"] = ""
            region["region_ocr_confidence"] = None
            region["region_ocr_skipped"] = True
            region["paddle_ocr_text"] = ""
            region["paddle_ocr_confidence"] = None
            region["selected_text"] = ""
            region["selected_recognizer"] = "not processed"
            region["speech_validated"] = False
            continue

        region_crop = _prepare_region_crop(ocr_input_image, region)
        region_preprocess_start = time.perf_counter()
        region_preprocessed = preprocess_for_tesseract(
            region_crop,
            method=actual_preprocessing_method,
        )
        region["preprocessing_ms"] = (time.perf_counter() - region_preprocess_start) * 1000
        region_preprocess_ms += region["preprocessing_ms"]
        region_ocr_start = time.perf_counter()
        region_recognition = (
            engine.recognize(region_preprocessed)
            if ocr is not None
            else TesseractOCR(page_segmentation_mode=6).recognize(region_preprocessed)
        )
        region["region_ocr_ms"] = (time.perf_counter() - region_ocr_start) * 1000
        region_ocr_ms += region["region_ocr_ms"]
        region["region_ocr_text"] = clean_ocr_text(region_recognition["text"])
        region["region_ocr_confidence"] = region_recognition["confidence"]
        region["paddle_ocr_text"] = ""
        region["paddle_ocr_confidence"] = None
        region["paddle_ocr_ms"] = 0.0
        region["speech_validated"] = False
        region["selected_text"] = region["region_ocr_text"]
        region["selected_recognizer"] = "Tesseract"

        if paddle_engine is not None:
            paddle_start = time.perf_counter()
            try:
                paddle_result = paddle_engine.recognize(region_crop)
                region["paddle_ocr_text"] = clean_ocr_text(paddle_result["text"])
                region["paddle_ocr_confidence"] = paddle_result["confidence"]
            except Exception as exc:
                logger.exception("PaddleOCR failed on detected region %s", index + 1)
                paddle_warning = (
                    f"PaddleOCR failed on text region {index + 1} ({exc}); "
                    "Tesseract was used for that region."
                )
                paddle_engine = None
            finally:
                region["paddle_ocr_ms"] = (time.perf_counter() - paddle_start) * 1000
                region_ocr_ms += region["paddle_ocr_ms"]

        if _has_devanagari_text(region["paddle_ocr_text"]):
            region["selected_text"] = region["paddle_ocr_text"]
            region["selected_recognizer"] = "PaddleOCR-Devanagari"
            paddle_region_text_used = True
            tess_text = region["region_ocr_text"]
            tess_confidence = region["region_ocr_confidence"]
            matching_line = _matching_tesseract_line(region, ocr_word_regions)
            if (
                not _has_devanagari_text(tess_text)
                or _recognition_match_key(tess_text)
                != _recognition_match_key(region["paddle_ocr_text"])
            ) and matching_line is not None:
                tess_text = matching_line["text"]
                tess_confidence = matching_line["confidence"]
                region["speech_validation_source"] = "full-image Tesseract line"
            else:
                region["speech_validation_source"] = "text-region Tesseract OCR"
            region["speech_validation_confidence"] = tess_confidence
            region["speech_validated"] = bool(
                _has_devanagari_text(tess_text)
                and tess_confidence is not None
                and math.isfinite(float(tess_confidence))
                and float(tess_confidence) > 0
                and _recognition_match_key(tess_text)
                == _recognition_match_key(region["paddle_ocr_text"])
            )
            if not region["speech_validated"]:
                paddle_speech_validated = False
        else:
            region["speech_validation_source"] = "text-region Tesseract OCR"
            region["speech_validation_confidence"] = region["region_ocr_confidence"]
            region["speech_validated"] = (
                bool(region["region_ocr_text"])
                and region["region_ocr_confidence"] is not None
                and math.isfinite(float(region["region_ocr_confidence"]))
                and float(region["region_ocr_confidence"]) > 0
            )
        if region["selected_text"]:
            selected_region_texts.append(region["selected_text"])

        region_crops.append(
            {
                "index": index + 1,
                "image": region_crop,
                "preprocessed_image": region_preprocessed,
                "text": region.get("text", ""),
                "confidence": region["confidence"],
                "region_ocr_text": region["region_ocr_text"],
                "region_ocr_confidence": region["region_ocr_confidence"],
                "region_ocr_ms": region["region_ocr_ms"],
                "paddle_ocr_text": region["paddle_ocr_text"],
                "paddle_ocr_confidence": region["paddle_ocr_confidence"],
                "paddle_ocr_ms": region["paddle_ocr_ms"],
                "selected_text": region["selected_text"],
                "selected_recognizer": region["selected_recognizer"],
                "speech_validated": region["speech_validated"],
                "speech_validation_source": region["speech_validation_source"],
                "speech_validation_confidence": region["speech_validation_confidence"],
                "region_ocr_skipped": False,
                "source": region.get("source", detection_source),
                "polygon": region.get("polygon"),
                "bounds": (
                    region["left"],
                    region["top"],
                    region["right"],
                    region["bottom"],
                ),
            }
        )

    if (
        paddle_region_text_used
        and len(detected_regions) <= MAX_REGION_OCR_RETRIES
        and selected_region_texts
    ):
        text = clean_ocr_text("\n".join(selected_region_texts))
        recognition_engine = "PaddleOCR-Devanagari with Tesseract fallback"
        speech_candidates_validated = (
            paddle_speech_validated
            and len(region_crops) == len(detected_regions)
            and all(
                region["speech_validated"] and bool(region["selected_text"])
                for region in detected_regions
            )
        )
    else:
        text = clean_ocr_text(recognition["text"])
        if ocr_word_regions:
            text = clean_ocr_text("\n".join(region["text"] for region in ocr_word_regions))
        recognition_engine = "Tesseract"
        speech_candidates_validated = True

    if crop_box is not None and crop_box != (0, 0, width, height):
        crop_left, crop_top, _, _ = crop_box
        for region in detected_regions:
            region["left"] = round(region["left"] * scale_x) + crop_left
            region["right"] = round(region["right"] * scale_x) + crop_left
            region["top"] = round(region["top"] * scale_y) + crop_top
            region["bottom"] = round(region["bottom"] * scale_y) + crop_top
            if region.get("polygon"):
                region["polygon"] = [
                    (round(x * scale_x) + crop_left, round(y * scale_y) + crop_top)
                    for x, y in region["polygon"]
                ]
            if region.get("polygons"):
                region["polygons"] = [
                    [
                        (round(x * scale_x) + crop_left, round(y * scale_y) + crop_top)
                        for x, y in polygon
                    ]
                    for polygon in region["polygons"]
                ]
        for region_crop in region_crops:
            left, top, right, bottom = region_crop["bounds"]
            region_crop["bounds"] = (
                round(left * scale_x) + crop_left,
                round(top * scale_y) + crop_top,
                round(right * scale_x) + crop_left,
                round(bottom * scale_y) + crop_top,
            )
    else:
        for region in detected_regions:
            region["left"] = round(region["left"] * scale_x)
            region["right"] = round(region["right"] * scale_x)
            region["top"] = round(region["top"] * scale_y)
            region["bottom"] = round(region["bottom"] * scale_y)
            if region.get("polygon"):
                region["polygon"] = [
                    (round(x * scale_x), round(y * scale_y))
                    for x, y in region["polygon"]
                ]
            if region.get("polygons"):
                region["polygons"] = [
                    [
                        (round(x * scale_x), round(y * scale_y))
                        for x, y in polygon
                    ]
                    for polygon in region["polygons"]
                ]
        for region_crop in region_crops:
            left, top, right, bottom = region_crop["bounds"]
            region_crop["bounds"] = (
                round(left * scale_x),
                round(top * scale_y),
                round(right * scale_x),
                round(bottom * scale_y),
            )
    ocr_ms = full_frame_ocr_ms + region_ocr_ms

    audio_path: Path | None = None
    tts_ms: float | None = None
    error_message: str | None = None
    tts_skipped_reason: str | None = None
    status = "success" if text else "empty_text"
    if generate_audio and text:
        confidence = recognition["confidence"]
        if paddle_region_text_used and not speech_candidates_validated:
            tts_skipped_reason = (
                "Speech was not generated because Tesseract did not independently confirm "
                "every detected text region. Review the displayed text before using it."
            )
        elif confidence is None or not math.isfinite(confidence) or confidence <= 0:
            tts_skipped_reason = (
                "Speech was not generated because Tesseract confidence was zero, invalid, "
                "or unavailable. Try a sharper, closer, or better-lit image."
            )
        elif not _has_devanagari_text(text):
            tts_skipped_reason = (
                "Speech was not generated because OCR returned no Devanagari letters or numerals. "
                "Try an image containing clearer Hindi text."
            )
        else:
            READINGS_DIR.mkdir(parents=True, exist_ok=True)
            requested_audio_path = READINGS_DIR / f"{session_id}.wav"
            tts_start = time.perf_counter()
            audio_ok, audio_result = generate_hindi_audio(
                text, output_path=requested_audio_path
            )
            tts_ms = (time.perf_counter() - tts_start) * 1000
            if not audio_ok:
                status = "tts_failed"
                error_message = f"Piper speech generation failed: {audio_result}"
            else:
                audio_path = Path(audio_result)

    total_ms = (time.perf_counter() - total_start) * 1000
    return {
        "id": session_id,
        "status": status,
        "input_kind": input_kind,
        "original_image": image.copy(),
        "selected_region_image": selected_region,
        "crop_box": crop_box,
        "ocr_input_shape": ocr_input_image.shape,
        "ocr_input_scale": (scale_x, scale_y),
        "preprocessed_image": tesseract_image,
        "original_image_path": original_path,
        "preprocessed_image_path": preprocessed_path,
        "recognized_text": text,
        "recognition_engine": recognition_engine,
        "confidence_mean": recognition["confidence"],
        "word_confidences": recognition["word_confidences"],
        "word_details": recognition.get("word_details", []),
        "detected_regions": detected_regions,
        "detection_source": detection_source,
        "detection_warning": detection_warning,
        "recognition_warning": paddle_warning,
        "speech_candidates_validated": speech_candidates_validated,
        "region_crops": region_crops,
        "word_count": recognition["word_count"],
        "preprocessing_method": actual_preprocessing_method,
        "page_segmentation_mode": actual_page_segmentation_mode,
        "tesseract_version": availability["version"],
        "preprocess_ms": preprocess_ms,
        "text_detection_ms": text_detection_ms,
        "region_preprocess_ms": region_preprocess_ms,
        "ocr_ms": ocr_ms,
        "full_frame_ocr_ms": full_frame_ocr_ms,
        "region_ocr_ms": region_ocr_ms,
        "region_ocr_retry_limit": MAX_REGION_OCR_RETRIES,
        "regions_not_retried": max(0, len(detected_regions) - MAX_REGION_OCR_RETRIES),
        "tts_ms": tts_ms,
        "tts_skipped_reason": tts_skipped_reason,
        "total_ms": total_ms,
        "audio_path": audio_path,
        "error_message": error_message,
    }
