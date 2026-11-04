from __future__ import annotations

import argparse
import os
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config.settings import DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_IMAGE_PATH, OUTPUT_DIR, TEXT_DETECTION_MODEL_NAME, TEXT_RECOGNITION_MODEL_NAME
from detection.detector import Detector
from postprocessing.confidence_filter import filter_by_confidence
from postprocessing.normalize_unicode import normalize_unicode_text
from postprocessing.reading_order import order_by_reading_boxes
from preprocessing.preprocess import preprocess_image
from recognition.recognizer import DevanagariRecognizer
from tts.piper_tts import generate_hindi_audio


def _draw_boxes(image, boxes):
    boxed = image.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(boxed, str(idx), (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return boxed


def _sort_recognized_results(results):
    return sorted(results, key=lambda item: (item["box"][1], item["box"][0]))


def run_end_to_end_pipeline(
    image_path: str | Path,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    tts_enabled: bool = False,
):
    image_path = Path(image_path)
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    processed = preprocess_image(image_path)
    cv2.imwrite(str(OUTPUT_DIR / "preprocessed_demo.png"), processed)
    cv2.imwrite(str(OUTPUT_DIR / "original_demo.png"), original)

    print(f"Input image: {image_path}")
    print(f"Preprocessed image saved to: {OUTPUT_DIR / 'preprocessed_demo.png'}")

    detector = Detector(detector_model_name=TEXT_DETECTION_MODEL_NAME)
    boxes = detector.detect(processed)
    print(f"Detected text regions: {len(boxes)}")

    if boxes:
        ordered_boxes = order_by_reading_boxes(boxes)
        boxed_image = _draw_boxes(processed, ordered_boxes)
        cv2.imwrite(str(OUTPUT_DIR / "detected_regions.png"), boxed_image)
        print(f"Detected regions image saved to: {OUTPUT_DIR / 'detected_regions.png'}")
    else:
        ordered_boxes = []
        print("No text regions were detected by the PP-OCRv5 detector on this image.")

    recognizer = DevanagariRecognizer(model_name=TEXT_RECOGNITION_MODEL_NAME)
    raw_results = recognizer.recognize(processed, boxes)
    filtered = filter_by_confidence(raw_results, threshold=confidence_threshold)
    ordered = _sort_recognized_results(filtered)
    normalized = []
    for item in ordered:
        text = normalize_unicode_text(str(item.get("text", "")).strip())
        if not text:
            continue
        normalized.append({**item, "text": text})

    print("Recognized output:")
    for item in normalized:
        print(f"Box={item['box']} | Text={item['text']} | Confidence={item['confidence']:.4f}")

    final_text = " ".join(item["text"] for item in normalized).strip()
    print(f"Final Hindi text: {final_text}")

    if tts_enabled:
        if not final_text:
            print("OCR returned empty Hindi text; audio generation skipped.")
        else:
            ok, message = generate_hindi_audio(
                final_text,
                output_path=OUTPUT_DIR / "spoken_hindi.wav",
                data_dir=r"C:\Users\Aadhya\Desktop\Final MP\models\voices",
                model_name="hi_IN-rohan-medium",
            )
            if ok:
                print(f"Hindi audio generated: {message}")
            else:
                print(f"Hindi audio generation failed: {message}")

    return {
        "input_path": str(image_path),
        "boxes": ordered_boxes,
        "recognized": normalized,
        "final_text": final_text,
    }


def run_demo(image_path: str | Path):
    image_path = Path(image_path)
    result = run_end_to_end_pipeline(image_path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartLens-Indic OCR pipeline")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE_PATH), help="Path to a local Hindi image")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="Recognition confidence threshold")
    parser.add_argument("--tts", action="store_true", help="Generate Hindi audio from the final OCR text using Piper hi_IN-rohan-medium")
    args = parser.parse_args()
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
    run_end_to_end_pipeline(args.image, confidence_threshold=args.threshold, tts_enabled=args.tts)
