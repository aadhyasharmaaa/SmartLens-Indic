from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from config.settings import DEFAULT_IMAGE_PATH
from services.reading_service import process_image


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="SmartLens-Indic Tesseract Hindi OCR and Piper demo")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE_PATH, help="Path to a Hindi image")
    parser.add_argument("--tts", action="store_true", help="Generate offline Hindi speech with Piper")
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error: image could not be opened: {args.image}", file=sys.stderr)
        return 2

    try:
        result = process_image(image, input_kind="upload", generate_audio=args.tts)
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Input image: {args.image}")
    print(f"Tesseract version: {result['tesseract_version']}")
    print(f"Recognized Hindi text:\n{result['recognized_text'] or '(No Hindi text recognized)'}")
    print(f"Mean word confidence: {result['confidence_mean']}")
    print(
        f"Text detection: {result['detection_source']} "
        f"({len(result['detected_regions'])} regions, {result['text_detection_ms']:.2f} ms)"
    )
    if result["detection_warning"]:
        print(f"Text detection warning: {result['detection_warning']}", file=sys.stderr)
    print(f"Preprocessing time: {result['preprocess_ms']:.2f} ms")
    print(f"Region preprocessing time: {result['region_preprocess_ms']:.2f} ms")
    print(f"OCR time: {result['ocr_ms']:.2f} ms")
    if result["tts_ms"] is not None:
        print(f"TTS time: {result['tts_ms']:.2f} ms")
        print(f"Audio file: {result['audio_path']}")
        if result["error_message"]:
            print(f"TTS error: {result['error_message']}", file=sys.stderr)
    print(f"Total time: {result['total_ms']:.2f} ms")
    return 1 if result["status"] == "tts_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
