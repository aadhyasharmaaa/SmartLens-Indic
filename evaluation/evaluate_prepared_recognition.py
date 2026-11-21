from __future__ import annotations

import argparse
import csv
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR
from evaluation.evaluate_ocr import character_error_rate, word_error_rate
from postprocessing.normalize_unicode import normalize_unicode_text
from preprocessing.preprocess import preprocess_for_tesseract
from recognition.paddle_devanagari_recognizer import (
    get_paddle_devanagari_recognizer,
)
from recognition.tesseract_ocr import TesseractOCR


FIELDS = (
    "split",
    "image_path",
    "recognizer",
    "reference_text",
    "recognized_text",
    "cer",
    "wer",
    "exact_match",
    "confidence",
    "confidence_type",
    "processing_ms",
    "error",
)


def _normalized_for_exact_match(text: str) -> str:
    normalized = normalize_unicode_text(text)
    return " ".join(normalized.split())


def _recognize_one(
    *,
    image: np.ndarray,
    reference: str,
    recognizer_name: str,
    recognizer: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    error: str | None = None
    try:
        if recognizer_name == "Tesseract":
            candidate = preprocess_for_tesseract(image, method="auto")
            output = recognizer.recognize(candidate)
        else:
            output = recognizer.recognize(image)
        recognized = normalize_unicode_text(str(output.get("text", ""))).strip()
        confidence = output.get("confidence")
        confidence_type = output.get(
            "confidence_type",
            "Tesseract word-confidence mean (diagnostic)"
            if recognizer_name == "Tesseract"
            else "Paddle model score (not calibrated)",
        )
    except Exception as exc:
        recognized = ""
        confidence = None
        confidence_type = "unavailable"
        error = f"{type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "recognized_text": recognized,
        "cer": character_error_rate(reference, recognized) if error is None else None,
        "wer": word_error_rate(reference, recognized) if error is None else None,
        "exact_match": (
            _normalized_for_exact_match(reference)
            == _normalized_for_exact_match(recognized)
            if error is None
            else False
        ),
        "confidence": confidence,
        "confidence_type": confidence_type,
        "processing_ms": round(elapsed_ms, 2),
        "error": error,
    }


def evaluate(split: str, limit: int | None, output_path: Path) -> None:
    manifest_path = ROOT_DIR / "dataset_work" / "recognition" / f"{split}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared recognition manifest not found: {manifest_path}")

    with manifest_path.open(encoding="utf-8", newline="") as manifest:
        rows = list(csv.DictReader(manifest))
    if not rows:
        raise ValueError(f"Prepared recognition manifest is empty: {manifest_path}")
    if limit is not None:
        rows = rows[:limit]

    archives: dict[str, zipfile.ZipFile] = {}
    summary: dict[str, list[dict[str, Any]]] = defaultdict(list)
    try:
        paddle_recognizer = get_paddle_devanagari_recognizer()
        tesseract = TesseractOCR(page_segmentation_mode=6)
        tesseract.check_available()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                archive_path = str((ROOT_DIR / row["archive_path"]).resolve())
                archive = archives.get(archive_path)
                if archive is None:
                    archive = zipfile.ZipFile(archive_path)
                    archives[archive_path] = archive
                image_bytes = archive.read(row["archive_member"])
                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise ValueError(f"Could not decode {row['archive_member']!r}")

                for recognizer_name, recognizer in (
                    ("Tesseract", tesseract),
                    ("PaddleOCR-Devanagari", paddle_recognizer),
                ):
                    result = _recognize_one(
                        image=image,
                        reference=row["transcription"],
                        recognizer_name=recognizer_name,
                        recognizer=recognizer,
                    )
                    writer.writerow(
                        {
                            "split": split,
                            "image_path": row["image_path"],
                            "recognizer": recognizer_name,
                            "reference_text": row["transcription"],
                            **result,
                        }
                    )
                    summary[recognizer_name].append(result | {"reference": row["transcription"]})

                if index % 100 == 0 or index == len(rows):
                    output_file.flush()
                    print(f"Evaluated {index}/{len(rows)} word images", flush=True)
    finally:
        for archive in archives.values():
            archive.close()

    print(f"\nEvaluation CSV: {output_path}")
    print(f"Split: {split}; images: {len(rows)}")
    for recognizer_name, results in summary.items():
        successful = [item for item in results if item["error"] is None]
        exact = sum(bool(item["exact_match"]) for item in successful)
        print(
            f"{recognizer_name}: "
            f"successful={len(successful)}/{len(results)}, "
            f"mean_CER={mean(item['cer'] for item in successful):.4f}, "
            f"mean_WER={mean(item['wer'] for item in successful):.4f}, "
            f"exact_match={exact}/{len(successful)} "
            f"({exact / len(successful):.4%}), "
            f"median_ms={median(item['processing_ms'] for item in successful):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Tesseract and the local Devanagari recognizer on a prepared "
            "labeled word-image split without extracting archive files."
        )
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Prepared labeled split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum image count for a quick smoke test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV report path (defaults under evaluation/results/).",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not (PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR / "inference.yml").is_file():
        raise FileNotFoundError(
            f"Local Devanagari recognizer is missing: "
            f"{PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR}"
        )
    output_path = args.output or (
        ROOT_DIR / "evaluation" / "results" / f"{args.split}_recognition.csv"
    )
    evaluate(args.split, args.limit, output_path)


if __name__ == "__main__":
    main()
