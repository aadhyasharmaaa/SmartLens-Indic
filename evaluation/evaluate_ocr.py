from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import ROOT_DIR as PROJECT_ROOT
from postprocessing.normalize_unicode import normalize_unicode_text
from preprocessing.preprocess import (
    TESSERACT_PREPROCESSING_METHODS,
    preprocess_for_tesseract,
    resolve_tesseract_preprocessing_method,
    resize_to_max_side,
)
from recognition.paddle_text_detector import get_text_detector
from recognition.tesseract_ocr import TesseractOCR

MAX_OCR_IMAGE_SIDE = 2000
CROP_PADDING_RATIO = 0.1
UPSCALE_CROP_HEIGHT_BELOW = 80
REPORT_FIELDS = (
    "track",
    "image",
    "scope",
    "region_index",
    "reference_text",
    "recognized_text",
    "cer",
    "wer",
    "confidence",
    "preprocessing_method",
    "page_segmentation_mode",
    "ocr_ms",
    "detection_ms",
)


def _edit_distance(reference: list[Any], hypothesis: list[Any]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_text = normalize_unicode_text(reference)
    hypothesis_text = normalize_unicode_text(hypothesis)
    return _edit_distance(list(reference_text), list(hypothesis_text)) / max(
        1, len(reference_text)
    )


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = normalize_unicode_text(reference).split()
    hypothesis_words = normalize_unicode_text(hypothesis).split()
    return _edit_distance(reference_words, hypothesis_words) / max(
        1, len(reference_words)
    )


def _metrics(reference: str | None, hypothesis: str) -> tuple[float | None, float | None]:
    if reference is None:
        return None, None
    return (
        character_error_rate(reference, hypothesis),
        word_error_rate(reference, hypothesis),
    )


def _record(
    *,
    track: str,
    image_name: str,
    scope: str,
    region_index: int | None,
    reference: str | None,
    recognition: dict[str, Any],
    preprocessing_method: str,
    psm: int,
    ocr_ms: float,
    detection_ms: float | None,
) -> dict[str, Any]:
    recognized_text = normalize_unicode_text(str(recognition.get("text", ""))).strip()
    cer, wer = _metrics(reference, recognized_text)
    return {
        "track": track,
        "image": image_name,
        "scope": scope,
        "region_index": region_index,
        "reference_text": reference,
        "recognized_text": recognized_text,
        "cer": cer,
        "wer": wer,
        "confidence": recognition.get("confidence"),
        "preprocessing_method": preprocessing_method,
        "page_segmentation_mode": psm,
        "ocr_ms": round(ocr_ms, 2),
        "detection_ms": round(detection_ms, 2) if detection_ms is not None else None,
    }


def _run_recognition(
    image: np.ndarray,
    *,
    track: str,
    image_name: str,
    scope: str,
    region_index: int | None,
    reference: str | None,
    method: str,
    psm: int,
    detection_ms: float | None = None,
    padding_ratio: float = 0,
    upscale_small_crop: bool = False,
) -> dict[str, Any]:
    candidate = image
    if padding_ratio:
        height, width = image.shape[:2]
        pad_x = max(3, round(width * padding_ratio))
        pad_y = max(3, round(height * padding_ratio))
        canvas = np.full(
            (height + 2 * pad_y, width + 2 * pad_x, 3),
            255,
            dtype=np.uint8,
        )
        canvas[pad_y : pad_y + height, pad_x : pad_x + width] = image
        candidate = canvas

    actual_method = resolve_tesseract_preprocessing_method(candidate, method)
    prepared = preprocess_for_tesseract(candidate, method=actual_method)
    if upscale_small_crop and prepared.shape[0] < UPSCALE_CROP_HEIGHT_BELOW:
        prepared = cv2.resize(
            prepared,
            None,
            fx=2,
            fy=2,
            interpolation=cv2.INTER_CUBIC,
        )

    started = time.perf_counter()
    recognition = TesseractOCR(page_segmentation_mode=psm).recognize(prepared)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return _record(
        track=track,
        image_name=image_name,
        scope=scope,
        region_index=region_index,
        reference=reference,
        recognition=recognition,
        preprocessing_method=actual_method,
        psm=psm,
        ocr_ms=elapsed_ms,
        detection_ms=detection_ms,
    )


def evaluate_image(
    image: np.ndarray,
    *,
    track: str,
    image_name: str,
    reference: str | None = None,
    detector: Any | None = None,
    recognizer: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Evaluation input must be a non-empty image.")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Evaluation input must be a three-channel uint8 BGR image.")

    ocr_image = resize_to_max_side(image, max_side=MAX_OCR_IMAGE_SIDE)
    psm = 6 if max(image.shape[:2]) <= 64 else 11
    recognize = recognizer or _run_recognition
    rows = [
        recognize(
            ocr_image,
            track=track,
            image_name=image_name,
            scope="full_frame",
            region_index=None,
            reference=reference,
            method="auto",
            psm=psm,
        )
    ]

    for method in TESSERACT_PREPROCESSING_METHODS:
        if method == "auto":
            continue
        rows.append(
            recognize(
                ocr_image,
                track=track,
                image_name=image_name,
                scope=f"full_frame_{method}",
                region_index=None,
                reference=reference,
                method=method,
                psm=psm,
            )
        )

    active_detector = detector or get_text_detector()
    detection_started = time.perf_counter()
    regions = active_detector.detect(ocr_image)
    detection_ms = (time.perf_counter() - detection_started) * 1000
    region_reference = reference if len(regions) == 1 else None
    height, width = ocr_image.shape[:2]
    for index, region in enumerate(regions, start=1):
        left = max(0, int(region["left"]))
        top = max(0, int(region["top"]))
        right = min(width, int(region["right"]))
        bottom = min(height, int(region["bottom"]))
        if left >= right or top >= bottom:
            continue
        crop = ocr_image[top:bottom, left:right].copy()
        common = {
            "track": track,
            "image_name": image_name,
            "region_index": index,
            "reference": region_reference,
            "psm": 6,
            "detection_ms": detection_ms,
        }
        rows.append(
            recognize(
                crop,
                scope="detector_crop_tight",
                method="auto",
                **common,
            )
        )
        rows.append(
            recognize(
                crop,
                scope="detector_crop_padded_grayscale",
                method="grayscale",
                padding_ratio=CROP_PADDING_RATIO,
                **common,
            )
        )
        rows.append(
            recognize(
                crop,
                scope="detector_crop_padded_clahe",
                method="clahe_grayscale",
                padding_ratio=CROP_PADDING_RATIO,
                **common,
            )
        )
        rows.append(
            recognize(
                crop,
                scope="detector_crop_padded_grayscale_upscaled_if_small",
                method="grayscale",
                padding_ratio=CROP_PADDING_RATIO,
                upscale_small_crop=True,
                **common,
            )
        )
    return rows


def _load_word_samples(limit: int, seed: int) -> list[tuple[Path, str]]:
    sample_data_dir = PROJECT_ROOT / "sample_data"
    validation_dir = sample_data_dir / "val"
    image_list = validation_dir / "val_images.txt"
    label_list = validation_dir / "val_labels.txt"
    if not image_list.is_file() or not label_list.is_file():
        raise FileNotFoundError(
            f"Validation image/label lists are missing under {validation_dir}."
        )

    image_names = image_list.read_text(encoding="utf-8").splitlines()
    labels = label_list.read_text(encoding="utf-8").splitlines()
    if len(image_names) != len(labels):
        raise ValueError(
            f"Validation list lengths do not match: {len(image_names)} images and "
            f"{len(labels)} labels."
        )
    available = [
        (sample_data_dir / name, label)
        for name, label in zip(image_names, labels)
        if (sample_data_dir / name).is_file()
    ]
    random.Random(seed).shuffle(available)
    return available[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    full_rows = [row for row in rows if row["scope"] == "full_frame"]
    if not full_rows:
        print("No full-frame OCR results were produced.")
        return
    print("\nFull-frame word-image summary (references from local labels):")
    for method in ("full_frame", "full_frame_grayscale", "full_frame_clahe_grayscale",
                   "full_frame_denoised_grayscale", "full_frame_adaptive_threshold",
                   "full_frame_otsu_threshold", "full_frame_legacy_otsu"):
        subset = [row for row in rows if row["scope"] == method and row["cer"] is not None]
        if not subset:
            continue
        print(
            f"{method}: n={len(subset)}, "
            f"mean CER={sum(row['cer'] for row in subset) / len(subset):.3f}, "
            f"mean WER={sum(row['wer'] for row in subset) / len(subset):.3f}, "
            f"exact={sum(row['cer'] == 0 for row in subset)}, "
            f"mean OCR={sum(row['ocr_ms'] for row in subset) / len(subset):.1f} ms"
        )
    print("\nSingle-region detector crop candidates (only images with one detected crop):")
    crop_scopes = (
        "detector_crop_tight",
        "detector_crop_padded_grayscale",
        "detector_crop_padded_clahe",
        "detector_crop_padded_grayscale_upscaled_if_small",
    )
    for scope in crop_scopes:
        subset = [row for row in rows if row["scope"] == scope and row["cer"] is not None]
        if not subset:
            continue
        print(
            f"{scope}: n={len(subset)}, "
            f"mean CER={sum(row['cer'] for row in subset) / len(subset):.3f}, "
            f"mean WER={sum(row['wer'] for row in subset) / len(subset):.3f}, "
            f"exact={sum(row['cer'] == 0 for row in subset)}, "
            f"mean OCR={sum(row['ocr_ms'] for row in subset) / len(subset):.1f} ms"
        )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Compare Tesseract settings and PP-OCRv5 detector crops without saving changes."
    )
    parser.add_argument(
        "--track",
        choices=("word-images", "scene-images"),
        default="word-images",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image",
        action="append",
        type=Path,
        help="Scene image path; repeat for each scene sample. Required for scene-images.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional path for a UTF-8 CSV report. Input images and labels are never modified.",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive.")

    samples: list[tuple[Path, str | None]]
    if args.track == "word-images":
        samples = [(path, label) for path, label in _load_word_samples(args.limit, args.seed)]
        if not samples:
            parser.error("No referenced validation images are available on disk.")
    else:
        if not args.image:
            parser.error("Provide one or more --image paths for the scene-images track.")
        samples = [(path, None) for path in args.image]

    TesseractOCR().check_available()
    rows: list[dict[str, Any]] = []
    for image_path, reference in samples:
        if not image_path.is_file():
            raise FileNotFoundError(f"Evaluation image not found: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Evaluation image could not be decoded: {image_path}")
        image_rows = evaluate_image(
            image,
            track=args.track,
            image_name=str(image_path),
            reference=reference,
        )
        rows.extend(image_rows)
        for row in image_rows:
            print(
                f"{row['track']} | {row['image']} | {row['scope']} "
                f"| ref={row['reference_text']!r} | text={row['recognized_text']!r} "
                f"| CER={row['cer']} | WER={row['wer']} "
                f"| conf={row['confidence']} "
                f"| prep={row['preprocessing_method']} | PSM={row['page_segmentation_mode']} "
                f"| OCR={row['ocr_ms']} ms"
            )

    if args.track == "word-images":
        _print_summary(rows)
    else:
        print(
            "\nScene-image results have no reference transcriptions; CER/WER are intentionally "
            "not calculated."
        )
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"CSV report: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
