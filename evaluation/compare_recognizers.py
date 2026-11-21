from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import psutil

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import ROOT_DIR as PROJECT_ROOT
from evaluation.evaluate_ocr import character_error_rate, word_error_rate
from postprocessing.normalize_unicode import normalize_unicode_text
from preprocessing.preprocess import preprocess_for_tesseract, resize_to_max_side
from recognition.paddle_devanagari_recognizer import PaddleDevanagariRecognizer
from recognition.paddle_text_detector import get_text_detector
from recognition.tesseract_ocr import TesseractOCR

PROCESS = psutil.Process()
MAX_IMAGE_SIDE = 2000
REPEATS = 3
CSV_FIELDS = (
    "track",
    "image",
    "scope",
    "region_index",
    "reference_text",
    "recognized_text",
    "cer",
    "wer",
    "confidence",
    "confidence_type",
    "inference_ms_median",
    "process_cpu_ms_median",
    "process_rss_before_mb",
    "process_rss_after_mb",
    "device",
    "detector_regions",
    "error",
)


def _read_pair_lists() -> list[tuple[Path, str]]:
    root = PROJECT_ROOT / "sample_data"
    validation = root / "val"
    image_list = validation / "val_images.txt"
    label_list = validation / "val_labels.txt"
    if not image_list.is_file() or not label_list.is_file():
        raise FileNotFoundError(f"Validation annotation files are missing in {validation}.")
    images = image_list.read_text(encoding="utf-8").splitlines()
    labels = label_list.read_text(encoding="utf-8").splitlines()
    if len(images) != len(labels):
        raise ValueError(f"Validation image/label counts differ: {len(images)} != {len(labels)}.")
    return [
        (root / image_name, label)
        for image_name, label in zip(images, labels)
        if (root / image_name).is_file()
    ]


def _measure(
    *,
    track: str,
    image_name: str,
    scope: str,
    region_index: int | None,
    reference: str | None,
    image: np.ndarray,
    recognizer_name: str,
    tesseract: TesseractOCR,
    paddle: PaddleDevanagariRecognizer,
    detector_regions: int | None = None,
) -> dict[str, Any]:
    is_tesseract = recognizer_name == "Tesseract"
    input_image = (
        preprocess_for_tesseract(image, method="auto") if is_tesseract else image
    )
    run = tesseract.recognize if is_tesseract else paddle.recognize
    before_mb = PROCESS.memory_info().rss / (1024 * 1024)
    elapsed_values: list[float] = []
    cpu_values: list[float] = []
    output: dict[str, Any] | None = None
    error: str | None = None
    for _ in range(REPEATS):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            output = run(input_image)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        elapsed_values.append((time.perf_counter() - wall_start) * 1000)
        cpu_values.append((time.process_time() - cpu_start) * 1000)
    after_mb = PROCESS.memory_info().rss / (1024 * 1024)

    if output is None:
        text = ""
        confidence = None
        confidence_type = "unavailable"
    else:
        text = normalize_unicode_text(str(output.get("text", ""))).strip()
        confidence = output.get("confidence")
        confidence_type = (
            "Tesseract word-confidence mean (diagnostic)"
            if is_tesseract
            else output.get("confidence_type", "PaddleOCR model score; not calibrated")
        )

    if reference is not None and error is None:
        cer = character_error_rate(reference, text)
        wer = word_error_rate(reference, text)
    else:
        cer = None
        wer = None
    return {
        "track": track,
        "image": image_name,
        "scope": f"{scope}:{recognizer_name}",
        "region_index": region_index,
        "reference_text": reference,
        "recognized_text": text,
        "cer": cer,
        "wer": wer,
        "confidence": confidence,
        "confidence_type": confidence_type,
        "inference_ms_median": round(median(elapsed_values), 2) if elapsed_values else None,
        "process_cpu_ms_median": (
            round(median(cpu_values), 2)
            if cpu_values and not is_tesseract
            else None
        ),
        "process_rss_before_mb": round(before_mb, 2),
        "process_rss_after_mb": round(after_mb, 2),
        "device": paddle.device if not is_tesseract else "Tesseract local executable",
        "detector_regions": detector_regions,
        "error": error,
    }


def _run_word_track(
    samples: list[tuple[Path, str]],
    tesseract: TesseractOCR,
    paddle: PaddleDevanagariRecognizer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, reference in samples:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode validation image: {path}")
        for name in ("Tesseract", "PaddleOCR-Devanagari"):
            rows.append(
                _measure(
                    track="word-images",
                    image_name=str(path),
                    scope="identical_word_image",
                    region_index=None,
                    reference=reference,
                    image=image,
                    recognizer_name=name,
                    tesseract=tesseract,
                    paddle=paddle,
                )
            )
    return rows


def _run_scene_track(
    image_paths: list[Path],
    tesseract: TesseractOCR,
    paddle: PaddleDevanagariRecognizer,
) -> list[dict[str, Any]]:
    detector = get_text_detector()
    rows: list[dict[str, Any]] = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode scene image: {path}")
        ocr_image = resize_to_max_side(image, max_side=MAX_IMAGE_SIDE)
        regions = detector.detect(ocr_image)
        image_name = str(path)
        if not regions:
            rows.append(
                _measure(
                    track="scene-images",
                    image_name=image_name,
                    scope="full_image_no_detector_regions",
                    region_index=None,
                    reference=None,
                    image=ocr_image,
                    recognizer_name="Tesseract",
                    tesseract=tesseract,
                    paddle=paddle,
                    detector_regions=0,
                )
            )
            rows.append(
                _measure(
                    track="scene-images",
                    image_name=image_name,
                    scope="full_image_no_detector_regions",
                    region_index=None,
                    reference=None,
                    image=ocr_image,
                    recognizer_name="PaddleOCR-Devanagari",
                    tesseract=tesseract,
                    paddle=paddle,
                    detector_regions=0,
                )
            )
            continue

        height, width = ocr_image.shape[:2]
        region_text: dict[str, list[str]] = {
            "Tesseract": [],
            "PaddleOCR-Devanagari": [],
        }
        for index, region in enumerate(regions, start=1):
            left = max(0, int(region["left"]))
            top = max(0, int(region["top"]))
            right = min(width, int(region["right"]))
            bottom = min(height, int(region["bottom"]))
            if left >= right or top >= bottom:
                continue
            crop = ocr_image[top:bottom, left:right].copy()
            for recognizer_name in ("Tesseract", "PaddleOCR-Devanagari"):
                row = _measure(
                    track="scene-images",
                    image_name=image_name,
                    scope="same_detector_crop",
                    region_index=index,
                    reference=None,
                    image=crop,
                    recognizer_name=recognizer_name,
                    tesseract=tesseract,
                    paddle=paddle,
                    detector_regions=len(regions),
                )
                rows.append(row)
                if row["recognized_text"]:
                    region_text[recognizer_name].append(row["recognized_text"])
        for name in ("Tesseract", "PaddleOCR-Devanagari"):
            rows.append(
                {
                    "track": "scene-images",
                    "image": image_name,
                    "scope": "ordered_detector_crop_join",
                    "region_index": None,
                    "reference_text": None,
                    "recognized_text": "\n".join(region_text[name]),
                    "cer": None,
                    "wer": None,
                    "confidence": None,
                    "confidence_type": "Region-level scores are in individual crop rows",
                    "inference_ms_median": None,
                    "process_cpu_ms_median": None,
                    "process_rss_before_mb": None,
                    "process_rss_after_mb": None,
                    "device": "CPU",
                    "detector_regions": len(regions),
                    "error": None,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]], track: str) -> None:
    print(f"\n{track} summary (unavailable references excluded):")
    for recognizer_name in ("Tesseract", "PaddleOCR-Devanagari"):
        subset = [
            row for row in rows
            if row["scope"] == "identical_word_image:" + recognizer_name
            and row["cer"] is not None
            and row["error"] is None
        ]
        if subset:
            count = len(subset)
            print(
                f"{recognizer_name}: n={count}, "
                f"mean CER={sum(row['cer'] for row in subset) / count:.3f}, "
                f"mean WER={sum(row['wer'] for row in subset) / count:.3f}, "
                f"exact={sum(row['cer'] == 0 for row in subset)}, "
                f"median latency={median(row['inference_ms_median'] for row in subset):.2f} ms"
            )
    if track == "scene-images":
        print("Scene images have no verified transcriptions; CER/WER are not reported.")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(
        description="Compare Tesseract and Devanagari PP-OCRv5 on identical images/crops."
    )
    parser.add_argument("--word-limit", type=int, default=10)
    parser.add_argument(
        "--scene-image",
        action="append",
        type=Path,
        default=[],
        help="Scene image path; repeat. Scene outputs have no CER/WER without references.",
    )
    parser.add_argument(
        "--word-csv",
        type=Path,
        default=PROJECT_ROOT / "runtime_data" / "recognizer-word-comparison.csv",
    )
    parser.add_argument(
        "--scene-csv",
        type=Path,
        default=PROJECT_ROOT / "runtime_data" / "recognizer-scene-comparison.csv",
    )
    args = parser.parse_args()
    if args.word_limit < 1:
        parser.error("--word-limit must be positive.")
    tesseract = TesseractOCR(page_segmentation_mode=6)
    tesseract.check_available()
    rss_before_load = PROCESS.memory_info().rss / (1024 * 1024)
    paddle = PaddleDevanagariRecognizer(device="cpu")
    rss_after_load = PROCESS.memory_info().rss / (1024 * 1024)
    print(
        f"Recognizer runtime: PaddleOCR {paddle.device}; process RSS after model load "
        f"{rss_after_load:.1f} MB (pre-inference baseline {rss_before_load:.1f} MB)."
    )

    available = _read_pair_lists()
    random.Random(42).shuffle(available)
    word_samples = available[: args.word_limit]
    word_rows = _run_word_track(word_samples, tesseract, paddle)
    _write_csv(args.word_csv, word_rows)
    _print_summary(word_rows, "word-images")
    print(f"Word comparison CSV: {args.word_csv}")

    if args.scene_image:
        scene_rows = _run_scene_track(args.scene_image, tesseract, paddle)
        _write_csv(args.scene_csv, scene_rows)
        for row in scene_rows:
            if row["scope"] != "ordered_detector_crop_join":
                print(
                    f"{row['track']} | {row['image']} | {row['scope']} "
                    f"| crop={row['region_index']} | text={row['recognized_text']!r} "
                    f"| confidence={row['confidence']} ({row['confidence_type']}) "
                    f"| latency={row['inference_ms_median']} ms "
                    f"| RSS={row['process_rss_after_mb']} MB | error={row['error']}"
                )
        _print_summary(scene_rows, "scene-images")
        print(f"Scene comparison CSV: {args.scene_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
