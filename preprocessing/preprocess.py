from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from config.settings import OUTPUT_DIR


def load_image(image_path: str | Path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    return image


def resize_to_max_side(image: np.ndarray, max_side: int = 960):
    h, w = image.shape[:2]
    scale = min(max_side / float(max(h, w)), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def apply_clahe(image: np.ndarray):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def denoise_image(image: np.ndarray):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


TESSERACT_PREPROCESSING_METHODS = (
    "auto",
    "grayscale",
    "clahe_grayscale",
    "denoised_grayscale",
    "adaptive_threshold",
    "otsu_threshold",
    "legacy_otsu",
)


def resolve_tesseract_preprocessing_method(image: np.ndarray, method: str = "auto") -> str:
    if method not in TESSERACT_PREPROCESSING_METHODS:
        raise ValueError(f"Unsupported Tesseract preprocessing method: {method}")
    if method == "auto":
        return "denoised_grayscale" if max(image.shape[:2]) <= 64 else "clahe_grayscale"
    return method


def preprocess_for_tesseract(image: np.ndarray, method: str = "auto") -> np.ndarray:
    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected a grayscale or three-channel BGR image for Tesseract preprocessing.")

    resolved_method = resolve_tesseract_preprocessing_method(image, method)
    if resolved_method == "grayscale":
        return gray
    if resolved_method == "clahe_grayscale":
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if resolved_method == "denoised_grayscale":
        return cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    if resolved_method == "otsu_threshold":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if resolved_method == "adaptive_threshold":
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        smallest_dimension = min(gray.shape)
        block_size = min(31, smallest_dimension if smallest_dimension % 2 else smallest_dimension - 1)
        if block_size < 3:
            return enhanced
        return cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            9,
        )
    if resolved_method == "legacy_otsu":
        color = image if image.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        resized = resize_to_max_side(color)
        enhanced = apply_clahe(resized)
        color_denoised = denoise_image(enhanced)
        legacy_gray = cv2.cvtColor(color_denoised, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(legacy_gray, None, 10, 7, 21)
        return cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    raise ValueError(f"Unsupported Tesseract preprocessing method: {resolved_method}")


def preprocess_image(image_path: str | Path, max_side: int = 960, mode: str = "word"):
    image = load_image(image_path)
    resized = resize_to_max_side(image, max_side=max_side)
    enhanced = apply_clahe(resized)
    denoised = denoise_image(enhanced)
    return denoised


def save_preprocessed_demo(original_path: str | Path, output_name: str = "preprocessed_demo.png"):
    OUTPUT_DIR.mkdir(exist_ok=True)
    original = load_image(original_path)
    prepared = preprocess_image(original_path)
    cv2.imwrite(str(OUTPUT_DIR / output_name), prepared)
    cv2.imwrite(str(OUTPUT_DIR / "original_demo.png"), original)
    return OUTPUT_DIR / output_name
