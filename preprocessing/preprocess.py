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
