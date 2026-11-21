from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st

from config.settings import (
    DATABASE_PATH,
    LOGS_DIR,
    PP_OCR_DETECTOR_MODEL_DIR,
    PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR,
    PIPER_VOICE_PATH,
    ROOT_DIR,
    TESSERACT_CMD,
    TESSDATA_DIR,
)
from database.repository import (
    get_reading_count,
    get_recent_readings,
    initialize_database,
    record_failure,
    record_reading,
)
from recognition.tesseract_ocr import TesseractOCR
from services.reading_service import process_image

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
logger = logging.getLogger("smartlens")

PREPROCESSING_OPTIONS = {
    "Automatic (recommended)": "auto",
    "Grayscale (no threshold)": "grayscale",
    "CLAHE grayscale": "clahe_grayscale",
    "Denoised grayscale": "denoised_grayscale",
    "Adaptive threshold (retry)": "adaptive_threshold",
    "Otsu threshold (retry)": "otsu_threshold",
    "Previous CLAHE + denoise + Otsu": "legacy_otsu",
}
PAGE_MODE_OPTIONS = {
    "Automatic (PSM 6 for tiny word images; PSM 11 otherwise)": None,
    "Text block or multiple lines (PSM 6)": 6,
    "Sparse text, phone screen, or sign (PSM 11)": 11,
}


def _configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if not logger.handlers:
        handler = logging.FileHandler(LOGS_DIR / "smartlens.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("No image data was supplied.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB upload limit.")

    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("The selected file is not a readable image.")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("Image dimensions are too large to process safely.")
    return image


def _uploaded_image_bytes(uploaded: Any) -> bytes:
    if uploaded.size > MAX_IMAGE_BYTES:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB upload limit.")
    return uploaded.getvalue()


def _is_low_information_camera_frame(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    counts = np.bincount(gray.ravel() // 8, minlength=32)
    probabilities = counts[counts > 0] / gray.size
    entropy = -np.sum(probabilities * np.log2(probabilities))
    dominant_fraction = counts.max() / gray.size
    return entropy < 1.0 and dominant_fraction >= 0.98


def _available_samples() -> dict[str, Path]:
    samples: dict[str, Path] = {}
    roots = (
        ("Validation word images", ROOT_DIR / "sample_data" / "val" / "images"),
        ("Extracted scene images", ROOT_DIR / "Data" / "scene_test"),
    )
    for label, root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                relative = path.relative_to(ROOT_DIR)
                samples[f"{label}: {relative}"] = path
    return samples


def _show_result(result: dict, *, show_audio: bool = True) -> None:
    status = result["status"]
    if status == "empty_text":
        st.warning("No Hindi text was recognized. Try a sharper, closer, or better-lit image.")
    elif status == "tts_failed":
        st.error(result["error_message"] or "OCR succeeded, but Piper could not generate audio.")
    else:
        st.success("Image processing completed.")
    if result.get("detection_warning"):
        st.warning(result["detection_warning"])
    if result.get("recognition_warning"):
        st.warning(result["recognition_warning"])
    if result.get("tts_skipped_reason"):
        st.warning(result["tts_skipped_reason"])
    st.caption(f"Text-region source: {result.get('detection_source', 'not available')}")
    st.caption(f"Hindi recognition: {result.get('recognition_engine', 'Tesseract')}")

    original_column, detected_column, selected_column, processed_column = st.columns(4)
    with original_column:
        st.markdown("**Original image**")
        st.image(result["original_image"], channels="BGR", width="stretch")
    with detected_column:
        st.markdown("**Automatically detected regions**")
        region_preview = result["original_image"].copy()
        for region in result.get("detected_regions", []):
            polygons = region.get("polygons") or [region.get("polygon")]
            polygons = [polygon for polygon in polygons if polygon]
            if polygons:
                cv2.polylines(
                    region_preview,
                    [np.asarray(polygon, dtype=np.int32) for polygon in polygons],
                    True,
                    (0, 200, 190),
                    2,
                )
            else:
                cv2.rectangle(
                    region_preview,
                    (region["left"], region["top"]),
                    (region["right"], region["bottom"]),
                    (0, 200, 190),
                    2,
                )
        st.image(region_preview, channels="BGR", width="stretch")
    with selected_column:
        st.markdown("**Text region crop**")
        if result.get("region_crops"):
            st.image(result["region_crops"][0]["image"], channels="BGR", width="stretch")
        else:
            st.image(result["selected_region_image"], channels="BGR", width="stretch")
    with processed_column:
        st.markdown(f"**Preprocessed image ({result['preprocessing_method']})**")
        st.image(result["preprocessed_image"], clamp=True, width="stretch")
    if result.get("region_crops"):
        with st.expander("Detected text regions and per-region OCR details"):
            for region in result["region_crops"]:
                if region["confidence"] is None:
                    confidence_text = "not available"
                elif region["source"] == "PP-OCRv5_mobile_det":
                    confidence_text = f"{region['confidence'] * 100:.1f}/100"
                else:
                    confidence_text = f"{region['confidence']:.1f}/100"
                region_confidence_text = (
                    f"{region['region_ocr_confidence']:.1f}/100"
                    if region.get("region_ocr_confidence") is not None
                    else "not available"
                )
                st.markdown(
                    f"**Region {region['index']}** · detection/OCR-box score: {confidence_text}"
                )
                st.image(
                    region["preprocessed_image"],
                    caption="Preprocessed text-region crop",
                    clamp=True,
                    width="stretch",
                )
                if region["text"]:
                    st.write(f"Full-image OCR line: {region['text']}")
                if region.get("paddle_ocr_text"):
                    paddle_score = region.get("paddle_ocr_confidence")
                    paddle_score_text = (
                        f"{paddle_score:.3f} (model score, not calibrated)"
                        if paddle_score is not None
                        else "not available"
                    )
                    st.write(
                        f"Paddle Devanagari: {region['paddle_ocr_text']} "
                        f"(score {paddle_score_text})"
                    )
                st.write(
                    f"Tesseract fallback: {region.get('region_ocr_text') or '(no text)'}"
                )
                validation_confidence = region.get("speech_validation_confidence")
                validation_score = (
                    f"{validation_confidence:.1f}/100"
                    if validation_confidence is not None
                    else "not available"
                )
                validation_status = (
                    "independent speech check passed"
                    if region.get("speech_validated")
                    else "speech check not passed"
                )
                st.caption(
                    f"Selected: {region.get('selected_recognizer', 'Tesseract')} · "
                    f"Tesseract confirmation: "
                    f"{region.get('speech_validation_source', 'not available')} "
                    f"({validation_score}) · {validation_status}"
                )
    if result.get("regions_not_retried", 0):
        st.info(
            f"{result['regions_not_retried']} additional detected text lines were kept in the "
            "full-image OCR result; per-region retry is capped at "
            f"{result['region_ocr_retry_limit']} regions to keep processing responsive."
        )

    st.markdown("**Recognized Hindi text**")
    if result["recognized_text"]:
        st.text_area(
            "OCR result",
            value=result["recognized_text"],
            height=130,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("There is no recognized text to display.")

    confidence = result["confidence_mean"]
    st.caption(
        "Mean Tesseract word confidence: "
        + (f"{confidence:.1f}/100" if confidence is not None else "not available")
        + " (a diagnostic score, not a calibrated probability)"
    )
    st.caption(f"Tesseract page segmentation mode: PSM {result['page_segmentation_mode']}")
    word_details = result.get("word_details", [])
    review_tokens = [
        token
        for token in word_details
        if token["uncertain"]
        or not any("\u0900" <= character <= "\u097f" for character in token["text"])
    ]
    if review_tokens:
        st.warning(
            "Review uncertain or non-Devanagari fragments below. They are retained in the OCR "
            "text and are not automatically deleted."
        )
        st.text(
            " · ".join(
                f"{token['text']} ("
                + (
                    f"{token['confidence']:.1f}/100"
                    if token["confidence"] is not None
                    else "confidence unavailable"
                )
                + ")"
                for token in review_tokens
            )
        )
    st.caption(
        f"Preprocessing {result['preprocess_ms']:.1f} ms · "
        f"Detection {result['text_detection_ms']:.1f} ms · "
        f"Region prep {result['region_preprocess_ms']:.1f} ms · "
        f"OCR total {result['ocr_ms']:.1f} ms · "
        + (f"TTS {result['tts_ms']:.1f} ms · " if result["tts_ms"] is not None else "")
        + f"Total {result['total_ms']:.1f} ms"
    )
    if show_audio and result["audio_path"]:
        audio_path = Path(result["audio_path"])
        if audio_path.is_file():
            st.markdown("**Hindi audio**")
            spoken_ids = st.session_state.setdefault("autoplayed_audio_ids", set())
            autoplay = result["id"] not in spoken_ids
            if autoplay:
                spoken_ids.add(result["id"])
            st.audio(
                audio_path.read_bytes(),
                format="audio/wav",
                autoplay=autoplay,
            )


def _render_ocr_controls(key: str) -> tuple[str, int | None]:
    method_label = st.selectbox(
        "Preprocessing method",
        list(PREPROCESSING_OPTIONS),
        key=f"{key}-preprocessing-method",
        help=(
            "Automatic mode uses the previous thresholding pipeline only for tiny word images; "
            "larger images use grayscale CLAHE without global thresholding."
        ),
    )
    page_mode_label = st.selectbox(
        "Text layout",
        list(PAGE_MODE_OPTIONS),
        key=f"{key}-page-mode",
        help="PSM 6 suits a text block; PSM 11 treats content as sparse, separately placed text.",
    )
    return PREPROCESSING_OPTIONS[method_label], PAGE_MODE_OPTIONS[page_mode_label]


def _run_reading(
    image: np.ndarray,
    input_kind: str,
    generate_audio: bool,
    *,
    crop_box: tuple[int, int, int, int] | None = None,
    preprocessing_method: str = "auto",
    page_segmentation_mode: int | None = None,
) -> None:
    if input_kind == "webcam" and _is_low_information_camera_frame(image):
        st.warning(
            "The captured frame is almost blank, so OCR and speech were skipped. "
            "Allow camera access and capture a real scene, or upload an image."
        )
        return

    try:
        result = process_image(
            image,
            input_kind=input_kind,
            generate_audio=generate_audio,
            crop_box=crop_box,
            preprocessing_method=preprocessing_method,
            page_segmentation_mode=page_segmentation_mode,
        )
        try:
            record_reading(result)
        except Exception as exc:
            logger.exception("Could not save reading history")
            st.error(f"OCR completed, but the result could not be saved to history: {exc}")
        st.session_state["last_reading"] = result
    except Exception as exc:
        logger.exception("Image processing failed")
        try:
            record_failure(str(exc), input_kind=input_kind)
        except Exception:
            logger.exception("Could not save failed reading to history")
        st.error(f"Could not process this image: {exc}")


def _render_read_page() -> None:
    st.header("Read Hindi text")
    st.write("Capture a scene to detect and read Hindi aloud. Processing stays on this computer.")
    with st.expander("Advanced OCR options", expanded=False):
        preprocessing_method, page_segmentation_mode = _render_ocr_controls("reading")
    if "reading-preprocessing-method" not in st.session_state:
        preprocessing_method = "auto"
    if "reading-page-mode" not in st.session_state:
        page_segmentation_mode = None

    camera_tab, upload_tab = st.tabs(["Capture and read", "Upload image"])

    with camera_tab:
        st.caption(
            "Allow camera access, point it at Hindi text, then capture. "
            "OCR and speech run automatically."
        )
        try:
            captured = st.camera_input("Capture a Hindi text scene", key="reading_camera")
            if captured is None:
                st.info(
                    "No camera image has been captured yet. If your browser asks, allow camera "
                    "access; if it is blocked, check the browser's site permissions and try "
                    "again. The Upload image tab is also available."
                )
            else:
                capture_bytes = _uploaded_image_bytes(captured)
                capture_id = hashlib.sha256(capture_bytes).hexdigest()
                preview = _decode_image(capture_bytes)
                st.image(preview, channels="BGR", caption="Captured scene", width="stretch")
                if st.session_state.get("processed_camera_capture") != capture_id:
                    _run_reading(
                        preview,
                        "webcam",
                        True,
                        preprocessing_method=preprocessing_method,
                        page_segmentation_mode=page_segmentation_mode,
                    )
                    st.session_state["processed_camera_capture"] = capture_id
                if st.button("Retry this capture", type="secondary"):
                    _run_reading(
                        preview,
                        "webcam",
                        True,
                        preprocessing_method=preprocessing_method,
                        page_segmentation_mode=page_segmentation_mode,
                    )
                    st.session_state["processed_camera_capture"] = capture_id
        except ValueError as exc:
            logger.exception("Could not use the captured webcam image")
            st.error(f"Could not use this camera image: {exc}")
        except Exception as exc:
            logger.exception("Webcam capture is unavailable")
            st.warning(f"Webcam is unavailable. You can upload an image instead. Details: {exc}")

    with upload_tab:
        uploaded = st.file_uploader(
            "Choose a Hindi text image",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            key="reading_upload",
        )
        if uploaded is not None:
            try:
                preview = _decode_image(_uploaded_image_bytes(uploaded))
                st.image(preview, channels="BGR", caption=uploaded.name, width="stretch")
                if st.button("Process image and read aloud", type="primary"):
                    _run_reading(
                        preview,
                        "upload",
                        True,
                        preprocessing_method=preprocessing_method,
                        page_segmentation_mode=page_segmentation_mode,
                    )
            except ValueError as exc:
                st.error(str(exc))

    last_result = st.session_state.get("last_reading")
    if last_result is not None:
        st.divider()
        _show_result(last_result)


def _render_sample_tests() -> None:
    st.header("Test extracted sample images")
    st.write(
        "Choose images already extracted in this workspace. The original dataset ZIP archives "
        "are not opened or modified by this test."
    )
    preprocessing_method, page_segmentation_mode = _render_ocr_controls("samples")
    samples = _available_samples()
    if not samples:
        st.info("No extracted sample images are available.")
        return

    options = list(samples)
    selected = st.multiselect(
        "Images to process",
        options,
        default=options[:3],
        help="These are real extracted files; OCR output is shown without claiming accuracy.",
    )
    if st.button("Run selected image tests", type="primary", disabled=not selected):
        st.session_state["sample_results"] = []
        progress = st.progress(0, text="Preparing sample test…")
        for index, label in enumerate(selected):
            image_path = samples[label]
            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode image: {image_path}")
                result = process_image(
                    image,
                    input_kind="sample_test",
                    generate_audio=False,
                    preprocessing_method=preprocessing_method,
                    page_segmentation_mode=page_segmentation_mode,
                )
                result["sample_label"] = label
                record_reading(result)
                st.session_state["sample_results"].append(result)
            except Exception as exc:
                logger.exception("Sample image test failed for %s", image_path)
                try:
                    record_failure(str(exc), input_kind="sample_test")
                except Exception:
                    logger.exception("Could not save sample failure to history")
                st.session_state["sample_results"].append(
                    {"sample_label": label, "error_message": str(exc)}
                )
            progress.progress((index + 1) / len(selected), text=f"Processed {index + 1} of {len(selected)}")
        progress.empty()

    for result in st.session_state.get("sample_results", []):
        with st.expander(result["sample_label"], expanded=True):
            if "error_message" in result and "recognized_text" not in result:
                st.error(result["error_message"])
            else:
                _show_result(result, show_audio=False)


def _render_history() -> None:
    st.header("Reading history")
    try:
        readings = get_recent_readings(100)
    except Exception as exc:
        logger.exception("History retrieval failed")
        st.error(f"Could not load reading history: {exc}")
        return

    if not readings:
        st.info("No readings have been saved yet.")
        return

    for reading in readings:
        timestamp = reading["created_at"].replace("T", " ")
        heading = f"{timestamp} · {reading['status']} · {reading['input_kind']}"
        with st.expander(heading):
            if reading["recognized_text"]:
                st.text_area(
                    "Recognized text",
                    value=reading["recognized_text"],
                    height=100,
                    disabled=True,
                    key=f"history-text-{reading['id']}",
                )
            else:
                st.info("No recognized text was saved.")
            st.caption(
                f"Recognition engine: {reading.get('recognition_engine') or 'Tesseract'} · "
            )
            st.caption(
                f"Detection {reading.get('text_detection_ms') or 0:.1f} ms · "
                f"Total {reading['total_ms'] or 0:.1f} ms · "
                f"OCR {reading['ocr_ms'] or 0:.1f} ms"
            )
            if reading["confidence_mean"] is not None:
                st.caption(
                    f"Mean Tesseract word confidence: {reading['confidence_mean']:.1f}/100 "
                    "(diagnostic only)"
                )
            for label, key in (
                ("Original image", "original_image_path"),
                ("Preprocessed image", "preprocessed_image_path"),
            ):
                path = Path(reading[key]) if reading[key] else None
                if path and path.is_file():
                    st.image(str(path), caption=label, width="stretch")
            audio_path = Path(reading["audio_path"]) if reading["audio_path"] else None
            if audio_path and audio_path.is_file():
                st.audio(audio_path.read_bytes(), format="audio/wav")
            if reading["error_message"]:
                st.error(reading["error_message"])


def _render_dashboard(ocr_status: str | None) -> None:
    st.title("SmartLens-Indic")
    st.subheader("Offline Hindi image-to-speech assistant")
    st.write(
        "Read Devanagari text from an uploaded image or laptop webcam. "
        "This Windows application uses local OpenCV, PP-OCRv5, Tesseract Hindi OCR, "
        "and Piper speech."
    )
    left, middle, right, camera, detector, recognizer = st.columns(6)
    left.metric("Saved readings", get_reading_count())
    middle.metric("Tesseract Hindi OCR", "Ready" if ocr_status is None else "Needs setup")
    voice_ready = PIPER_VOICE_PATH.is_file() and Path(f"{PIPER_VOICE_PATH}.json").is_file()
    right.metric("Piper Hindi voice", "Ready" if voice_ready else "Missing")
    camera.metric("Laptop camera", "Permission on capture")
    detector_ready = (PP_OCR_DETECTOR_MODEL_DIR / "inference.yml").is_file()
    detector.metric("PP-OCRv5 text detector", "Ready" if detector_ready else "Fallback mode")
    recognizer_ready = (
        PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR / "inference.yml"
    ).is_file()
    recognizer.metric(
        "Paddle Devanagari model",
        "Files found" if recognizer_ready else "Tesseract fallback",
    )
    if ocr_status:
        st.error(ocr_status)
    st.info(
        "Capture Hindi text; SmartLens detects text regions, uses the local Devanagari "
        "recognizer with Tesseract fallback, and speaks only results independently confirmed "
        "by Tesseract."
    )
    st.button(
        "Start camera and read",
        type="primary",
        on_click=lambda: st.session_state.__setitem__("Navigation", "Camera & read"),
    )
    st.caption("Windows laptop edition; performance depends on this computer.")


def main() -> None:
    st.set_page_config(page_title="SmartLens-Indic", page_icon="📖", layout="wide")
    _configure_logging()
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(135deg, #081321 0%, #101e2b 55%, #0c202a 100%); color: #eef7f7;}
        [data-testid="stSidebar"] {background: #0a1724;}
        .block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1440px;}
        [data-testid="stMetric"] {background: #122634; border: 1px solid #1b5360; padding: 1rem; border-radius: 0.9rem;}
        [data-testid="stButton"] button[kind="primary"] {background: #11a9a0; border: 0; color: #061822; min-height: 3.2rem; font-weight: 700;}
        [data-testid="stButton"] button:focus-visible, [data-testid="stFileUploader"] button:focus-visible {outline: 3px solid #49e1d3; outline-offset: 3px;}
        [data-testid="stAlert"] {border-radius: 0.75rem;}
        h1, h2, h3 {color: #f0ffff;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        initialize_database()
    except Exception as exc:
        logger.exception("Database initialization failed")
        st.error(f"Could not initialize the local reading-history database: {exc}")
        st.stop()

    try:
        ocr_status = None
        TesseractOCR().check_available()
    except Exception as exc:
        ocr_status = str(exc)
        logger.error("Tesseract startup check failed: %s", exc)

    menu = st.sidebar.radio(
        "Navigation",
        ["Camera & read", "Dashboard", "Upload image", "Sample tests", "History"],
        index=0,
        key="Navigation",
    )
    st.sidebar.caption(f"Local database: {DATABASE_PATH}")
    st.sidebar.caption("Processing is performed locally.")

    if menu == "Dashboard":
        _render_dashboard(ocr_status)
    elif menu in {"Camera & read", "Upload image"}:
        if ocr_status:
            st.error(f"Tesseract is not ready: {ocr_status}")
        else:
            _render_read_page()
    elif menu == "Sample tests":
        if ocr_status:
            st.error(f"Tesseract is not ready: {ocr_status}")
        else:
            _render_sample_tests()
    else:
        _render_history()


if __name__ == "__main__":
    main()
