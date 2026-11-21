import os
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
LOGS_DIR = ROOT_DIR / "logs"
RUNTIME_DATA_DIR = ROOT_DIR / "runtime_data"
UPLOADS_DIR = RUNTIME_DATA_DIR / "uploads"
READINGS_DIR = RUNTIME_DATA_DIR / "readings"
DATABASE_PATH = RUNTIME_DATA_DIR / "smartlens.sqlite3"

DEFAULT_IMAGE_PATH = SAMPLE_DATA_DIR / "val" / "images" / "13042023_L_GH012834_image_000102_15.png"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_BOX_AREA = 500

PP_OCR_DETECTOR_MODEL_DIR = MODELS_DIR / "ppocr" / "PP-OCRv5_mobile_det_infer"
PP_OCR_DETECTOR_MODEL_NAME = "PP-OCRv5_mobile_det"
PP_OCR_DEVANAGARI_RECOGNIZER_MODEL_DIR = (
    MODELS_DIR
    / "ppocr"
    / "devanagari_PP-OCRv5_mobile_rec_infer"
    / "devanagari_PP-OCRv5_mobile_rec_infer"
)
PP_OCR_RECOGNIZER_MODEL_DIR = MODELS_DIR / "ppocr" / "rec"
PIPER_VOICE_PATH = MODELS_DIR / "voices" / "hi_IN-rohan-medium.onnx"
PIPER_VOICE_DATA_DIR = PIPER_VOICE_PATH.parent
TESSERACT_CMD = Path(
    os.environ.get(
        "SMARTLENS_TESSERACT_CMD",
        shutil.which("tesseract") or str(
            Path.home() / "AppData" / "Local" / "Programs"
            / "Tesseract-OCR" / "tesseract.exe"
        )
    )
)
TESSDATA_DIR = MODELS_DIR / "tesseract" / "tessdata"
TESSERACT_LANGUAGE = "hin"
TESSERACT_PAGE_SEGMENTATION_MODE = 6

TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"
