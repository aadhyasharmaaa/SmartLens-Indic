from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "Data"
SAMPLE_DATA_DIR = ROOT_DIR / "sample_data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
LOGS_DIR = ROOT_DIR / "logs"

DEFAULT_IMAGE_PATH = SAMPLE_DATA_DIR / "val" / "images" / "13042023_L_GH012834_image_000102_15.png"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_BOX_AREA = 500

# Exact research-paper architecture expects the model assets to be provided by the user.
PP_OCR_DETECTOR_MODEL_DIR = MODELS_DIR / "ppocr" / "det"
PP_OCR_RECOGNIZER_MODEL_DIR = MODELS_DIR / "ppocr" / "rec"
PIPER_VOICE_PATH = MODELS_DIR / "voices" / "hi_IN-rohan-medium.onnx"

TEXT_DETECTION_MODEL_NAME = "PP-OCRv5_server_det"
TEXT_RECOGNITION_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"
