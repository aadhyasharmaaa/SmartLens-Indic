<<<<<<< HEAD
# SmartLens-Indic
University Major Project
=======
# SmartLens-Indic Windows OCR Prototype

This workspace contains a starter prototype for the SmartLens-Indic project.

## Purpose
This prototype is built around the paper's architecture and keeps the modules separated so the exact OCR and speech pipeline can be swapped in as soon as the required model assets are available.

## Current status
- Image input and preprocessing are implemented and tested with sample images from the workspace dataset.
- OCR detection and recognizer modules are in place, but they intentionally fail clearly when the exact PP-OCRv5 model files are not available.
- Hindi speech output via Piper is implemented as a modular function and remains pending until a hi_IN voice model is available.

## Required research-paper architecture
- OpenCV preprocessing
- PP-OCRv5 detector
- devanagari_PP-OCRv5_mobile_rec recognizer
- confidence filtering
- Unicode NFC normalization
- reading-order arrangement
- Piper hi_IN speech output

## Quick demo
Run from the workspace root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py --image sample_data\val\images\13042023_L_GH012834_image_000102_15.png
```

## Important note
This workspace intentionally does not auto-download large OCR model files or voice models. The exact paper-specific model assets must be supplied or approved before real OCR inference or speech synthesis can run.
>>>>>>> bb08a96 (Initial project upload for SmartLens-Indic)
