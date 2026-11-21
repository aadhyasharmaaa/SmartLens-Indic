# SmartLens-Indic: Windows demo and test instructions

These steps use the existing project environment and local files. They do not
install packages or download OCR models, language data, or voice files. Run
PowerShell from the project root (the folder containing `app.py`).

## 1. Check what is already installed

Run this read-only check:

```powershell
Test-Path .\.venv\Scripts\python.exe
Test-Path .\models\tesseract\tessdata\hin.traineddata
Test-Path .\models\ppocr\PP-OCRv5_mobile_det_infer\inference.yml
Test-Path .\models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer\devanagari_PP-OCRv5_mobile_rec_infer\inference.yml
Test-Path .\models\voices\hi_IN-rohan-medium.onnx
Test-Path .\models\voices\hi_IN-rohan-medium.onnx.json
$tesseract = if ($env:SMARTLENS_TESSERACT_CMD) { $env:SMARTLENS_TESSERACT_CMD } else { Join-Path $env:LOCALAPPDATA 'Programs\Tesseract-OCR\tesseract.exe' }
Test-Path $tesseract
```

`True` means that path exists; it does not prove that the file is valid or
compatible. The Python environment, Tesseract executable, and Hindi
`hin.traineddata` are required for OCR. If those are missing, stop here and ask
the project maintainer about setup. The detector and Devanagari recognizer are
optional and the app has Tesseract fallback behavior; without the Piper voice
pair, OCR can still run but speech synthesis is unavailable. This demo does
not fetch missing dependencies or models.

## 2. Run the automated tests

Run the new cross-component regression test:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_parallel_integration.py
```

To run the repository test suite instead:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The new integration test uses deterministic test doubles for OCR, detection,
and speech. It checks image decoding, service composition, speech gating, and
reading-history persistence; it does **not** measure real OCR quality or verify
real speech synthesis.

## 3. Start the app

From the project root, launch Streamlit with the already-existing environment:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).
For a predictable manual demo, choose **Upload image** in the sidebar, select a
local image containing Devanagari text, then select **Process image and read
aloud**. Review the displayed OCR candidates, warnings, and scores; they are
diagnostic outputs, not proof that the transcription is correct. If all
configured components and model files are available, generated history and
runtime files are stored under `runtime_data\`.

The app also has a **Sample tests** page for images already extracted in this
checkout, and a **History** page for saved readings. Sample availability
depends on the local files. Processing real images may produce empty or
incorrect text, and speech may be skipped when the independent Tesseract
confirmation checks do not pass.

## Limitations and unverified behavior

- Webcam capture depends on browser/device availability and the browser's
  permission flow. Camera permission and a physical-camera capture were not
  verified as part of these instructions; use upload for the demo if camera
  capture is unavailable.
- OCR output, recognition scores, and processing times depend on image quality,
  installed local models, and the computer. Scores are not calibrated
  probabilities, and no accuracy claim is made here.
- Piper output depends on the local Hindi voice files and phonemizer behavior.
  The test above substitutes a fake speech function and does not verify audible
  speech.
- The new integration test does not exercise browser camera capture, actual
  Tesseract/PaddleOCR inference, real Piper synthesis, or offline behavior
  across every dependency/runtime configuration.
