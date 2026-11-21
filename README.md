<<<<<<< HEAD
<<<<<<< HEAD
# SmartLens-Indic
University Major Project
=======
# SmartLens-Indic Windows OCR Prototype
=======
# SmartLens-Indic
>>>>>>> c81e519 (Initial SmartLens-Indic project upload)

SmartLens-Indic is a Windows laptop application for reading Hindi/Devanagari text from an uploaded image or laptop webcam and speaking confirmed text offline. It does not include Raspberry Pi deployment or dedicated hardware integration.

## What the application does

1. Starts on the laptop-camera capture screen; each new capture is processed once automatically. Image upload remains available for testing.
2. Preserves the original image. OpenCV prepares grayscale/CLAHE input; thresholding is retained as a selectable retry method.
3. Uses the local PP-OCRv5 mobile detector to find and order text lines. Each detected crop is padded and small crops are enlarged without changing the saved original.
4. Tries the local `devanagari_PP-OCRv5_mobile_rec` recognizer on detected lines. Tesseract Hindi OCR is retained as an independent candidate and fallback. If PaddleOCR is unavailable or produces no Devanagari text, the ordered full-image Tesseract result remains the fallback.
5. Does not automatically speak a Paddle-selected line unless Tesseract independently returns the same letters and digits (whitespace and punctuation differences are ignored for this comparison) with a positive Tesseract diagnostic score. This is a conservative agreement check, not an accuracy guarantee or calibrated confidence threshold. Unconfirmed text remains visible for the user to review. Tesseract-only fallback speech retains the existing non-empty Devanagari and positive-confidence checks.
6. Keeps browser audio playback controls, avoids repeating camera processing after a Streamlit rerun, and stores reading history, images, and generated audio under `runtime_data/`.

The existing PaddleOCR CLI and detector modules remain intact. All image processing, recognition, and speech synthesis run locally after software and model setup. Webcam access is provided by the browser and depends on its camera permission. The optional recognizer's small labeled-word comparison is preliminary; no general accuracy claim is made.

The webcam path rejects nearly blank or otherwise information-poor captured frames before OCR and speech. This is an input-usability safeguard, not an OCR confidence threshold. A real webcam permission grant and physical frame capture still need to be verified in the target browser.

The detector proposes regions, not verified text. Complex images can still contain false detections, and either recognizer can return incorrect text. Scores are displayed as diagnostics; low-confidence words are not silently deleted or corrected.

## Requirements

- Windows 10/11, 64-bit
- Python 3.10 or later (the current development environment uses Python 3.13)
- Tesseract OCR for Windows
- Hindi `hin.traineddata`
- PaddlePaddle and PaddleOCR 3 for the local PP-OCRv5 detector
- PP-OCRv5 mobile detector inference model (about 4.7 MB)
- Devanagari PP-OCRv5 Mobile recognition model (about 8.1 MB; Tesseract remains available as fallback)
- The Piper `hi_IN-rohan-medium` model and matching JSON configuration

The required Python packages are listed in `requirements.txt`. The already-required PaddleOCR/PaddlePaddle installation runs both the detector and optional Devanagari recognizer.

## One-time setup

Run these commands from the project root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

<<<<<<< HEAD
## Important note
This workspace intentionally does not auto-download large OCR model files or voice models. The exact paper-specific model assets must be supplied or approved before real OCR inference or speech synthesis can run.
>>>>>>> bb08a96 (Initial project upload for SmartLens-Indic)
=======
If PowerShell prevents virtual-environment activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Tesseract and Hindi language data

Install the 64-bit Windows Tesseract OCR engine. The default executable configured by the application is:

```text
%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe
```

The Hindi data file must be at:

```text
models\tesseract\tessdata\hin.traineddata
```

The project copy is from the official `tesseract-ocr/tessdata_best` repository:
<https://github.com/tesseract-ocr/tessdata_best/blob/main/hin.traineddata>

If Tesseract is installed elsewhere, set `SMARTLENS_TESSERACT_CMD` to the full executable path before launching the app. The Hindi language-data directory is configured in `config/settings.py`.

### PP-OCRv5 mobile text detector

The application expects the extracted model files under:

```text
models\ppocr\PP-OCRv5_mobile_det_infer\
```

The approved official PaddleOCR inference archive is 4,935,680 bytes (about 4.7 MiB). It is downloaded from PaddlePaddle's model host, not from an unofficial mirror. For a fresh checkout, download and extract it once:

```powershell
New-Item -ItemType Directory -Force models\ppocr | Out-Null
Invoke-WebRequest `
  -Uri "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar" `
  -OutFile "models\ppocr\PP-OCRv5_mobile_det_infer.tar"
tar -xf "models\ppocr\PP-OCRv5_mobile_det_infer.tar" -C "models\ppocr"
```

The model archive and extracted weights are ignored by Git and are not downloaded automatically when the app starts.

### Devanagari PP-OCRv5 Mobile recognition model

The optional recognizer is stored separately from the detector under:

```text
models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer\devanagari_PP-OCRv5_mobile_rec_infer\
```

For a fresh checkout, download the official 8,069,120-byte archive once and extract it:

```powershell
New-Item -ItemType Directory -Force models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer | Out-Null
Invoke-WebRequest `
  -Uri "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv5_mobile_rec_infer.tar" `
  -OutFile "models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer\devanagari_PP-OCRv5_mobile_rec_infer.tar"
tar -xf "models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer\devanagari_PP-OCRv5_mobile_rec_infer.tar" -C "models\ppocr\devanagari_PP-OCRv5_mobile_rec_infer"
```

If this model is absent or fails to load, the app continues with Tesseract Hindi OCR.

### Piper Hindi voice

The application expects both of these existing files:

```text
models\voices\hi_IN-rohan-medium.onnx
models\voices\hi_IN-rohan-medium.onnx.json
```

No voice or dataset download is performed when the app starts. The dashboard reports whether the OCR engine and Piper voice files are available.

The official `hi_IN-rohan-medium` model/config pair was verified, and the requested Hindi test sentence generated a non-empty mono 22,050 Hz WAV. Audio is not generated if the installed phonemizer returns no speech phonemes for Hindi text.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Allow camera permission in the browser to use webcam capture. Use:

- **Camera & read** to capture an image; each new capture is automatically processed and sent to Piper when text is found.
- **Upload image** to manually submit an image for processing and speech.
- **Sample tests** to process multiple images already extracted under `sample_data/val/images/` and `Data/scene_test/`. The original ZIP archives are not opened or modified.
- **History** to revisit saved outputs, processing times, and generated audio.

Text regions are detected and cropped automatically; no manual crop is required. The results page shows the original image, detector regions/crops (or explicitly labeled Tesseract-box fallback), preprocessed image, recognized text, both recognizer candidates, their diagnostic scores, the selected engine, speech-validation status, and preprocessing/detection/OCR/TTS timings. Runtime uploads, database records, and audio are kept under `runtime_data/uploads/`, `runtime_data/smartlens.sqlite3`, and `runtime_data/readings/`; the `Data/` directory remains for dataset resources.

## Command-line OCR

The separate Tesseract CLI does not change the existing PaddleOCR `main.py` workflow:

```powershell
.\.venv\Scripts\python.exe tesseract_main.py --image "sample_data\val\images\13042023_L_GH012834_image_000102_15.png"
.\.venv\Scripts\python.exe tesseract_main.py --image "sample_data\val\images\13042023_L_GH012834_image_000102_15.png" --tts
```

The optional `--tts` flag generates speech only when OCR returns non-empty text. Audio filenames are unique so an existing WAV is not overwritten.

## Tests and checks

Run the focused automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Check Python package dependency consistency:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

The Streamlit app displays actual OCR outputs and measured local preprocessing/OCR/TTS/total times. Tesseract confidence is shown as a diagnostic score, not a calibrated probability. Do not treat OCR output as ground truth or report accuracy without verified reference transcriptions. Timings are local to the Windows laptop and should not be generalized to other computers.

### OCR evaluation

Run the reference-labeled word-image evaluation using image/label pairs already present in `sample_data/val`:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_ocr.py --track word-images --limit 10 --csv runtime_data\word-image-evaluation.csv
```

The report compares full-frame Tesseract preprocessing variants and logs each PP-OCRv5 region crop independently. It includes reference/recognized text, CER/WER when a reference applies, Tesseract confidence, preprocessing, PSM, OCR time, and detector time. A reference is associated with an individual crop only when the detector returns exactly one region for that word image. Crop outputs are not silently concatenated or substituted for the full-frame result.

Evaluate real-world images separately, without accuracy metrics unless verified reference text is supplied:

```powershell
.\.venv\Scripts\python.exe evaluation\evaluate_ocr.py --track scene-images --image "runtime_data\uploads\<saved-mobile-image>.png" --image "Data\scene_test\Smartlens Dataset\detection\D\image_1.jpg" --image "Data\scene_test\Smartlens Dataset\detection\D\image_100.jpg" --csv runtime_data\scene-image-evaluation.csv
```

The scene track reports full-frame and individual crop candidates but deliberately leaves CER/WER blank. It does not write to the dataset or reading history. For accessibility safety, Piper is not called when Tesseract confidence is zero/unavailable or when OCR contains no Devanagari letters/numerals. No positive-confidence cutoff is used because one has not been calibrated on verified scene transcriptions.

### Optional Devanagari recognizer comparison

The official `devanagari_PP-OCRv5_mobile_rec` inference model is stored separately from the PP-OCRv5 detector at:

```text
models/ppocr/devanagari_PP-OCRv5_mobile_rec_infer/devanagari_PP-OCRv5_mobile_rec_infer/
```

Its 8,069,120-byte archive was obtained from the official Paddle model host:
`https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv5_mobile_rec_infer.tar`.
The archive and model folder are Git-ignored. The application does not automatically load or download this recognizer; the production reading pipeline continues using Tesseract.

Run both recognizers over the same local printed-word images and, optionally, the same PP-OCRv5 detector crops from scene images:

```powershell
.\.venv\Scripts\python.exe evaluation\compare_recognizers.py --word-limit 10 --scene-image "runtime_data\uploads\<saved-mobile-image>.png" --scene-image "Data\scene_test\Smartlens Dataset\detection\D\image_1.jpg" --scene-image "Data\scene_test\Smartlens Dataset\detection\D\image_100.jpg" --scene-image "Data\scene_test\Smartlens Dataset\detection\D\image_1003.jpg"
```

Word-image metrics use the reference strings in the dataset's local label list (these have not been independently hand-verified). Scene text has no verified references; CER/WER are therefore blank. The scene comparison supplies each identical detector crop to both recognizers, while Tesseract applies its configured grayscale preprocessing and PaddleOCR receives the BGR crop. CSV output keeps word and scene tracks separate under `runtime_data/`. Paddle confidence is the model's `rec_score`, not a calibrated probability. Timings and process RSS are local observations, not isolated peak-memory measurements. Process CPU time is recorded only for PaddleOCR: Tesseract runs in a child process, so the parent process CPU clock would not make a valid comparison.

### Optional Colab model training

The separate notebooks `SmartLens_Recognition_Training.ipynb` and
`SmartLens_Detection_Training.ipynb` use the official PaddleOCR v3.7.0 source
revision and configs. Before opening either notebook in Colab, copy both
unchanged ZIP archives to `MyDrive/SmartLens-Indic/Data/` and copy the prepared
manifests from `dataset_work/` to the corresponding Drive folder. Colab cannot
read Windows `C:\` paths. The notebooks materialize only selected images under
ephemeral `/content/`, while checkpoints, logs, metrics, and exports are saved
to Drive.

The official Devanagari recognition recipe has no pretrained training
checkpoint configured, so that notebook trains from scratch; the existing
inference weights are not substituted for training weights. The detection
recipe uses its official pretrained PPLCNetV3 backbone (about 19.2 MiB).
Neither notebook starts training automatically. The Windows application uses
PaddlePaddle 3.0.0 CPU with PaddleOCR 3.7.0 / PaddleX 3.7.2, while Colab cells
select the official PaddlePaddle 3.3.0 GPU wheel for the runtime. An export
must pass the notebook CPU smoke test and a second smoke test under the
project's Windows versions before it is adopted. No training has been run. The
existing local inference models were compared on the
prepared labeled word-image test split (`evaluation/results/test_recognition.csv`);
those results do not estimate
real-world scene accuracy. Detection training is currently blocked by the
out-of-image polygons documented in `dataset_work/README.md`.

## Local data and privacy

After setup, image processing, OCR, text cleanup, Piper synthesis, and SQLite history operate locally. Uploaded/captured images and generated audio remain in `runtime_data/`. Avoid sharing that folder if it contains private images or text.
>>>>>>> c81e519 (Initial SmartLens-Indic project upload)
