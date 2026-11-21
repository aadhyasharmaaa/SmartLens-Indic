# SmartLens-Indic: next-phase status

## Summary

- The Windows OCR pipeline and models ran locally; no model was downloaded and no training was started.
- The two training notebooks now save checkpoints every epoch and resume automatically only from a complete checkpoint.
- The detection notebook correctly stops at the annotation-validation gate. Detection training is **blocked** until the out-of-image annotations are reviewed.
- The original dataset ZIPs and annotation JSON were read-only throughout this work. No corrected manifest was created.
- The full test suite passes: **47 passed**.

## Detection annotation audit

The selected `BSTD_v17.57.json` was checked against every decoded detection image in the original ZIPs:

| Check | Result |
|---|---:|
| Detection images decoded | 1,218 |
| Images with polygons | 1,214 |
| Polygons checked | 19,264 |
| Polygons strictly outside image bounds | 191 |
| Images containing an out-of-bounds polygon | 52 |
| Vertices strictly outside bounds | 704 |
| Vertices exactly on the right/bottom geometric edge | 180 |
| Malformed polygons | 0 |

Coordinates equal to image width or height are treated as lying on the geometric edge and are accepted. Of the 704 invalid vertices, 621 are below the image; the largest y-coordinate is 1.294 times the image height. Some affected images likely have an annotation/image-size mismatch, but the JSON has no original per-image dimensions. The evidence cannot justify a safe per-row scale, clip, or axis correction. **No annotation was silently changed.**

The detailed audit is in [geometry_audit.md](./dataset_work/detection/geometry_audit.md), with all records in [geometry_audit.csv](./dataset_work/detection/geometry_audit.csv) and [geometry_audit.jsonl](./dataset_work/detection/geometry_audit.jsonl). Run it again with:

```powershell
.\.venv\Scripts\python.exe .\dataset_work\audit_detection_geometry.py
```

## Prepared split checks

- Recognition manifests contain 15,901 train / 2,044 validation / 2,055 test rows; all have non-empty transcriptions.
- Detection manifests contain 867 / 109 / 238 images. Four images without polygons are excluded rather than treated as verified negative examples.
- Pairwise comparisons of `group_id`, image SHA-256, and archive member show **zero overlap** between train, validation, and test for both tracks.
- Existing split manifests were not regenerated or changed.

## Google Drive and Colab

Both notebooks use the shared project root `MyDrive/SmartLens-Indic`. No Windows `C:\` data path is used by the Colab workflow. The notebooks were statically checked for valid JSON and Python syntax, but they were **not run in Colab**; Drive accessibility and GPU availability have not been verified.

### One-time transfer procedure

1. In Google Drive, create `MyDrive/SmartLens-Indic/Data/`.
2. Copy these two original ZIPs there once, without extracting or editing them:
   - `Smartlens Dataset-20261030T185508Z-1-001.zip`
   - `Smartlens Dataset-20261030T185508Z-1-002.zip`
3. Save both notebooks in `MyDrive/SmartLens-Indic/`:
   - `SmartLens_Recognition_Training.ipynb`
   - `SmartLens_Detection_Training.ipynb`
4. Copy only the recognition split files to `MyDrive/SmartLens-Indic/dataset_work/recognition/`:
   - `train.csv`, `validation.csv`, `test.csv`
5. Copy only the detection split files to `MyDrive/SmartLens-Indic/dataset_work/detection/`:
   - `train.jsonl`, `validation.jsonl`, `test.jsonl`, `excluded_no_polygons.jsonl`
6. Reuse the same two ZIPs for both notebooks. Do not upload extracted copies or duplicate archives.

Recognition and detection checkpoints/logs are written to `training/recognition/checkpoints/` and `training/detection/checkpoints/` in the project Drive folder. The notebooks save each epoch. After a runtime interruption, remount Drive and rerun setup and validation cells; training resumes only if all three `latest.pdparams`, `latest.pdopt`, and `latest.states` files exist. A partial checkpoint stops visibly instead of being treated as a fresh run. The detection notebook also stops before training if invalid polygons remain.

## Windows pipeline findings

Five existing scene images were run through the PP-OCRv5 detector without a detector exception. Saved overlays, crops, and scale metadata are under [scene_diagnostics](./output/scene_diagnostics/). The input-to-original coordinate scaling and crop extraction matched the detector boxes in the overlays.

The detector output is still not reliable enough for complex scenes:

- The five images produced 3, 5, 7, 8, and 7 regions, respectively, against 5, 21, 68, 27, and 27 annotated polygons. These counts are diagnostic only, not detection precision/recall.
- On `image_100.jpg`, one predicted crop spans from x=444 to the image's right edge (x=3,264) and includes multiple text/graphic areas. On the station scene, several small signs are missed while some non-Hindi text is detected.
- The signboard crops can mix languages/lines. This is consistent with overly broad detector regions and region merging; it is not evidence of an image-coordinate scaling or crop-indexing bug.
- Detector-only timings ranged from about 0.52 to 0.95 seconds on this Windows machine. They are not Raspberry Pi measurements and do not establish real-time performance.

An isolated Windows upload-processing smoke test on `image_1.jpg` completed detection and recognition for three crops. OCR returned noisy text (`रेशबाग क्` / `बलाहररबमलना`); Tesseract did not confirm every Paddle region, so the speech gate correctly **did not generate audio**. This demonstrates that poor OCR is not automatically spoken; it is not an accuracy result. No CER/WER was calculated for these scenes because manually verified references were not established.

The upload-processing service path was exercised, and automated tests cover application components. A real webcam permission grant and physical frame capture could not be verified: the available browser showed a camera-off placeholder, not a camera frame. The new webcam-only low-information-frame guard prevents such a placeholder from reaching OCR or speech. It does not affect uploaded images and is not an OCR confidence threshold.

The requested sentence, “नमस्ते, यह स्मार्टलेंस इंडिक हिंदी पाठ पढ़ने वाला सहायक है।”, produced Hindi phonemes and a valid mono 22,050 Hz WAV using the existing `hi_IN-rohan-medium` model/configuration. Subjective listening for intelligibility still requires a person. No voice replacement was needed.

## Ready, blocked, and next actions

**Ready**

- Windows launch and upload-processing code; automatic detection, crop creation, ordered recognition, conservative speech validation, and history/audio components.
- Hindi Piper phonemization and WAV generation for the test sentence.
- Both Colab notebooks' paths, dependency/configuration setup, dataset validation gates, per-epoch checkpointing, and resume logic are prepared.
- Existing recognition evaluation results were not repeated or changed.

**Blocked or not verified**

- Detection training is blocked by 191 polygons outside image bounds. The source archives do not provide enough information to correct them safely.
- No Colab run, GPU check, or Drive transfer has been performed.
- A physical webcam capture and subjective speech-intelligibility test remain unverified.
- Detector false positives, missed text, and overly broad crops remain; no recognition-accuracy improvement is claimed.

**Minimum actions**

1. For Colab, perform the one-time transfer above, then run each notebook only through its dataset validation cells first.
2. Review the 52 affected detection images and their records in the audit. Supply a documented corrected working annotation set if justified; do not edit the source JSON or ZIPs.
3. To verify the webcam, allow camera access in the browser and capture a real frame. Listen to the generated Hindi WAV to judge intelligibility.
4. Launch the Windows application with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Run the complete local tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
