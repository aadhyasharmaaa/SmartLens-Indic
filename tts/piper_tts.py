from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_DATA_DIR = ROOT_DIR / "models" / "voices"
DEFAULT_MODEL_NAME = "hi_IN-rohan-medium"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "output" / "spoken_hindi.wav"


def speak_hindi(text: str, voice_path: str | Path | None = None) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "No Hindi text to speak."

    cli_path = shutil.which("piper")
    if cli_path is None:
        return False, "Piper CLI was not found. Hindi speech output remains pending until the Piper hi_IN voice model is installed."

    if voice_path is None:
        return False, "No Piper voice file path was provided. The hi_IN voice model is required for speech output."

    voice = Path(voice_path)
    if not voice.exists():
        return False, f"Piper voice file not found: {voice}"

    try:
        subprocess.run([cli_path, "--model", str(voice), "--output_file", str(Path("output") / "spoken_hindi.wav")], check=False)
        return True, "Piper speech-generation command was executed."
    except Exception as exc:  # pragma: no cover
        return False, f"Piper execution failed: {exc}"


def generate_hindi_audio(
    text: str,
    output_path: str | Path | None = None,
    data_dir: str | Path | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[bool, str]:
    if not text or not text.strip():
        return False, "OCR returned empty Hindi text; audio generation was skipped."

    out_path = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    voice_dir = Path(data_dir) if data_dir is not None else DEFAULT_VOICE_DATA_DIR
    if not voice_dir.exists():
        return False, f"Piper voice directory not found: {voice_dir}"

    try:
        cmd = [
            sys.executable,
            "-m",
            "piper",
            "-m",
            model_name,
            "--data-dir",
            str(voice_dir),
            "-f",
            str(out_path),
            "--",
            text.strip(),
        ]
        subprocess.run(cmd, check=False)
        if out_path.exists() and out_path.stat().st_size > 0:
            return True, str(out_path)
        return False, f"Audio file was not generated: {out_path}"
    except Exception as exc:  # pragma: no cover
        return False, f"Piper audio generation failed: {exc}"
