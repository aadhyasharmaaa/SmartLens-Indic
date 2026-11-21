from __future__ import annotations

import re
import uuid
import wave
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_DATA_DIR = ROOT_DIR / "models" / "voices"
DEFAULT_MODEL_NAME = "hi_IN-rohan-medium"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "output" / "spoken_hindi.wav"


def _has_speech_phonemes(sentences: list[list[str]]) -> bool:
    return any(
        character.isalpha()
        for sentence in sentences
        for phoneme in sentence
        for character in phoneme
    )


def speak_hindi(text: str, voice_path: str | Path | None = None) -> tuple[bool, str]:
    voice = Path(voice_path) if voice_path is not None else DEFAULT_VOICE_DATA_DIR / f"{DEFAULT_MODEL_NAME}.onnx"
    return generate_hindi_audio(
        text,
        output_path=DEFAULT_OUTPUT_PATH,
        data_dir=voice.parent,
        model_name=voice.stem,
    )


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

    voice_model = voice_dir / f"{model_name}.onnx"
    voice_config = Path(f"{voice_model}.json")
    if not voice_model.is_file():
        return False, f"Piper voice model was not found: {voice_model}"
    if not voice_config.is_file():
        return False, f"Piper voice configuration was not found: {voice_config}"

    final_path = out_path
    if final_path.exists():
        final_path = out_path.with_name(f"{out_path.stem}-{uuid.uuid4().hex[:8]}{out_path.suffix}")
    temporary_path = out_path.with_name(f".{out_path.stem}-{uuid.uuid4().hex}.tmp.wav")

    try:
        from piper import PiperVoice

        voice = PiperVoice.load(str(voice_model))
        text_to_speak = text.strip()
        hindi_segments = re.findall(r"[\u0900-\u097f]+", text_to_speak)
        for segment in hindi_segments:
            if not _has_speech_phonemes(voice.phonemize(segment)):
                return (
                    False,
                    "Piper could not produce Hindi phonemes for the supplied text; "
                    "no audio was generated.",
                )
        if not _has_speech_phonemes(voice.phonemize(text_to_speak)):
            return False, "Piper produced no speech phonemes; no audio was generated."

        with wave.open(str(temporary_path), "wb") as audio_file:
            voice.synthesize_wav(text_to_speak, audio_file)

        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            return False, "Piper completed without creating a non-empty WAV file."

        try:
            with wave.open(str(temporary_path), "rb") as audio:
                if audio.getnchannels() < 1 or audio.getframerate() < 1 or audio.getnframes() < 1:
                    return False, "Piper created an invalid or empty WAV file."
        except (wave.Error, EOFError) as exc:
            return False, f"Piper created an unreadable WAV file: {exc}"

        temporary_path.replace(final_path)
        return True, str(final_path)
    except Exception as exc:
        return False, f"Piper speech generation failed: {exc}"
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
