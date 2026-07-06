"""Sarvam Saaras v3 batch STT + speaker diarization API helpers."""

from __future__ import annotations

import json
import os
import re
import struct
import tempfile
from pathlib import Path

import soundfile as sf
import librosa
from pyannote.core import Annotation, Segment
from sarvamai import SarvamAI

API_KEY_ENV = "SARVAM_API_KEY"
DEFAULT_MODEL = "saaras:v3"
DEFAULT_MODE = "transcribe"
# Sarvam batch diarization requires saaras:v3; use "unknown" for auto-detect.
DEFAULT_LANGUAGE = "unknown"


def get_api_key() -> str:
    token = os.environ.get(API_KEY_ENV, "").strip().strip('"').strip("'")
    if not token:
        raise RuntimeError(
            f"Sarvam API key not found. Set {API_KEY_ENV} in .env or your shell. "
            "Create a key at https://dashboard.sarvam.ai/"
        )
    return token


def get_client() -> SarvamAI:
    return SarvamAI(api_subscription_key=get_api_key())


def verify_api_key() -> str:
    """Lightweight auth check via the translation API."""
    client = get_client()
    response = client.text.translate(
        input="Hello",
        source_language_code="en-IN",
        target_language_code="hi-IN",
    )
    translated = getattr(response, "translated_text", None) or str(response)
    return translated


def _normalize_speaker_id(raw: str) -> str:
    match = re.search(r"(\d+)", str(raw))
    idx = int(match.group(1)) if match else 0
    return f"SPEAKER_{idx:02d}"


def _extract_diarized_entries(response: dict) -> list[dict]:
    diarized = response.get("diarized_transcript")
    if diarized is None:
        return []

    if isinstance(diarized, list):
        return diarized

    if isinstance(diarized, dict):
        entries = diarized.get("entries")
        if isinstance(entries, list):
            return entries
        # Some responses may nest segments under another key.
        for key in ("segments", "utterances", "results"):
            value = diarized.get(key)
            if isinstance(value, list):
                return value

    return []


def response_to_annotation(response: dict, *, uri: str = "audio") -> Annotation:
    """Build pyannote Annotation from Sarvam diarized_transcript entries."""
    annotation = Annotation(uri=uri)
    for entry in _extract_diarized_entries(response):
        start = float(entry["start_time_seconds"])
        end = float(entry["end_time_seconds"])
        speaker = _normalize_speaker_id(entry.get("speaker_id", "0"))
        annotation[Segment(start, end)] = speaker
    return annotation


def _wav_header_suspicious(path: Path) -> bool:
    """Detect WAV headers that confuse cloud parsers (e.g. RIFF size 0xFFFFFFFF)."""
    try:
        header = path.read_bytes()[:12]
        if len(header) < 12 or header[:4] != b"RIFF":
            return True
        riff_size = struct.unpack("<I", header[4:8])[0]
        if riff_size >= 0xFFFF_F000:
            return True
    except OSError:
        return True
    return False


def prepare_upload_wav(audio_path: Path) -> tuple[str, Path | None]:
    """Return a clean 16 kHz mono WAV path for Sarvam batch upload.

    Returns (upload_path, temp_path). Caller must delete temp_path when done.
    """
    audio_path = audio_path.resolve()
    data, sr = sf.read(str(audio_path), always_2d=True)
    mono = data.mean(axis=1)
    target_sr = 16000
    needs_resample = sr != target_sr
    needs_rewrite = _wav_header_suspicious(audio_path) or needs_resample

    if not needs_rewrite:
        return str(audio_path), None

    if needs_resample:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)

    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    temp_path = Path(handle.name)
    sf.write(temp_path, mono, target_sr, subtype="PCM_16")
    return str(temp_path), temp_path


def transcribe_diarize(
    audio: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    mode: str = DEFAULT_MODE,
    language_code: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
    poll_interval: int = 5,
    timeout: int = 900,
) -> dict:
    """Run Sarvam batch STT + diarization; return parsed output JSON."""
    audio_path = Path(audio).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    upload_path, temp_path = prepare_upload_wav(audio_path)
    try:
        client = get_client()
        job = client.speech_to_text_job.create_job(
            model=model,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            with_diarization=True,
            language_code=language_code,  # type: ignore[arg-type]
            num_speakers=num_speakers,
        )

        job.upload_files(file_paths=[upload_path])
        job.start()
        status = job.wait_until_complete(poll_interval=poll_interval, timeout=timeout)

        if status.job_state.lower() == "failed":
            raise RuntimeError(f"Sarvam batch job failed: {status}")

        file_results = job.get_file_results()
        failed = file_results.get("failed") or []
        if failed:
            raise RuntimeError(f"Sarvam batch file processing failed: {failed}")

        successful = file_results.get("successful") or []
        if not successful:
            raise RuntimeError("Sarvam batch job completed but returned no successful files.")

        with tempfile.TemporaryDirectory(prefix="sarvam-diar-") as tmp_dir:
            job.download_outputs(tmp_dir)
            uploaded_name = Path(upload_path).name
            output_path = Path(tmp_dir) / f"{uploaded_name}.json"
            if not output_path.exists():
                candidates = list(Path(tmp_dir).glob("*.json"))
                if len(candidates) == 1:
                    output_path = candidates[0]
                else:
                    raise FileNotFoundError(
                        f"Expected Sarvam output JSON for {uploaded_name} in {tmp_dir}, "
                        f"found: {[p.name for p in candidates]}"
                    )
            return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
