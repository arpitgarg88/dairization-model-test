"""Deepgram Speech-to-Text + speaker diarization API helpers."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from pyannote.core import Annotation, Segment

API_URL = "https://api.deepgram.com/v1/listen"
API_KEY_ENV = "DEEPGRAM_API_KEY"

# Batch: nova-3 STT + diarization v2 (best batch diarizer as of 2026).
DEFAULT_MODEL = "nova-3"
DEFAULT_DIARIZE_MODEL = "v2"


def get_api_key() -> str:
    token = os.environ.get(API_KEY_ENV, "").strip().strip('"').strip("'")
    if not token:
        raise RuntimeError(
            f"Deepgram API key not found. Set {API_KEY_ENV} in .env or your shell. "
            "Create a key at https://console.deepgram.com/"
        )
    return token


def _listen_params(
    *,
    model: str = DEFAULT_MODEL,
    diarize_model: str = DEFAULT_DIARIZE_MODEL,
) -> str:
    params = {
        "model": model,
        "diarize_model": diarize_model,
        "utterances": "true",
        "punctuate": "true",
        "smart_format": "true",
    }
    return urllib.parse.urlencode(params)


def transcribe_diarize(
    audio: bytes | str | Path,
    *,
    model: str = DEFAULT_MODEL,
    diarize_model: str = DEFAULT_DIARIZE_MODEL,
    content_type: str = "audio/wav",
) -> dict:
    """Call Deepgram pre-recorded listen API; return parsed JSON response."""
    if isinstance(audio, (str, Path)):
        audio_path = Path(audio)
        audio_bytes = audio_path.read_bytes()
        if content_type == "audio/wav":
            content_type = _guess_content_type(audio_path)
    else:
        audio_bytes = audio

    url = f"{API_URL}?{_listen_params(model=model, diarize_model=diarize_model)}"
    request = urllib.request.Request(
        url,
        data=audio_bytes,
        method="POST",
        headers={
            "Authorization": f"Token {get_api_key()}",
            "Content-Type": content_type,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Deepgram API error {exc.code}: {body}") from exc


def _guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")


def utterances_to_annotation(response: dict, *, uri: str = "audio") -> Annotation:
    """Build pyannote Annotation from Deepgram utterances (speaker diarization segments)."""
    utterances = response.get("results", {}).get("utterances") or []
    annotation = Annotation(uri=uri)
    for utt in utterances:
        start = float(utt["start"])
        end = float(utt["end"])
        speaker_idx = int(utt["speaker"])
        speaker = f"SPEAKER_{speaker_idx:02d}"
        annotation[Segment(start, end)] = speaker
    return annotation
