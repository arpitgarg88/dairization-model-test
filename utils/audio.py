"""Audio loading helpers (works without FFmpeg via in-memory waveforms)."""

from __future__ import annotations

import io
from pathlib import Path

import soundfile as sf
import torch


def load_audio_file(path: str | Path) -> dict:
    """Load a WAV/audio file into pyannote's in-memory format."""
    data, sample_rate = sf.read(str(path), always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sample_rate}


def load_audio_bytes(audio_bytes: bytes) -> dict:
    """Load audio bytes (e.g. from parquet) into pyannote's in-memory format."""
    data, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=True)
    waveform = torch.from_numpy(data.T).float()
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sample_rate}
