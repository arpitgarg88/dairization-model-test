"""Sortformer diarization + Nemotron 3.5 ASR (composed pipeline)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from pyannote.core import Annotation

from utils.sortformer import (
    TARGET_SAMPLE_RATE,
    diarize_path,
    load_mono_16k,
    load_sortformer_model,
)

DIAR_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
ASR_MODEL_ID = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MIN_SEGMENT_S = 0.08


@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    text: str

    @property
    def start_ms(self) -> int:
        return int(round(self.start * 1000))

    @property
    def end_ms(self) -> int:
        return int(round(self.end * 1000))

    def to_dict(self) -> dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "text": self.text,
        }


def annotation_to_segments(annotation: Annotation) -> list[tuple[str, float, float]]:
    rows: list[tuple[str, float, float]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        rows.append((speaker, turn.start, turn.end))
    rows.sort(key=lambda row: (row[1], row[2]))
    return rows


def load_nemotron35_asr(*, device: str = "cpu"):
    """Load Nemotron 3.5 via Transformers."""
    from transformers import AutoModelForRNNT, AutoProcessor

    processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)
    model = AutoModelForRNNT.from_pretrained(ASR_MODEL_ID)
    target = torch.device(device)
    model = model.to(target)
    model.eval()
    return processor, model


def transcribe_chunk(
    processor,
    model,
    waveform: np.ndarray,
    *,
    sample_rate: int,
    language: str,
    device: str,
) -> str:
    """Transcribe a mono float32 waveform slice."""
    if waveform.size == 0:
        return ""
    duration_s = waveform.size / sample_rate
    if duration_s < MIN_SEGMENT_S:
        return ""

    inputs = processor(waveform, sampling_rate=sample_rate, language=language)
    target = torch.device(device)
    model_dtype = next(model.parameters()).dtype
    inputs = inputs.to(target, dtype=model_dtype)

    with torch.inference_mode():
        output = model.generate(**inputs, return_dict_in_generate=True)

    text = processor.decode(output.sequences, skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0] if text else ""
    return str(text).strip()


def run_pipeline(
    audio_path: Path,
    *,
    language: str = "auto",
    device: str = "cpu",
    diar_model=None,
    asr_processor=None,
    asr_model=None,
) -> dict:
    """Diarize then transcribe each segment."""
    audio_path = audio_path.resolve()
    waveform, sample_rate = load_mono_16k(audio_path)

    owns_diar = diar_model is None
    if owns_diar:
        diar_model = load_sortformer_model(
            DIAR_MODEL_ID,
            device=device,
            streaming_offline=True,
        )

    t0 = time.time()
    annotation = diarize_path(diar_model, audio_path)
    diar_s = time.time() - t0

    owns_asr = asr_processor is None or asr_model is None
    if owns_asr:
        asr_processor, asr_model = load_nemotron35_asr(device=device)

    raw_segments = annotation_to_segments(annotation)
    results: list[DiarizedSegment] = []
    t_asr0 = time.time()
    for speaker, start, end in raw_segments:
        start_idx = max(0, int(start * sample_rate))
        end_idx = min(waveform.size, int(end * sample_rate))
        chunk = waveform[start_idx:end_idx]
        text = transcribe_chunk(
            asr_processor,
            asr_model,
            chunk,
            sample_rate=sample_rate,
            language=language,
            device=device,
        )
        results.append(DiarizedSegment(speaker=speaker, start=start, end=end, text=text))
    asr_s = time.time() - t_asr0

    speakers = sorted(set(r.speaker for r in results))
    full_transcript = " ".join(r.text for r in results if r.text)

    return {
        "pipeline": "sortformer-v2.1 + nemotron-3.5-asr",
        "diar_model": DIAR_MODEL_ID,
        "asr_model": ASR_MODEL_ID,
        "target_lang": language,
        "audio_file": audio_path.name,
        "backend": f"nemo-{device}+transformers-{device}",
        "inference_s": {
            "total": round(diar_s + asr_s, 2),
            "diarization": round(diar_s, 2),
            "transcription": round(asr_s, 2),
        },
        "num_speakers": len(speakers),
        "num_segments": len(results),
        "speakers": speakers,
        "segments": [segment.to_dict() for segment in results],
        "full_transcript": full_transcript,
    }


def save_pipeline_results(
    payload: dict,
    audio_path: Path,
    *,
    slug: str = "nemotron-3.5-pipeline",
    output_dir: Path | None = None,
) -> Path:
    """Write JSON output next to the source WAV or under output_dir."""
    audio_path = Path(audio_path)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{audio_path.stem}.{slug}.json"
    else:
        out_path = audio_path.parent / f"{audio_path.stem}.{slug}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path
