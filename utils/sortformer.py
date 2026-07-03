"""NVIDIA NeMo Sortformer diarization helpers (CPU/GPU)."""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from pyannote.core import Annotation, Segment
from tqdm import tqdm

from utils.dataset import DEFAULT_DATASET_DIR, load_dataset
from utils.eval import build_reference_annotation, compute_der

# Force CPU when no GPU is available (NeMo may still pick CUDA if visible).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

TARGET_SAMPLE_RATE = 16000
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

# Offline-style streaming config (30.4 s buffer) from the v2.1 model card.
STREAMING_OFFLINE_CHUNK_LEN = 340
STREAMING_OFFLINE_RIGHT_CONTEXT = 40
STREAMING_OFFLINE_FIFO_LEN = 40
STREAMING_OFFLINE_SPKCACHE_UPDATE_PERIOD = 300


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_mono_16k(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 at 16 kHz (Sortformer input requirement)."""
    waveform, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32), TARGET_SAMPLE_RATE


def load_mono_16k_from_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Load embedded WAV bytes as mono float32 at 16 kHz."""
    waveform, _ = librosa.load(io.BytesIO(audio_bytes), sr=TARGET_SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32), TARGET_SAMPLE_RATE


def load_sortformer_model(
    model_id: str,
    *,
    device: str | None = None,
    streaming_offline: bool = False,
):
    """Load a SortformerEncLabelModel from Hugging Face onto CPU or GPU."""
    from nemo.collections.asr.models import SortformerEncLabelModel

    target = resolve_device(device)
    model = SortformerEncLabelModel.from_pretrained(model_id, map_location=target)
    model.eval()

    if streaming_offline:
        configure_streaming_offline(model)

    return model


def configure_streaming_offline(model) -> None:
    """Use high-latency streaming settings for best offline accuracy (v2.x)."""
    modules = model.sortformer_modules
    modules.chunk_len = STREAMING_OFFLINE_CHUNK_LEN
    modules.chunk_right_context = STREAMING_OFFLINE_RIGHT_CONTEXT
    modules.fifo_len = STREAMING_OFFLINE_FIFO_LEN
    modules.spkcache_update_period = STREAMING_OFFLINE_SPKCACHE_UPDATE_PERIOD
    if hasattr(modules, "_check_streaming_parameters"):
        modules._check_streaming_parameters()


def segments_to_annotation(segments: list[str], uri: str = "audio") -> Annotation:
    """Convert NeMo segment strings ('start end speaker_N') to pyannote Annotation."""
    annotation = Annotation(uri=uri)
    for line in segments:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        start, end, speaker = float(parts[0]), float(parts[1]), parts[2]
        annotation[Segment(start, end)] = speaker
    return annotation


def diarize_path(model, audio_path: str | Path) -> Annotation:
    """Run Sortformer on a single audio file path."""
    waveform, sample_rate = load_mono_16k(audio_path)
    return diarize_waveform(model, waveform, sample_rate, uri=Path(audio_path).stem)


def diarize_waveform(
    model,
    waveform: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    *,
    uri: str = "audio",
) -> Annotation:
    """Run Sortformer on an in-memory mono waveform."""
    predicted = model.diarize(audio=[waveform], batch_size=1, sample_rate=sample_rate)
    if not predicted:
        return Annotation(uri=uri)
    return segments_to_annotation(predicted[0], uri=uri)


def run_benchmark(
    model_id: str,
    result_filename: str,
    *,
    tqdm_desc: str,
    streaming_offline: bool = False,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
    device: str | None = "cpu",
) -> dict:
    """Run DER benchmark on the parquet dataset (same eval settings as pyannote scripts)."""
    print(f"Loading {model_id} ...")
    model = load_sortformer_model(
        model_id,
        device=device,
        streaming_offline=streaming_offline,
    )
    samples = load_dataset(dataset_dir)
    if limit is not None:
        samples = samples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file: list[dict] = []
    total_ref_speech = 0.0

    for sample in tqdm(samples, desc=tqdm_desc):
        waveform, sample_rate = load_mono_16k_from_bytes(sample.audio_bytes)
        reference = build_reference_annotation(sample)

        t0 = time.time()
        hypothesis = diarize_waveform(
            model,
            waveform,
            sample_rate,
            uri=sample.sample_id,
        )
        elapsed = time.time() - t0

        der = compute_der(reference, hypothesis)
        total_ref_speech += der.total
        per_file.append(
            {
                "sample_id": sample.sample_id,
                "duration_s": sample.duration,
                "ref_speakers": sample.num_speakers,
                "hyp_speakers": len(set(hypothesis.labels())),
                "inference_s": round(elapsed, 2),
                "der_pct": round(der.der, 3),
                "false_alarm_pct": round(der.false_alarm, 3),
                "missed_detection_pct": round(der.missed_detection, 3),
                "confusion_pct": round(der.confusion, 3),
                "evaluated_speech_s": round(der.total, 3),
            }
        )

    weighted_der = sum(r["der_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    weighted_fa = sum(r["false_alarm_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    weighted_miss = (
        sum(r["missed_detection_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    )
    weighted_conf = sum(r["confusion_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech

    summary = {
        "model": model_id,
        "backend": "nemo-cpu" if (device or "cpu") == "cpu" else "nemo",
        "num_samples": len(samples),
        "total_audio_duration_s": round(sum(s.duration for s in samples), 2),
        "total_evaluated_speech_s": round(total_ref_speech, 2),
        "der_pct": round(weighted_der, 3),
        "false_alarm_pct": round(weighted_fa, 3),
        "missed_detection_pct": round(weighted_miss, 3),
        "confusion_pct": round(weighted_conf, 3),
        "evaluation": {"collar_s": 0.0, "skip_overlap": False},
        "per_file": per_file,
    }

    out_path = output_dir / result_filename
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark summary ({len(samples)} files):")
    print(f"  DER:  {summary['der_pct']:.2f}%")
    print(f"  FA:   {summary['false_alarm_pct']:.2f}%")
    print(f"  Miss: {summary['missed_detection_pct']:.2f}%")
    print(f"  Conf: {summary['confusion_pct']:.2f}%")
    print(f"Results saved to {out_path}")
    return summary
