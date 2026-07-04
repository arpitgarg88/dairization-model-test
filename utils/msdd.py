"""NVIDIA NeMo MSDD (Multi-Scale Diarization Decoder) helpers (CPU/GPU)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import soundfile as sf
from pyannote.core import Annotation
from tqdm import tqdm

from utils.dataset import DEFAULT_DATASET_DIR, load_dataset
from utils.eval import build_reference_annotation, compute_der
from utils.sortformer import load_mono_16k, load_mono_16k_from_bytes

# Force CPU when no GPU is available (NeMo may still pick CUDA if visible).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

DEFAULT_MSDD_MODEL = "diar_msdd_telephonic"
DEFAULT_VAD_MODEL = "vad_multilingual_marblenet"


def load_msdd_model(
    model_name: str = DEFAULT_MSDD_MODEL,
    *,
    vad_model_name: str = DEFAULT_VAD_MODEL,
    device: str | None = "cpu",
    verbose: bool = False,
):
    """Load the NeMo NeuralDiarizer (VAD + clustering + MSDD) pipeline."""
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer

    map_location = device or ("cuda" if _cuda_available() else "cpu")
    return NeuralDiarizer.from_pretrained(
        model_name,
        vad_model_name=vad_model_name,
        map_location=map_location,
        verbose=verbose,
    )


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _write_mono_16k_wav(audio_path: str | Path | None = None, *, audio_bytes: bytes | None = None) -> str:
    """Convert audio to mono 16 kHz WAV on disk (MSDD/VAD requirement)."""
    if audio_bytes is not None:
        waveform, sample_rate = load_mono_16k_from_bytes(audio_bytes)
    elif audio_path is not None:
        waveform, sample_rate = load_mono_16k(audio_path)
    else:
        raise ValueError("Provide audio_path or audio_bytes")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        sf.write(handle.name, waveform, sample_rate)
        return handle.name


def diarize_path(
    model,
    audio_path: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
    max_speakers: int | None = 8,
    out_dir: str | Path | None = None,
) -> Annotation:
    """Run the full MSDD pipeline on a single audio file path."""
    audio_path = Path(audio_path).resolve()
    temp_path = _write_mono_16k_wav(audio_path=audio_path)
    try:
        annotation = model(
            temp_path,
            batch_size=batch_size,
            num_workers=num_workers,
            max_speakers=max_speakers,
            out_dir=str(out_dir) if out_dir else None,
        )
        annotation.uri = audio_path.stem
        return annotation
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def diarize_bytes(
    model,
    audio_bytes: bytes,
    *,
    uri: str = "audio",
    batch_size: int = 64,
    num_workers: int = 0,
    max_speakers: int | None = 8,
) -> Annotation:
    """Run MSDD on in-memory WAV bytes (writes a temporary mono 16 kHz file for NeMo)."""
    temp_path = _write_mono_16k_wav(audio_bytes=audio_bytes)
    try:
        annotation = model(
            temp_path,
            batch_size=batch_size,
            num_workers=num_workers,
            max_speakers=max_speakers,
        )
        annotation.uri = uri
        return annotation
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_benchmark(
    model_name: str,
    result_filename: str,
    *,
    tqdm_desc: str,
    vad_model_name: str = DEFAULT_VAD_MODEL,
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
    device: str | None = "cpu",
    batch_size: int = 64,
    num_workers: int = 0,
    max_speakers: int | None = 8,
) -> dict:
    """Run DER benchmark on the parquet dataset (same eval settings as other scripts)."""
    print(f"Loading MSDD pipeline ({model_name}, VAD={vad_model_name}) ...")
    model = load_msdd_model(
        model_name,
        vad_model_name=vad_model_name,
        device=device,
    )
    samples = load_dataset(dataset_dir)
    if limit is not None:
        samples = samples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file: list[dict] = []
    total_ref_speech = 0.0

    for sample in tqdm(samples, desc=tqdm_desc):
        reference = build_reference_annotation(sample)

        print(f"\n[{sample.sample_id}] duration={sample.duration:.0f}s, starting MSDD ...", flush=True)
        t0 = time.time()
        hypothesis = diarize_bytes(
            model,
            sample.audio_bytes,
            uri=sample.sample_id,
            batch_size=batch_size,
            num_workers=num_workers,
            max_speakers=max_speakers,
        )
        elapsed = time.time() - t0
        print(f"[{sample.sample_id}] done in {elapsed:.1f}s", flush=True)

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

        # Checkpoint after each file — MSDD is slow on CPU and runs can take hours.
        partial = {
            "model": model_name,
            "vad_model": vad_model_name,
            "backend": "nemo-cpu" if (device or "cpu") == "cpu" else "nemo",
            "num_samples": len(samples),
            "completed_samples": len(per_file),
            "total_audio_duration_s": round(sum(s.duration for s in samples), 2),
            "evaluation": {"collar_s": 0.0, "skip_overlap": False},
            "per_file": per_file,
        }
        out_path = output_dir / result_filename
        out_path.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    weighted_der = sum(r["der_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    weighted_fa = sum(r["false_alarm_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    weighted_miss = (
        sum(r["missed_detection_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech
    )
    weighted_conf = sum(r["confusion_pct"] * r["evaluated_speech_s"] for r in per_file) / total_ref_speech

    backend = "nemo-cpu" if (device or "cpu") == "cpu" else "nemo"
    summary = {
        "model": model_name,
        "vad_model": vad_model_name,
        "backend": backend,
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
