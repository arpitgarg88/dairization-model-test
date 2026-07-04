"""pyannote community-1 — local diarization.

Usage:
  python diarize_community_1.py --audio test-audio/test_1.wav
  python diarize_community_1.py --benchmark --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import get_token
from pyannote.audio import Pipeline
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.audio import load_audio_bytes, load_audio_file
from utils.dataset import DEFAULT_DATASET_DIR, load_dataset
from utils.env import load_dotenv
from utils.eval import (
    build_reference_annotation,
    compute_der,
    extract_annotation,
    save_diarization_results,
)

MODEL_ID = "pyannote/speaker-diarization-community-1"
MODEL_SLUG = "speaker-diarization-community-1"
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv()


def load_pipeline(device: str | None = None) -> Pipeline:
    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Run `huggingface-cli login` or set HF_TOKEN. "
            "You must also accept the model terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1"
        )

    pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(torch.device(target))
    return pipeline


def diarize(pipeline: Pipeline, audio_input: dict | str | Path, **kwargs):
    if isinstance(audio_input, (str, Path)):
        audio_input = load_audio_file(audio_input)
    output = pipeline(audio_input, **kwargs)
    return extract_annotation(output)


def print_diarization(annotation, max_segments: int = 20) -> None:
    print(f"Speakers detected: {len(set(annotation.labels()))}")
    print(f"Segments: {len(list(annotation.itertracks()))}")
    for idx, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        if idx >= max_segments:
            print("  ...")
            break
        print(f"  {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")


def run_smoke_test(audio_path: Path) -> None:
    audio_path = audio_path.resolve()
    print(f"Loading {MODEL_ID} ...")
    pipeline = load_pipeline()
    print(f"Running diarization on {audio_path} ...")
    t0 = time.time()
    annotation = diarize(pipeline, audio_path)
    elapsed = time.time() - t0
    json_path, rttm_path = save_diarization_results(
        annotation,
        audio_path,
        MODEL_SLUG,
        model_id=MODEL_ID,
        inference_s=elapsed,
    )
    print(f"Done in {elapsed:.1f}s")
    print_diarization(annotation)
    print(f"Saved JSON: {json_path}")
    print(f"Saved RTTM: {rttm_path}")


def run_benchmark(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    print(f"Loading {MODEL_ID} ...")
    pipeline = load_pipeline()
    samples = load_dataset(dataset_dir)
    if limit is not None:
        samples = samples[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file: list[dict] = []
    total_ref_speech = 0.0

    for sample in tqdm(samples, desc="speaker-diarization-community-1"):
        audio = load_audio_bytes(sample.audio_bytes)
        reference = build_reference_annotation(sample)

        t0 = time.time()
        hypothesis = diarize(pipeline, audio)
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
        "model": MODEL_ID,
        "pyannote_audio": "4.0.7",
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

    out_path = output_dir / "benchmark_speaker_diarization_community_1.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark summary ({len(samples)} files):")
    print(f"  DER:  {summary['der_pct']:.2f}%")
    print(f"  FA:   {summary['false_alarm_pct']:.2f}%")
    print(f"  Miss: {summary['missed_detection_pct']:.2f}%")
    print(f"  Conf: {summary['confusion_pct']:.2f}%")
    print(f"Results saved to {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="pyannote speaker-diarization-community-1")
    parser.add_argument("--audio", type=Path, help="Path to a WAV file for smoke testing")
    parser.add_argument("--benchmark", action="store_true", help="Run DER benchmark on dataset/")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark to N samples")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(args.dataset_dir, limit=args.limit, output_dir=args.output_dir)
    elif args.audio:
        run_smoke_test(args.audio)
    else:
        default_audio = PROJECT_ROOT / "test-audio" / "test_1.wav"
        run_smoke_test(default_audio)


if __name__ == "__main__":
    main()
