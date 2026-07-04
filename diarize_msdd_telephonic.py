"""NVIDIA MSDD telephonic — NeMo cascaded diarization.

Usage:
  python diarize_msdd_telephonic.py --audio test-audio/test_1.wav
  python diarize_msdd_telephonic.py --benchmark --limit 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataset import DEFAULT_DATASET_DIR
from utils.env import load_dotenv
from utils.eval import save_diarization_results
from utils.msdd import (
    DEFAULT_MSDD_MODEL,
    DEFAULT_VAD_MODEL,
    RESULTS_DIR,
    diarize_path,
    load_msdd_model,
    run_benchmark,
)

MODEL_ID = DEFAULT_MSDD_MODEL
MODEL_SLUG = "diar-msdd-telephonic"
BACKEND = "nemo-cpu"
BENCHMARK_RESULT = "benchmark_diar_msdd_telephonic.json"

load_dotenv()


def print_diarization(annotation, max_segments: int = 20) -> None:
    print(f"Speakers detected: {len(set(annotation.labels()))}")
    print(f"Segments: {len(list(annotation.itertracks()))}")
    for idx, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        if idx >= max_segments:
            print("  ...")
            break
        print(f"  {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")


def run_smoke_test(
    audio_path: Path,
    *,
    device: str | None = None,
    vad_model: str = DEFAULT_VAD_MODEL,
    batch_size: int = 64,
    num_workers: int = 0,
    max_speakers: int | None = 8,
) -> None:
    audio_path = audio_path.resolve()
    print(f"Loading MSDD pipeline ({MODEL_ID}, VAD={vad_model}) on {device or 'cpu'} ...")
    model = load_msdd_model(
        MODEL_ID,
        vad_model_name=vad_model,
        device=device or "cpu",
    )
    print(f"Running diarization on {audio_path} ...")
    t0 = time.time()
    annotation = diarize_path(
        model,
        audio_path,
        batch_size=batch_size,
        num_workers=num_workers,
        max_speakers=max_speakers,
    )
    elapsed = time.time() - t0
    json_path, rttm_path = save_diarization_results(
        annotation,
        audio_path,
        MODEL_SLUG,
        model_id=MODEL_ID,
        inference_s=elapsed,
    )
    print(f"Done in {elapsed:.1f}s ({BACKEND})")
    print_diarization(annotation)
    print(f"Saved JSON: {json_path}")
    print(f"Saved RTTM: {rttm_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA NeMo MSDD telephonic diarization")
    parser.add_argument("--audio", type=Path, help="Path to a WAV file for smoke testing")
    parser.add_argument("--benchmark", action="store_true", help="Run DER benchmark on dataset/")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark to N samples")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--vad-model",
        type=str,
        default=DEFAULT_VAD_MODEL,
        help="NeMo VAD model name (default: vad_multilingual_marblenet)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for MSDD inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (use 0 on Windows)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Max speakers for clustering (CallHome has 2-4)",
    )
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            MODEL_ID,
            BENCHMARK_RESULT,
            tqdm_desc=MODEL_SLUG,
            vad_model_name=args.vad_model,
            dataset_dir=args.dataset_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_speakers=args.max_speakers,
        )
    elif args.audio:
        run_smoke_test(
            args.audio,
            device=args.device,
            vad_model=args.vad_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_speakers=args.max_speakers,
        )
    else:
        run_smoke_test(
            PROJECT_ROOT / "test-audio" / "test_1.wav",
            device=args.device,
            vad_model=args.vad_model,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_speakers=args.max_speakers,
        )


if __name__ == "__main__":
    main()
