"""Streaming Sortformer v2 (NeMo).

Usage:
  python diarize_sortformer_v2.py --audio test-audio/test_1.wav
  python diarize_sortformer_v2.py --benchmark --limit 5
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
from utils.sortformer import RESULTS_DIR, diarize_path, load_sortformer_model, run_benchmark

MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2"
MODEL_SLUG = "diar-streaming-sortformer-4spk-v2"
BACKEND = "nemo-cpu"
BENCHMARK_RESULT = "benchmark_diar_streaming_sortformer_4spk_v2.json"

load_dotenv()


def print_diarization(annotation, max_segments: int = 20) -> None:
    print(f"Speakers detected: {len(set(annotation.labels()))}")
    print(f"Segments: {len(list(annotation.itertracks()))}")
    for idx, (turn, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        if idx >= max_segments:
            print("  ...")
            break
        print(f"  {speaker}: {turn.start:.2f}s - {turn.end:.2f}s")


def run_smoke_test(audio_path: Path, device: str | None = None) -> None:
    audio_path = audio_path.resolve()
    print(f"Loading {MODEL_ID} on CPU ...")
    model = load_sortformer_model(
        MODEL_ID,
        device=device or "cpu",
        streaming_offline=True,
    )
    print(f"Running diarization on {audio_path} ...")
    t0 = time.time()
    annotation = diarize_path(model, audio_path)
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
    parser = argparse.ArgumentParser(description="NVIDIA Streaming Sortformer 4spk v2")
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
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark(
            MODEL_ID,
            BENCHMARK_RESULT,
            tqdm_desc="diar-streaming-sortformer-4spk-v2",
            streaming_offline=True,
            dataset_dir=args.dataset_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            device=args.device,
        )
    elif args.audio:
        run_smoke_test(args.audio, device=args.device)
    else:
        run_smoke_test(PROJECT_ROOT / "test-audio" / "test_1.wav", device=args.device)


if __name__ == "__main__":
    main()
