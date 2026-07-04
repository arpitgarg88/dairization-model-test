"""Sortformer v2.1 + Nemotron 3.5 ASR per segment.

Usage:
  python transcribe_nemotron35_pipeline.py --test-audio-dir test-audio --output-dir results/nemotron-3.5-pipeline
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_dotenv
from utils.nemotron35_pipeline import (
    ASR_MODEL_ID,
    DIAR_MODEL_ID,
    load_nemotron35_asr,
    run_pipeline,
    save_pipeline_results,
)
from utils.sortformer import load_sortformer_model

load_dotenv()


def iter_wav_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sortformer diarization + Nemotron 3.5 transcription pipeline",
    )
    parser.add_argument("--audio", type=Path, help="Single WAV file")
    parser.add_argument(
        "--test-audio-dir",
        type=Path,
        default=PROJECT_ROOT / "test-audio",
        help="Process all *.wav files in this folder",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        help="Nemotron language prompt: auto, en-US, hi-IN, etc.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for JSON outputs (default: next to each WAV file)",
    )
    args = parser.parse_args()

    if args.audio:
        audio_files = [args.audio.resolve()]
    else:
        audio_files = iter_wav_files(args.test_audio_dir.resolve())
        if not audio_files:
            print(f"No .wav files found in {args.test_audio_dir}")
            sys.exit(1)

    print(f"Loading diarization model: {DIAR_MODEL_ID}")
    diar_model = load_sortformer_model(
        DIAR_MODEL_ID,
        device=args.device,
        streaming_offline=True,
    )

    print(f"Loading ASR model: {ASR_MODEL_ID} (first run downloads ~600M weights)")
    asr_processor, asr_model = load_nemotron35_asr(device=args.device)

    for audio_path in audio_files:
        print(f"\n=== {audio_path.name} ===")
        t0 = time.time()
        payload = run_pipeline(
            audio_path,
            language=args.language,
            device=args.device,
            diar_model=diar_model,
            asr_processor=asr_processor,
            asr_model=asr_model,
        )
        payload["inference_s"]["wall_clock"] = round(time.time() - t0, 2)
        out_path = save_pipeline_results(payload, audio_path, output_dir=args.output_dir)
        print(f"Speakers: {payload['num_speakers']} | Segments: {payload['num_segments']}")
        print(f"Diar: {payload['inference_s']['diarization']}s | ASR: {payload['inference_s']['transcription']}s")
        for seg in payload["segments"]:
            if not seg["text"]:
                continue
            line = f"  [{seg['start']:.2f}-{seg['end']:.2f}s] {seg['speaker']}: {seg['text']}"
            try:
                print(line)
            except UnicodeEncodeError:
                print(line.encode("ascii", errors="replace").decode("ascii"))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
