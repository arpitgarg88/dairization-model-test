"""Sarvam Saaras v3 batch STT + speaker diarization (cloud API).

Diarization is available only via the Sarvam Batch API (async job workflow).

Usage:
  python diarize_sarvam_saaras.py --audio test-audio/hindi-test.wav
  python diarize_sarvam_saaras.py --audio test-audio/hindi-test.wav --language hi-IN
  python diarize_sarvam_saaras.py --benchmark --limit 5 --yes
  python diarize_sarvam_saaras.py --benchmark --limit 5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache import load_cached_annotation, save_cached_annotation
from utils.dataset import DEFAULT_DATASET_DIR, load_dataset
from utils.env import load_dotenv
from utils.eval import build_reference_annotation, compute_der, save_diarization_results
from utils.sarvam import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODE,
    DEFAULT_MODEL,
    get_api_key,
    response_to_annotation,
    transcribe_diarize,
    verify_api_key,
)

MODEL_ID = f"sarvam/{DEFAULT_MODEL}+diarization"
MODEL_SLUG = "sarvam-saaras-v3-diarize"
BACKEND = "sarvam-cloud-batch"
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_RESULT = "benchmark_sarvam_saaras_v3_diarize.json"
# ₹45/hour for STT + diarization (docs.sarvam.ai pricing).
INR_PER_HOUR = 45.0

load_dotenv()


def diarize_via_api(
    file_arg: str | Path,
    *,
    language_code: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
) -> tuple[object, float]:
    """Run Sarvam batch diarization. Returns (annotation, elapsed_seconds)."""
    t0 = time.time()
    response = transcribe_diarize(
        file_arg,
        language_code=language_code,
        num_speakers=num_speakers,
    )
    uri = Path(file_arg).stem if isinstance(file_arg, (str, Path)) else "audio"
    annotation = response_to_annotation(response, uri=uri)
    return annotation, time.time() - t0


def get_annotation(
    *,
    cache_key: str,
    file_arg: str | Path | None = None,
    use_cache: bool = True,
    force: bool = False,
    language_code: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
) -> tuple[object, float, bool]:
    """Return (annotation, elapsed_s, api_was_called)."""
    if use_cache and not force:
        cached = load_cached_annotation(MODEL_SLUG, cache_key)
        if cached is not None:
            return cached, 0.0, False

    if file_arg is None:
        raise ValueError("file_arg required when cache miss")

    annotation, elapsed = diarize_via_api(
        file_arg,
        language_code=language_code,
        num_speakers=num_speakers,
    )
    save_cached_annotation(
        annotation,
        MODEL_SLUG,
        cache_key,
        model_id=MODEL_ID,
        inference_s=elapsed,
    )
    return annotation, elapsed, True


def _bytes_to_temp_wav(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        handle.write(audio_bytes)
        return handle.name


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
    use_cache: bool = True,
    force: bool = False,
    language_code: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
) -> None:
    audio_path = audio_path.resolve()
    cache_key = audio_path.stem

    print(
        f"Running Sarvam {DEFAULT_MODEL} batch diarization on {audio_path} "
        f"(language={language_code or 'auto'}) ..."
    )
    annotation, elapsed, used_api = get_annotation(
        cache_key=cache_key,
        file_arg=audio_path,
        use_cache=use_cache,
        force=force,
        language_code=language_code,
        num_speakers=num_speakers,
    )

    json_path, rttm_path = save_diarization_results(
        annotation,
        audio_path,
        MODEL_SLUG,
        model_id=MODEL_ID,
        inference_s=elapsed if used_api else None,
    )

    if used_api:
        print(f"Done in {elapsed:.1f}s (batch API job completed)")
    else:
        print("Done (loaded from cache — no API call)")

    print_diarization(annotation)
    print(f"Saved JSON: {json_path}")
    print(f"Saved RTTM: {rttm_path}")


def preview_benchmark(samples: list, use_cache: bool, force: bool) -> tuple[int, int]:
    api_calls = 0
    cached = 0
    for sample in samples:
        if use_cache and not force and load_cached_annotation(MODEL_SLUG, sample.sample_id) is not None:
            cached += 1
        else:
            api_calls += 1
    return api_calls, cached


def run_benchmark(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    limit: int | None = None,
    output_dir: Path = RESULTS_DIR,
    *,
    use_cache: bool = True,
    force: bool = False,
    dry_run: bool = False,
    language_code: str | None = DEFAULT_LANGUAGE,
    num_speakers: int | None = None,
) -> dict | None:
    samples = load_dataset(dataset_dir)
    if limit is not None:
        samples = samples[:limit]

    api_calls, cached_hits = preview_benchmark(samples, use_cache=use_cache, force=force)

    print(f"Benchmark plan: {len(samples)} files, {api_calls} batch job(s), {cached_hits} cached")
    total_audio_s = sum(s.duration for s in samples)
    print(f"Total audio duration: {total_audio_s / 60:.1f} min")
    print(f"Model: {DEFAULT_MODEL}, mode={DEFAULT_MODE}, language={language_code or 'auto'}")

    if dry_run:
        est_inr = total_audio_s / 3600 * INR_PER_HOUR
        print("Dry run only — no API calls made.")
        print(f"Estimated cost (list price): ~INR {est_inr:.2f}")
        return None

    if api_calls == 0:
        print("All samples are cached. Re-run with --force to call the API again.")
    elif api_calls > 0:
        print(f"WARNING: {api_calls} paid batch job(s) will be submitted.")

    output_dir.mkdir(parents=True, exist_ok=True)
    per_file: list[dict] = []
    total_ref_speech = 0.0
    calls_made = 0

    for sample in tqdm(samples, desc=MODEL_SLUG):
        reference = build_reference_annotation(sample)
        temp_path: str | None = None

        try:
            if not force and use_cache and load_cached_annotation(MODEL_SLUG, sample.sample_id):
                annotation, elapsed, used_api = get_annotation(
                    cache_key=sample.sample_id,
                    use_cache=True,
                    force=False,
                    language_code=language_code,
                    num_speakers=num_speakers,
                )
            else:
                temp_path = _bytes_to_temp_wav(sample.audio_bytes)
                annotation, elapsed, used_api = get_annotation(
                    cache_key=sample.sample_id,
                    file_arg=temp_path,
                    use_cache=use_cache,
                    force=force,
                    language_code=language_code,
                    num_speakers=num_speakers,
                )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        if used_api:
            calls_made += 1

        der = compute_der(reference, annotation)
        total_ref_speech += der.total
        per_file.append(
            {
                "sample_id": sample.sample_id,
                "duration_s": sample.duration,
                "ref_speakers": sample.num_speakers,
                "hyp_speakers": len(set(annotation.labels())),
                "inference_s": round(elapsed, 2),
                "api_call": used_api,
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
        "backend": BACKEND,
        "sarvam_model": DEFAULT_MODEL,
        "sarvam_mode": DEFAULT_MODE,
        "language_code": language_code,
        "num_speakers_hint": num_speakers,
        "num_samples": len(samples),
        "api_calls_made": calls_made,
        "api_calls_skipped_cached": len(samples) - calls_made,
        "total_audio_duration_s": round(total_audio_s, 2),
        "total_evaluated_speech_s": round(total_ref_speech, 2),
        "der_pct": round(weighted_der, 3),
        "false_alarm_pct": round(weighted_fa, 3),
        "missed_detection_pct": round(weighted_miss, 3),
        "confusion_pct": round(weighted_conf, 3),
        "evaluation": {"collar_s": 0.0, "skip_overlap": False},
        "per_file": per_file,
    }

    out_path = output_dir / BENCHMARK_RESULT
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBenchmark summary ({len(samples)} files, {calls_made} batch job(s)):")
    print(f"  DER:  {summary['der_pct']:.2f}%")
    print(f"  FA:   {summary['false_alarm_pct']:.2f}%")
    print(f"  Miss: {summary['missed_detection_pct']:.2f}%")
    print(f"  Conf: {summary['confusion_pct']:.2f}%")
    print(f"Results saved to {out_path}")
    return summary


def run_key_check() -> None:
    get_api_key()
    verify_api_key()
    print("Sarvam API key OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sarvam Saaras v3 batch STT + diarization")
    parser.add_argument("--audio", type=Path, help="Path to a WAV file for smoke testing")
    parser.add_argument("--benchmark", action="store_true", help="Run DER benchmark on dataset/")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark to N samples")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help='BCP-47 language code (e.g. hi-IN, en-IN) or "unknown" for auto-detect',
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Optional speaker count hint (1-10); omit for auto-detect",
    )
    parser.add_argument(
        "--check-key",
        action="store_true",
        help="Verify SARVAM_API_KEY without running diarization",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm paid API usage (required for --benchmark)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many batch jobs would be submitted, without calling the API",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and submit a new batch job",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not read or write local cache",
    )
    args = parser.parse_args()
    use_cache = not args.no_cache
    language_code = None if args.language.lower() in {"auto", "none"} else args.language

    if args.check_key:
        run_key_check()
        return

    if args.benchmark:
        if not args.yes and not args.dry_run:
            parser.error(
                "Benchmark uses paid API credits. Re-run with --yes to confirm, "
                "or --dry-run to preview usage without calling the API."
            )
        run_benchmark(
            args.dataset_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            use_cache=use_cache,
            force=args.force,
            dry_run=args.dry_run,
            language_code=language_code,
            num_speakers=args.num_speakers,
        )
    elif args.audio:
        run_smoke_test(
            args.audio,
            use_cache=use_cache,
            force=args.force,
            language_code=language_code,
            num_speakers=args.num_speakers,
        )
    else:
        run_key_check()


if __name__ == "__main__":
    main()
