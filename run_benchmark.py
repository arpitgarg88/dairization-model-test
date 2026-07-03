"""
Run pyannote diarization benchmarks and produce a comparison summary.

Usage:
  # Local models only (3.1 + community-1)
  python run_benchmark.py --limit 5

  # Include existing precision-2 results from results/ if available
  python run_benchmark.py --limit 5 --merge-precision-2

  # Run all three models (precision-2 uses paid API — requires --yes)
  python run_benchmark.py --limit 5 --include-precision-2 --yes

  # Preview precision-2 API usage without calling the API
  python run_benchmark.py --limit 5 --include-precision-2 --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diarize_3_1 import run_benchmark as benchmark_3_1
from diarize_community_1 import run_benchmark as benchmark_community_1
from diarize_precision_2 import run_benchmark as benchmark_precision_2
from utils.dataset import DEFAULT_DATASET_DIR, load_dataset
from utils.env import load_dotenv

RESULTS_DIR = Path(__file__).resolve().parent / "results"
load_dotenv()
PRECISION_2_RESULTS = "benchmark_speaker_diarization_precision_2.json"


def _metrics_from_summary(summary: dict) -> dict:
    return {
        "der_pct": summary["der_pct"],
        "false_alarm_pct": summary["false_alarm_pct"],
        "missed_detection_pct": summary["missed_detection_pct"],
        "confusion_pct": summary["confusion_pct"],
        "backend": summary.get("backend", "local"),
        "avg_inference_s": round(
            sum(f["inference_s"] for f in summary["per_file"]) / len(summary["per_file"]),
            2,
        ),
    }


def _load_precision_2_results(output_dir: Path, num_samples: int) -> dict | None:
    path = output_dir / PRECISION_2_RESULTS
    if not path.exists():
        return None

    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("num_samples") != num_samples:
        print(
            f"Skipping saved precision-2 results ({summary.get('num_samples')} samples) "
            f"— expected {num_samples}. Re-run with --include-precision-2 --yes."
        )
        return None

    print(f"Loaded precision-2 results from {path}")
    return summary


def _winner_by_der(models: dict[str, dict]) -> str:
    return min(models, key=lambda name: models[name]["der_pct"])


def _print_comparison_table(models: dict[str, dict]) -> None:
    print("\n=== Comparison ===")
    print(f"{'Model':<35} {'DER':>7} {'FA':>7} {'Miss':>7} {'Conf':>7} {'Infer':>8}")
    print("-" * 75)
    for name, m in sorted(models.items(), key=lambda x: x[1]["der_pct"]):
        infer = f"{m['avg_inference_s']:.1f}s" if m.get("avg_inference_s") else "—"
        print(
            f"{name:<35} {m['der_pct']:>6.2f}% {m['false_alarm_pct']:>6.2f}% "
            f"{m['missed_detection_pct']:>6.2f}% {m['confusion_pct']:>6.2f}% {infer:>8}"
        )
    print(f"\nBest model (lowest DER): {_winner_by_der(models)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pyannote diarization models")
    parser.add_argument("--limit", type=int, default=None, help="Limit benchmark to N samples")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument(
        "--include-precision-2",
        action="store_true",
        help="Run precision-2 cloud benchmark (paid API)",
    )
    parser.add_argument(
        "--merge-precision-2",
        action="store_true",
        help="Include saved precision-2 results in comparison if available",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm paid API usage for precision-2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only preview precision-2 API usage (skips all inference)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore precision-2 cache and call the API again",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable precision-2 local cache",
    )
    args = parser.parse_args()

    if args.dry_run:
        samples = load_dataset(args.dataset_dir)
        if args.limit is not None:
            samples = samples[: args.limit]
        benchmark_precision_2(
            args.dataset_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            force=args.force,
            dry_run=True,
        )
        return

    if args.include_precision_2 and not args.yes:
        parser.error(
            "precision-2 uses paid API credits. Re-run with --yes to confirm, "
            "or use --merge-precision-2 to include previously saved results."
        )

    r31 = benchmark_3_1(
        dataset_dir=args.dataset_dir,
        limit=args.limit,
        output_dir=args.output_dir,
    )
    rc1 = benchmark_community_1(
        dataset_dir=args.dataset_dir,
        limit=args.limit,
        output_dir=args.output_dir,
    )

    models: dict[str, dict] = {
        "speaker-diarization-3.1": _metrics_from_summary(r31),
        "speaker-diarization-community-1": _metrics_from_summary(rc1),
    }

    rp2: dict | None = None
    if args.include_precision_2:
        rp2 = benchmark_precision_2(
            dataset_dir=args.dataset_dir,
            limit=args.limit,
            output_dir=args.output_dir,
            use_cache=not args.no_cache,
            force=args.force,
            dry_run=False,
        )
    elif args.merge_precision_2:
        rp2 = _load_precision_2_results(args.output_dir, r31["num_samples"])

    if rp2 is not None:
        models["speaker-diarization-precision-2"] = _metrics_from_summary(rp2)
        if "api_calls_made" in rp2:
            models["speaker-diarization-precision-2"]["api_calls_made"] = rp2["api_calls_made"]

    comparison = {
        "evaluation": r31["evaluation"],
        "num_samples": r31["num_samples"],
        "total_audio_duration_s": r31["total_audio_duration_s"],
        "models": models,
        "winner_by_der": _winner_by_der(models),
        "per_file_der": {
            sample["sample_id"]: {
                "speaker-diarization-3.1": next(
                    f["der_pct"] for f in r31["per_file"] if f["sample_id"] == sample["sample_id"]
                ),
                "speaker-diarization-community-1": next(
                    f["der_pct"] for f in rc1["per_file"] if f["sample_id"] == sample["sample_id"]
                ),
                **(
                    {
                        "speaker-diarization-precision-2": next(
                            f["der_pct"]
                            for f in rp2["per_file"]
                            if f["sample_id"] == sample["sample_id"]
                        )
                    }
                    if rp2 is not None
                    else {}
                ),
            }
            for sample in r31["per_file"]
        },
    }

    out_path = args.output_dir / "benchmark_comparison.json"
    out_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    _print_comparison_table(models)
    print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    main()
