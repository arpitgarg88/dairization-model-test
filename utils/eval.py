"""Diarization error rate (DER) evaluation utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from utils.dataset import DiarizationSample


def build_reference_annotation(sample: DiarizationSample) -> Annotation:
    """Convert segment-level ground truth into a pyannote Annotation."""
    reference = Annotation(uri=sample.sample_id)
    for start, end, speaker in zip(
        sample.timestamps_start, sample.timestamps_end, sample.speakers
    ):
        reference[Segment(start, end)] = speaker
    return reference


def extract_annotation(output) -> Annotation:
    """Handle pyannote.audio 4.x DiarizeOutput vs legacy Annotation return types."""
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output


def save_diarization_results(
    annotation: Annotation,
    audio_path: Path | str,
    model_slug: str,
    *,
    model_id: str,
    inference_s: float | None = None,
) -> tuple[Path, Path]:
    """Save JSON + RTTM diarization outputs next to the input audio file."""
    audio_path = Path(audio_path)
    stem = audio_path.stem
    out_dir = audio_path.parent

    segments = [
        {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    payload = {
        "model": model_id,
        "audio_file": audio_path.name,
        "inference_s": round(inference_s, 2) if inference_s is not None else None,
        "num_speakers": len(set(annotation.labels())),
        "num_segments": len(segments),
        "segments": segments,
    }

    json_path = out_dir / f"{stem}.{model_slug}.json"
    rttm_path = out_dir / f"{stem}.{model_slug}.rttm"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(rttm_path, "w", encoding="utf-8") as f:
        annotation.write_rttm(f)

    return json_path, rttm_path


@dataclass
class DerComponents:
    der: float
    false_alarm: float
    missed_detection: float
    confusion: float
    total: float


def compute_der(
    reference: Annotation,
    hypothesis: Annotation,
    *,
    collar: float = 0.0,
    skip_overlap: bool = False,
) -> DerComponents:
    """
    Compute DER using pyannote's official metric.

    Defaults match pyannote's "Full" benchmark setup:
    - no forgiveness collar
    - overlapping speech is evaluated
    """
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    components = metric(reference, hypothesis, detailed=True)
    total = components["total"]
    if total == 0:
        return DerComponents(der=0.0, false_alarm=0.0, missed_detection=0.0, confusion=0.0, total=0.0)

    return DerComponents(
        der=100.0 * components["diarization error rate"],
        false_alarm=100.0 * components["false alarm"] / total,
        missed_detection=100.0 * components["missed detection"] / total,
        confusion=100.0 * components["confusion"] / total,
        total=total,
    )
