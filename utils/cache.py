"""Local cache helpers for cloud API diarization (avoids duplicate API calls)."""

from __future__ import annotations

import json
from pathlib import Path

from pyannote.core import Annotation, Segment

CACHE_DIR = Path(__file__).resolve().parents[1] / "results" / "cache"


def cache_path(model_slug: str, cache_key: str) -> Path:
    return CACHE_DIR / model_slug / f"{cache_key}.json"


def load_cached_annotation(model_slug: str, cache_key: str) -> Annotation | None:
    path = cache_path(model_slug, cache_key)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    annotation = Annotation(uri=cache_key)
    for idx, segment in enumerate(data["segments"]):
        annotation[Segment(segment["start"], segment["end"]), idx] = segment["speaker"]
    return annotation.rename_tracks("string")


def save_cached_annotation(
    annotation: Annotation,
    model_slug: str,
    cache_key: str,
    *,
    model_id: str,
    inference_s: float | None = None,
) -> Path:
    path = cache_path(model_slug, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)

    segments = [
        {"speaker": speaker, "start": round(turn.start, 3), "end": round(turn.end, 3)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    payload = {
        "model": model_id,
        "cache_key": cache_key,
        "inference_s": round(inference_s, 2) if inference_s is not None else None,
        "num_speakers": len(set(annotation.labels())),
        "num_segments": len(segments),
        "segments": segments,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
