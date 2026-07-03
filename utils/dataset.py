"""Load the local parquet diarization benchmark dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "dataset"


@dataclass(frozen=True)
class DiarizationSample:
    sample_id: str
    audio_bytes: bytes
    timestamps_start: list[float]
    timestamps_end: list[float]
    speakers: list[str]

    @property
    def duration(self) -> float:
        return max(self.timestamps_end) if self.timestamps_end else 0.0

    @property
    def num_speakers(self) -> int:
        return len(set(self.speakers))


def load_dataset(dataset_dir: Path | str = DEFAULT_DATASET_DIR) -> list[DiarizationSample]:
    """Load all samples from the parquet shards in ``dataset_dir``."""
    dataset_dir = Path(dataset_dir)
    parquet_files = sorted(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {dataset_dir}")

    samples: list[DiarizationSample] = []
    offset = 0
    for parquet_path in parquet_files:
        table = pq.read_table(
            parquet_path,
            columns=["audio", "timestamps_start", "timestamps_end", "speakers"],
        )
        for row_idx in range(table.num_rows):
            audio = table["audio"][row_idx].as_py()
            samples.append(
                DiarizationSample(
                    sample_id=f"sample_{offset + row_idx:04d}",
                    audio_bytes=audio["bytes"],
                    timestamps_start=table["timestamps_start"][row_idx].as_py(),
                    timestamps_end=table["timestamps_end"][row_idx].as_py(),
                    speakers=table["speakers"][row_idx].as_py(),
                )
            )
        offset += table.num_rows

    return samples
