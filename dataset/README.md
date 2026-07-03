# Dataset

The benchmark dataset is **not included in git** (~2.2 GB of parquet shards).

## Expected layout

Place your parquet files here:

```
dataset/
  data-00000-of-00005.parquet
  data-00001-of-00005.parquet
  data-00002-of-00005.parquet
  data-00003-of-00005.parquet
  data-00004-of-00005.parquet
```

## Schema (per row)

| Column | Description |
|--------|-------------|
| `audio.bytes` | 16 kHz mono WAV embedded as bytes |
| `timestamps_start` | Segment start times (seconds) |
| `timestamps_end` | Segment end times (seconds) |
| `speakers` | Ground-truth speaker labels (`A`, `B`, …) |

**140 samples** total across 5 shards.

## Verify

```powershell
python -c "from utils.dataset import load_dataset; print(len(load_dataset()), 'samples')"
```
