# Dataset

The benchmark dataset is **not included in git** (~2.2 GB of parquet shards).

## Download source

These parquet files come from the **CALLHOME English** subset on Hugging Face:

**[talkbank/callhome — `eng`](https://huggingface.co/datasets/talkbank/callhome/tree/main/eng)**

1. Log in to Hugging Face and **accept the dataset terms** (gated dataset, CC-BY-NC-SA-4.0).
2. Download the 5 parquet shards into this folder:

```powershell
# Option A: Hugging Face CLI (recommended)
pip install huggingface_hub
huggingface-cli login
huggingface-cli download talkbank/callhome --repo-type dataset --include "eng/data-*.parquet" --local-dir dataset --local-dir-use-symlinks False
# Then move files from dataset/eng/ to dataset/ if needed

# Option B: hf CLI
hf download talkbank/callhome --repo-type dataset --include "eng/data-*.parquet" --local-dir dataset
```

Expected files from the Hub (same names as below):

| File | Size (approx.) |
|------|----------------|
| `data-00000-of-00005.parquet` | 446 MB |
| `data-00001-of-00005.parquet` | 488 MB |
| `data-00002-of-00005.parquet` | 473 MB |
| `data-00003-of-00005.parquet` | 438 MB |
| `data-00004-of-00005.parquet` | 453 MB |

## Expected layout

Place the parquet files here:

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
