# Speaker Diarization Benchmark

Benchmark and compare three pyannote speaker diarization pipelines on a local parquet dataset or individual audio files.

**Repository:** [github.com/arpitgarg88/dairization-model-test](https://github.com/arpitgarg88/dairization-model-test)

| Model | Script | Backend | API key |
|-------|--------|---------|---------|
| [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | `diarize_3_1.py` | Local (CPU/GPU) | Hugging Face |
| [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | `diarize_community_1.py` | Local (CPU/GPU) | Hugging Face |
| [speaker-diarization-precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2) | `diarize_precision_2.py` | pyannoteAI cloud | pyannoteAI |

---

## Quick start (clone)

```powershell
git clone https://github.com/arpitgarg88/dairization-model-test.git
cd dairization-model-test

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Configure secrets
copy .env.example .env
# Edit .env with your HF_TOKEN and PYANNOTEAI_API_KEY

# Add dataset — download from Hugging Face (see dataset/README.md)
# https://huggingface.co/datasets/talkbank/callhome/tree/main/eng
```

---

## Prerequisites

### 1. Python environment

```powershell
cd dairization-model-test
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tested with **Python 3.12**, **pyannote.audio 4.0.7**, **torch 2.12.1** (CPU).

### 2. Environment variables (`.env`)

```powershell
copy .env.example .env
```

Edit `.env`:

| Variable | Required for | Where to get it |
|----------|--------------|-----------------|
| `HF_TOKEN` | 3.1, community-1 | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `PYANNOTEAI_API_KEY` | precision-2 only | [dashboard.pyannote.ai](https://dashboard.pyannote.ai/) |

Scripts load `.env` automatically via `python-dotenv`. You can also set variables in the shell or use `huggingface-cli login`.

### 3. Dataset

The **parquet dataset (~2.2 GB) is not in git**. Download it from Hugging Face:

**[talkbank/callhome — English (`eng`)](https://huggingface.co/datasets/talkbank/callhome/tree/main/eng)**

- Accept the gated dataset terms on the Hub (login required).
- Download the 5 `data-*-of-00005.parquet` files into `dataset/`.

Full download steps: [dataset/README.md](dataset/README.md).

### 4. Hugging Face model access (local models)

1. Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Log in:

   ```powershell
   huggingface-cli login
   ```

3. Accept the model terms on Hugging Face:
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (required for 3.1)
   - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

### 5. pyannoteAI API key (precision-2 only)

Set `PYANNOTEAI_API_KEY` in `.env` or your shell. Do **not** commit `.env` to git.

```powershell
# Already loaded from .env if you copied .env.example → .env
python diarize_precision_2.py --benchmark --limit 5 --yes
```

---

## Project layout

```
dairization-model-test/
├── diarize_3_1.py              # Legacy 3.1 pipeline
├── diarize_community_1.py        # Community-1 pipeline
├── diarize_precision_2.py        # Precision-2 cloud API
├── run_benchmark.py              # Run & compare all models
├── requirements.txt
├── dataset/                      # Benchmark parquet shards (140 samples)
├── test-audio/                   # Sample WAV files for quick tests
├── results/                      # Benchmark JSON outputs
│   ├── benchmark_speaker_diarization_3_1.json
│   ├── benchmark_speaker_diarization_community_1.json
│   ├── benchmark_speaker_diarization_precision_2.json
│   ├── benchmark_comparison.json
│   └── cache/precision-2/        # Cached precision-2 API results
└── utils/
    ├── audio.py                  # Load WAV / bytes (no FFmpeg required)
    ├── dataset.py                # Parquet dataset loader
    ├── eval.py                   # DER metrics & result export
    └── cache.py                  # precision-2 response cache
```

---

## Dataset

The `dataset/` folder contains **5 parquet shards** with **140 samples** total.

| Field | Description |
|-------|-------------|
| `audio.bytes` | 16 kHz mono WAV embedded as bytes |
| `timestamps_start` | Segment start times (seconds) |
| `timestamps_end` | Segment end times (seconds) |
| `speakers` | Ground-truth speaker labels (`A`, `B`, …) |

Each file has **2–4 speakers**, ~300–600 s duration, with overlapping speech in the annotations.

---

## Run diarization on a single audio file

Outputs are saved **next to the input file** as `.json` and `.rttm`.

```powershell
.\.venv\Scripts\Activate.ps1

python diarize_3_1.py --audio test-audio/test_1.wav
python diarize_community_1.py --audio test-audio/test_1.wav

$env:PYANNOTEAI_API_KEY = "your-api-key"
python diarize_precision_2.py --audio test-audio/test_1.wav
```

**Output files** (example for `test_1.wav`):

```
test-audio/test_1.speaker-diarization-3.1.json
test-audio/test_1.speaker-diarization-3.1.rttm
test-audio/test_1.speaker-diarization-community-1.json
test-audio/test_1.speaker-diarization-community-1.rttm
test-audio/test_1.speaker-diarization-precision-2.json
test-audio/test_1.speaker-diarization-precision-2.rttm
```

**JSON format:**

```json
{
  "model": "pyannote/speaker-diarization-community-1",
  "audio_file": "test_1.wav",
  "inference_s": 26.5,
  "num_speakers": 2,
  "num_segments": 7,
  "segments": [
    { "speaker": "SPEAKER_00", "start": 0.28, "end": 2.16 }
  ]
}
```

---

## Run benchmarks

### Evaluation settings

All benchmarks use pyannote's **"Full" DER** setup:

- **Collar:** 0 s (no forgiveness around boundaries)
- **Overlapping speech:** evaluated (`skip_overlap=false`)
- **Aggregate DER:** speech-duration weighted across files

### Option A — Compare local models only (~25 min for 5 samples on CPU)

```powershell
python run_benchmark.py --limit 5
```

Runs **3.1** and **community-1**, writes `results/benchmark_comparison.json`.

### Option B — Include saved precision-2 results (no extra API cost)

If you already ran precision-2 separately:

```powershell
python run_benchmark.py --limit 5 --merge-precision-2
```

Re-runs local models and merges existing `results/benchmark_speaker_diarization_precision_2.json` into the comparison.

### Option C — Run all three models live

```powershell
$env:PYANNOTEAI_API_KEY = "your-api-key"
python run_benchmark.py --limit 5 --include-precision-2 --yes
```

### Option D — Preview precision-2 API usage (0 API calls)

```powershell
python run_benchmark.py --limit 5 --include-precision-2 --dry-run
```

### Run one model at a time

```powershell
python diarize_3_1.py --benchmark --limit 5
python diarize_community_1.py --benchmark --limit 5

$env:PYANNOTEAI_API_KEY = "your-api-key"
python diarize_precision_2.py --benchmark --limit 5 --yes
```

Use `--limit N` to evaluate only the **first N samples** from the dataset. Omit `--limit` for the full 140 samples (~hours on CPU).

### Full dataset

```powershell
python run_benchmark.py                          # local models, 140 files
python run_benchmark.py --merge-precision-2      # + saved precision-2

$env:PYANNOTEAI_API_KEY = "your-api-key"
python diarize_precision_2.py --benchmark --yes  # precision-2 only (140 API calls)
```

---

## precision-2 API safeguards

| Flag | Effect |
|------|--------|
| `--yes` | Required to confirm paid API usage on benchmark |
| `--dry-run` | Show how many API calls would be made; no inference |
| `--force` | Ignore cache and call the API again |
| `--no-cache` | Disable local cache read/write |

Cached responses are stored in `results/cache/precision-2/`. Re-running without `--force` skips API calls for already-processed samples.

**Preview before spending credits:**

```powershell
python diarize_precision_2.py --benchmark --limit 5 --dry-run
```

---

## Read benchmark results

### Per-model JSON files

| File | Contents |
|------|----------|
| `benchmark_speaker_diarization_3_1.json` | 3.1 aggregate + per-file metrics |
| `benchmark_speaker_diarization_community_1.json` | community-1 aggregate + per-file metrics |
| `benchmark_speaker_diarization_precision_2.json` | precision-2 aggregate + per-file metrics |

**Aggregate fields:**

| Field | Meaning |
|-------|---------|
| `der_pct` | Diarization Error Rate (lower is better) |
| `false_alarm_pct` | Speech detected where none in reference |
| `missed_detection_pct` | Reference speech missed |
| `confusion_pct` | Correct speech, wrong speaker |
| `num_samples` | Number of files evaluated |
| `total_audio_duration_s` | Sum of file durations |
| `total_evaluated_speech_s` | Speech time used for DER |

**Per-file fields** (`per_file` array):

| Field | Meaning |
|-------|---------|
| `sample_id` | e.g. `sample_0000` |
| `duration_s` | Audio length |
| `ref_speakers` | Ground-truth speaker count |
| `hyp_speakers` | Predicted speaker count |
| `inference_s` | Processing time |
| `der_pct` | File-level DER |
| `api_call` | precision-2 only: whether API was called |

### Comparison file

`results/benchmark_comparison.json` contains all models side by side:

```json
{
  "evaluation": { "collar_s": 0.0, "skip_overlap": false },
  "num_samples": 5,
  "total_audio_duration_s": 2112.57,
  "models": {
    "speaker-diarization-3.1": { "der_pct": 20.686, ... },
    "speaker-diarization-community-1": { "der_pct": 20.57, ... },
    "speaker-diarization-precision-2": { "der_pct": 11.651, ... }
  },
  "winner_by_der": "speaker-diarization-precision-2",
  "per_file_der": {
    "sample_0000": {
      "speaker-diarization-3.1": 13.899,
      "speaker-diarization-community-1": 13.474,
      "speaker-diarization-precision-2": 10.498
    }
  }
}
```

### Terminal comparison table

After `run_benchmark.py`, a summary table is printed:

```
=== Comparison ===
Model                                   DER      FA    Miss    Conf    Infer
---------------------------------------------------------------------------
speaker-diarization-precision-2       11.65%   5.00%   6.08%   0.57%    21.9s
speaker-diarization-community-1       20.57%   4.43%  10.69%   5.45%   148.5s
speaker-diarization-3.1               20.69%   4.43%  10.69%   5.56%   147.2s

Best model (lowest DER): speaker-diarization-precision-2
```

### How to interpret DER

```
DER = False Alarm + Missed Detection + Confusion
```

- **DER < 15%** — strong performance
- **DER 15–25%** — typical for challenging conversational audio
- **Confusion** — speaker swap errors (precision-2 usually lowest)
- **Miss** — speech segments not detected
- **False alarm** — non-speech classified as speech

Compare models on the **same `num_samples`** and `limit` for a fair comparison.

---

## Troubleshooting

### `torchcodec is not installed correctly` warning

Expected on Windows without FFmpeg. Local scripts load audio via **soundfile** in memory, so inference still works. The warning can be ignored.

`diarize_precision_2.py` calls `pipeline.apply()` directly to avoid a post-inference torchcodec check.

### `Hugging Face token not found`

```powershell
huggingface-cli login
```

### `pyannoteAI API key not found`

```powershell
$env:PYANNOTEAI_API_KEY = "your-api-key"
```

### `Benchmark uses paid API credits`

Add `--yes` to confirm, or use `--dry-run` to preview:

```powershell
python diarize_precision_2.py --benchmark --limit 5 --yes
```

### Slow local inference

Local models run on CPU by default. For GPU, both local scripts call `pipeline.to(torch.device("cuda"))` when CUDA is available.

---

## Quick reference

| Task | Command |
|------|---------|
| Smoke test one file | `python diarize_3_1.py --audio test-audio/test_1.wav` |
| Benchmark 5 samples (local) | `python run_benchmark.py --limit 5` |
| 3-model comparison (no new API calls) | `python run_benchmark.py --limit 5 --merge-precision-2` |
| Benchmark precision-2 only | `python diarize_precision_2.py --benchmark --limit 5 --yes` |
| Preview API usage | `python diarize_precision_2.py --benchmark --limit 5 --dry-run` |
