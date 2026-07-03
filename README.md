# Speaker Diarization Benchmark

Benchmark speaker diarization models on a local parquet dataset or individual audio files. Each model has its own script — run them one at a time.

**Repository:** [github.com/arpitgarg88/dairization-model-test](https://github.com/arpitgarg88/dairization-model-test)

| Model | Script | Backend | API key |
|-------|--------|---------|---------|
| [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | `diarize_3_1.py` | Local (CPU/GPU) | Hugging Face |
| [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | `diarize_community_1.py` | Local (CPU/GPU) | Hugging Face |
| [speaker-diarization-precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2) | `diarize_precision_2.py` | pyannoteAI cloud | pyannoteAI |
| [diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | `diarize_sortformer_v1.py` | Local NeMo (CPU/GPU) | Hugging Face |
| [diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | `diarize_sortformer_v2.py` | Local NeMo (CPU/GPU) | Hugging Face |
| [diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | `diarize_sortformer_v2_1.py` | Local NeMo (CPU/GPU) | Hugging Face |

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
| `HF_TOKEN` | pyannote local + NVIDIA Sortformer | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
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
   - [nvidia/diar_sortformer_4spk-v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)
   - [nvidia/diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)
   - [nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1)

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
├── diarize_3_1.py                  # pyannote 3.1
├── diarize_community_1.py          # pyannote community-1
├── diarize_precision_2.py          # pyannote precision-2 (cloud API)
├── diarize_sortformer_v1.py        # NVIDIA Sortformer offline v1
├── diarize_sortformer_v2.py        # NVIDIA Streaming Sortformer v2
├── diarize_sortformer_v2_1.py      # NVIDIA Streaming Sortformer v2.1
├── requirements.txt
├── dataset/                        # Benchmark parquet shards (140 samples)
├── test-audio/                     # Sample WAV files for quick tests
├── results/                        # Per-model benchmark JSON outputs
│   ├── benchmark_speaker_diarization_3_1.json
│   ├── benchmark_speaker_diarization_community_1.json
│   ├── benchmark_speaker_diarization_precision_2.json
│   ├── benchmark_diar_sortformer_4spk_v1.json
│   ├── benchmark_diar_streaming_sortformer_4spk_v2.json
│   ├── benchmark_diar_streaming_sortformer_4spk_v2_1.json
│   └── cache/precision-2/          # Cached precision-2 API results
└── utils/
    ├── audio.py                    # Load WAV / bytes (no FFmpeg required)
    ├── dataset.py                  # Parquet dataset loader
    ├── eval.py                     # DER metrics & result export
    ├── sortformer.py               # NVIDIA Sortformer helpers
    └── cache.py                    # precision-2 response cache
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
python diarize_sortformer_v1.py --audio test-audio/test_1.wav
python diarize_sortformer_v2.py --audio test-audio/test_1.wav
python diarize_sortformer_v2_1.py --audio test-audio/test_1.wav

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
test-audio/test_1.diar-sortformer-4spk-v1.json
test-audio/test_1.diar-sortformer-4spk-v1.rttm
test-audio/test_1.diar-streaming-sortformer-4spk-v2.json
test-audio/test_1.diar-streaming-sortformer-4spk-v2.rttm
test-audio/test_1.diar-streaming-sortformer-4spk-v2.1.json
test-audio/test_1.diar-streaming-sortformer-4spk-v2.1.rttm
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

Each model is benchmarked with its own script. Run them separately and compare the JSON files in `results/`.

### Evaluation settings

All benchmarks use the same **"Full" DER** setup:

- **Collar:** 0 s (no forgiveness around boundaries)
- **Overlapping speech:** evaluated (`skip_overlap=false`)
- **Aggregate DER:** speech-duration weighted across files

### pyannote models

```powershell
python diarize_3_1.py --benchmark --limit 5
python diarize_community_1.py --benchmark --limit 5

$env:PYANNOTEAI_API_KEY = "your-api-key"
python diarize_precision_2.py --benchmark --limit 5 --yes
```

### NVIDIA Sortformer models

```powershell
python diarize_sortformer_v1.py --benchmark --limit 5
python diarize_sortformer_v2.py --benchmark --limit 5
python diarize_sortformer_v2_1.py --benchmark --limit 5
```

Use `--limit N` to evaluate only the **first N samples** from the dataset. Omit `--limit` for the full 140 samples (~hours on CPU).

### Full dataset (one model at a time)

```powershell
python diarize_3_1.py --benchmark
python diarize_community_1.py --benchmark
python diarize_sortformer_v1.py --benchmark
python diarize_sortformer_v2.py --benchmark
python diarize_sortformer_v2_1.py --benchmark

$env:PYANNOTEAI_API_KEY = "your-api-key"
python diarize_precision_2.py --benchmark --yes   # 140 API calls
```

### Comparing models

After running benchmarks with the **same `--limit`**, compare `der_pct` in each model's JSON file under `results/`. Use the same `num_samples` and `total_evaluated_speech_s` to confirm you evaluated the same subset.

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
| `benchmark_diar_sortformer_4spk_v1.json` | Sortformer v1 aggregate + per-file metrics |
| `benchmark_diar_streaming_sortformer_4spk_v2.json` | Sortformer v2 aggregate + per-file metrics |
| `benchmark_diar_streaming_sortformer_4spk_v2_1.json` | Sortformer v2.1 aggregate + per-file metrics |

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

Local pyannote models run on CPU by default. For GPU, those scripts call `pipeline.to(torch.device("cuda"))` when CUDA is available. Sortformer scripts accept `--device cuda`.

---

## Quick reference

| Task | Command |
|------|---------|
| Smoke test one file | `python diarize_3_1.py --audio test-audio/test_1.wav` |
| Benchmark 5 samples (one model) | `python diarize_3_1.py --benchmark --limit 5` |
| Benchmark Sortformer v2.1 | `python diarize_sortformer_v2_1.py --benchmark --limit 5` |
| Benchmark precision-2 only | `python diarize_precision_2.py --benchmark --limit 5 --yes` |
| Preview API usage | `python diarize_precision_2.py --benchmark --limit 5 --dry-run` |
