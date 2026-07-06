# Diarization model tests

Local scripts to compare speaker diarization models (and one ASR+diar pipeline) on short WAVs in `test-audio/` or on the CallHome parquet benchmark in `dataset/`.

Each model has its own script. Run one at a time, compare JSON under `results/` or next to the audio file.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Nemotron 3.5 pipeline** also needs Transformers from git (not on PyPI yet):

```powershell
pip install git+https://github.com/huggingface/transformers.git
```

### `.env`

This repo **includes `.env` in git on purpose** for private testing. It holds API keys used by cloud scripts:

| Variable | Used by |
|----------|---------|
| `PYANNOTEAI_API_KEY` | `diarize_precision_2.py` |
| `DEEPGRAM_API_KEY` | `diarize_deepgram_nova3.py` |
| `SARVAM_API_KEY` | `diarize_sarvam_saaras.py` |

Local pyannote / NVIDIA models use `huggingface-cli login` (accept gated model terms on the Hub).

`.env.example` is a template if you clone without the committed `.env`.

### CallHome benchmark dataset

Not in git (~2.2 GB). Download 5 parquet shards from [talkbank/callhome/eng](https://huggingface.co/datasets/talkbank/callhome/tree/main/eng) into `dataset/`.

## Models

| Model | Script | Backend | Key / access |
|-------|--------|---------|--------------|
| [pyannote 3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) | `diarize_3_1.py` | Local CPU/GPU | Hugging Face |
| [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | `diarize_community_1.py` | Local CPU/GPU | Hugging Face |
| [pyannote precision-2](https://huggingface.co/pyannote/speaker-diarization-precision-2) | `diarize_precision_2.py` | pyannoteAI API | `.env` |
| [Sortformer 4spk v1](https://huggingface.co/nvidia/diar_sortformer_4spk-v1) | `diarize_sortformer_v1.py` | NeMo CPU/GPU | Hugging Face |
| [Streaming Sortformer v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2) | `diarize_sortformer_v2.py` | NeMo CPU/GPU | Hugging Face |
| [Streaming Sortformer v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | `diarize_sortformer_v2_1.py` | NeMo CPU/GPU | Hugging Face |
| [MSDD telephonic](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/diar_msdd_telephonic) | `diarize_msdd_telephonic.py` | NeMo CPU/GPU | NGC / HF |
| [Deepgram Nova-3 + diarize v2](https://developers.deepgram.com/) | `diarize_deepgram_nova3.py` | Cloud API | `.env` |
| [Sarvam Saaras v3 + diarization](https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/how-to/enable-speaker-diarization) | `diarize_sarvam_saaras.py` | Cloud batch API | `.env` |
| Sortformer v2.1 + [Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | `transcribe_nemotron35_pipeline.py` | NeMo + Transformers | Hugging Face |

Tested on **Python 3.12**, **pyannote.audio 4.0.7**, **torch 2.12.1** (CPU).

## Quick test (`test-audio/`)

Diarization only — writes `{stem}.{model-slug}.json` and `.rttm` next to the WAV:

```powershell
python diarize_sortformer_v2_1.py --audio test-audio/test_1.wav
python diarize_3_1.py --audio test-audio/test_1.wav
python diarize_precision_2.py --audio test-audio/test_1.wav
python diarize_deepgram_nova3.py --audio test-audio/test_1.wav
python diarize_sarvam_saaras.py --audio test-audio/hindi-test.wav
python diarize_msdd_telephonic.py --audio test-audio/test_1.wav
```

Diarization + transcription (Sortformer segments, Nemotron 3.5 text per segment):

```powershell
python transcribe_nemotron35_pipeline.py --test-audio-dir test-audio --output-dir results/nemotron-3.5-pipeline --language auto
```

Use `--language hi-IN`, `en-US`, or `auto` for mixed Hindi/English.

## Benchmark (CallHome DER)

Same eval for all models: **0 s collar**, **overlap included**, speech-weighted aggregate DER.

```powershell
python diarize_sortformer_v2_1.py --benchmark --limit 5
python diarize_3_1.py --benchmark --limit 5
python diarize_precision_2.py --benchmark --limit 5 --yes
python diarize_deepgram_nova3.py --benchmark --limit 5 --yes
python diarize_sarvam_saaras.py --benchmark --limit 5 --yes
python diarize_msdd_telephonic.py --benchmark --limit 5
```

Omit `--limit` for all 140 samples (slow on CPU). Cloud scripts need `--yes`; precision-2, Deepgram, and Sarvam support `--dry-run`.

Compare `der_pct` in `results/benchmark_*.json`. Summary for the 5-file run: `results/benchmark_comparison.json`.

## Output formats

**Diarization JSON** (segments only):

```json
{
  "model": "nvidia/diar_streaming_sortformer_4spk-v2.1",
  "audio_file": "test_1.wav",
  "num_speakers": 3,
  "segments": [{ "speaker": "speaker_0", "start": 0.32, "end": 2.08 }]
}
```

**Nemotron pipeline JSON** (`results/nemotron-3.5-pipeline/`):

```json
{
  "segments": [{
    "speaker": "speaker_0",
    "start": 3.44,
    "end": 4.56,
    "start_ms": 3440,
    "end_ms": 4560,
    "text": "..."
  }],
  "full_transcript": "..."
}
```

UTF-8 text (Hindi renders as Devanagari, not `\u` escapes).

## Layout

```
diarization-models/
├── diarize_*.py                  # one script per diarization model
├── transcribe_nemotron35_pipeline.py
├── test-audio/                   # sample WAVs + per-run outputs
├── dataset/                      # CallHome parquet (download separately)
├── results/
│   ├── benchmark_*.json
│   ├── benchmark_comparison.json
│   ├── nemotron-3.5-pipeline/     # ASR+diar pipeline outputs
│   └── cache/                    # precision-2 API cache (gitignored)
└── utils/
    ├── audio.py, dataset.py, eval.py
    ├── sortformer.py, msdd.py, deepgram.py, sarvam.py
    ├── nemotron35_pipeline.py
    ├── cache.py, env.py
```
