# Audio Processing Pipeline for TTS Data Preparation

A comprehensive multi-stage audio processing pipeline designed to prepare high-quality datasets for Text-to-Speech (TTS) training. The pipeline processes raw audio files through standardization, speaker diarization, quality filtering, and automatic transcription.

## Overview

This pipeline is inspired by the Amphion project structure and implements a sequential processing workflow optimized for TTS data preparation.

### Pipeline Stages

```
Raw Audio → Standardization → Diarization → Quality Filtering → Transcription → Sidon Cleaning → TTS Dataset
```

1. **Standardization**: Unifies audio format (24kHz, mono, normalized volume)
2. **Diarization**: Extracts single-speaker segments (2-40s) using pyannote
3. **Quality Filtering**: Removes low-quality audio using DNSMOS P.835
4. **Transcription**: Generates text transcripts using Omni ASR
5. **Sidon Cleaning**: Restores and enhances audio quality using Sidon speech restoration model (outputs 48kHz)

## Directory Structure

```
audio_pipeline/
├── config.json              # Pipeline configuration
├── main.py                  # Main pipeline orchestrator
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── processors/             # Pipeline stage processors
│   ├── __init__.py
│   ├── standardization.py  # Audio format standardization
│   ├── diarization.py      # Speaker diarization
│   ├── quality_filter.py   # Quality filtering
│   ├── transcription.py    # ASR transcription
│   └── sidon_cleaner.py    # Sidon audio cleaning (final step)
│
└── utils/                  # Utility modules
    ├── __init__.py
    ├── audio_utils.py      # Audio processing utilities
    └── omni_asr.py         # Omni ASR wrapper
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- HuggingFace account (for pyannote models)

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Authenticate with HuggingFace (for pyannote):
```bash
huggingface-cli login
```

3. Accept pyannote model agreements:
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Accept the user conditions

4. Ensure eval_pipeline is available (for DNSMOS and ASR):
   - The pipeline expects `../eval_pipeline/metrics/dnsmos.py`
   - The pipeline expects `../eval_pipeline/utils` for ASR factory

## Configuration

Edit `config.json` to customize pipeline behavior:

```json
{
  "standardization": {
    "target_sample_rate": 24000,
    "target_channels": 1,
    "target_rms_level": -20.0
  },
  "diarization": {
    "model": "pyannote/speaker-diarization-3.1",
    "min_segment_duration": 2.0,
    "max_segment_duration": 40.0
  },
  "quality_filtering": {
    "dnsmos_threshold": 3.0,
    "filter_criteria": {
      "min_ovrl": 3.0,
      "min_sig": 3.0,
      "min_bak": 3.5
    }
  },
  "asr": {
    "model_card": "omniASR_LLM_7B",
    "target_sample_rate": 16000
  },
  "sidon": {
    "enabled": true,
    "model_name": "facebook/w2v-bert-2.0",
    "adapter_path": "/home/coder/datasets/hf_models/sidon_raw_weight",
    "target_sample_rate": 48000,
    "use_fp16": true
  }
}
```

## Usage

### Full Pipeline

Process raw audio through all stages:

```bash
python main.py \
  --input /path/to/raw/audio \
  --output /path/to/output \
  --device cuda
```

### Skip Certain Steps

If you've already processed some stages:

```bash
python main.py \
  --input /path/to/audio \
  --output /path/to/output \
  --skip standardization diarization sidon
```

> **Note**: To disable Sidon cleaning entirely, set `"enabled": false` in the sidon config section, or use `--skip sidon` flag.

### Individual Processors

You can also run each stage independently:

#### Standardization
```bash
python processors/standardization.py \
  --input /path/to/raw/audio \
  --output /path/to/standardized \
  --config config.json
```

#### Diarization
```bash
python processors/diarization.py \
  --input /path/to/standardized \
  --output /path/to/diarized \
  --config config.json \
  --device cuda
```

#### Quality Filtering
```bash
python processors/quality_filter.py \
  --input /path/to/diarized \
  --output /path/to/filtered \
  --config config.json
```

#### Transcription
```bash
python processors/transcription.py \
  --input /path/to/filtered \
  --output /path/to/transcribed \
  --config config.json \
  --device cuda
```

#### Sidon Audio Cleaning
```bash
python processors/sidon_cleaner.py \
  --input /path/to/transcribed \
  --output /path/to/cleaned \
  --config config.json \
  --device cuda
```

## Output Structure

Each pipeline run creates a timestamped folder:

```
output/
└── run_20260130_143022/
    ├── pipeline_metadata.json       # Pipeline run metadata
    │
    ├── 01_standardized/
    │   ├── *.wav                    # Standardized audio files
    │   └── standardization_metadata.json
    │
    ├── 02_diarized/
    │   ├── *_spk*_seg*.wav         # Speaker segments
    │   └── diarization_metadata.json
    │
    ├── 03_filtered/
    │   ├── *.wav                    # Quality-filtered segments
    │   └── quality_filter_metadata.json
    │
    ├── 04_transcribed/
    │   ├── dataset.csv              # TTS dataset (before cleaning)
    │   └── transcription_metadata.json
    │
    └── 05_cleaned/
        ├── *.wav                    # Sidon-cleaned audio (48kHz, final)
        └── dataset.csv              # Final TTS dataset (cleaned audio paths)
```

### Final Dataset Format

The `dataset.csv` file is ready for TTS training:

```csv
audio_file,text
/path/to/audio1.wav,"transcribed text here"
/path/to/audio2.wav,"another transcription"
```

## Key Features

### Standardization
- Resamples to 24kHz (optimal for TTS)
- Converts to mono channel
- RMS-based volume normalization (-20dB target)
- Prevents clipping
- Supports multiple input formats (WAV, MP3, FLAC, M4A, OGG)

### Diarization
- Uses pyannote/speaker-diarization-3.1
- Extracts segments 2-40 seconds long
- Speaker-homogeneous segments
- Temporal annotations preserved

### Quality Filtering
- DNSMOS P.835 perceptual quality scores
- Filters noise, music, clipping, distortion
- Configurable thresholds for:
  - Overall quality (OVRL)
  - Signal quality (SIG)
  - Background quality (BAK)

### Transcription
- Uses Omni ASR (omniASR_LLM_7B)
- Automatic resampling to 16kHz for ASR
- Batch processing
- Error handling and logging

### Sidon Audio Cleaning (Final Step)
- **Speech restoration and enhancement** using Sidon model
- Removes noise, reverb, and artifacts
- Outputs high-quality 48kHz audio
- Uses w2v-bert-2.0 encoder with PEFT adapter
- DAC decoder for high-fidelity reconstruction
- FP16 support for faster inference
- Can be disabled via config or `--skip sidon` flag

**Note**: Sidon cleaning is the final processing step. The cleaned audio (48kHz) will be in `05_cleaned/` folder, and transcription CSV will reference these cleaned files.

## Performance Considerations

- **GPU recommended** for diarization and ASR
- **Processing time**: ~5-10x real-time for full pipeline
- **Storage**: Intermediate files require 3-4x input size
- **Memory**: 8GB+ RAM recommended, 16GB+ for large batches

## Troubleshooting

### HuggingFace Authentication Error
```bash
huggingface-cli login
# Then accept model agreements on HuggingFace website
```

### DNSMOS Import Error
Ensure eval_pipeline is properly set up:
```bash
ls ../eval_pipeline/metrics/dnsmos.py
```

### ASR Loading Error
Check that ASR factory is available:
```bash
ls ../eval_pipeline/utils.py
```

### CUDA Out of Memory
- Reduce batch size in config
- Process fewer files at once
- Use `--device cpu` (slower)

## Citation

If you use this pipeline in your research, please cite the relevant papers:

- **Pyannote.audio**: Bredin et al., "pyannote.audio: neural building blocks for speaker diarization"
- **DNSMOS**: Reddy et al., "DNSMOS P.835: A non-intrusive perceptual objective speech quality metric"

## License

This project follows the licenses of its dependencies (pyannote.audio, DNSMOS, etc.)

## Contact

For issues and questions, please open an issue on the repository.
