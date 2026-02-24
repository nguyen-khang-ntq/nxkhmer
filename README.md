# Interspeech TTS Project

This project contains pipelines and models for Text-to-Speech (TTS) research and development.

## Folder Structure

### 📊 `audio_pipeline/`
Audio data processing pipeline for cleaning, standardization, and preparation.

**Key features:**
- Audio standardization (sampling rate, channels, format)
- Speaker diarization
- Quality filtering
- Sidon cleaner for text normalization
- ASR transcription integration

**Main entry point:** `main.py`

### 🔬 `eval_pipeline/`
Comprehensive TTS evaluation metrics and testing framework.

**Supported metrics:**
- **CER** (Character Error Rate) - transcription accuracy
- **MCD** (Mel-Cepstral Distortion) - acoustic quality
- **MOS** (Mean Opinion Score) - perceptual quality via DNSMOS
- **Speaker Similarity** - voice cloning similarity
- **F0/Pitch** - prosody analysis

**Usage:** `run_all_tests.py` for automated evaluation

**Documentation:** See `TESTING.md` and `INSTALL.md`

### 🎙️ `model/`
TTS model implementations and training scripts.

**Supported models:**
- **F5-TTS** - Flow-based TTS with duration control
- **Orpheus** - LLM-based TTS using SNAC tokenization
- **NeuTTS** - Neural codec-based TTS
- **VITS** - Variational Inference TTS
- **XTTSv2** - Multilingual voice cloning (Coqui TTS)

Each model folder contains:
- Training scripts (`train.py`)
- Configuration files (`.yaml`)
- Data preparation utilities
- Tokenizers/vocoders

## Quick Start

### Audio Pipeline
```bash
cd audio_pipeline
python main.py --config config.json
```

### Evaluation
```bash
cd eval_pipeline
python run_all_tests.py
```

### Model Training
```bash
cd model/<model_name>
python train.py --config <config_file>
```

## Requirements

See individual `requirements.txt` in each subfolder for specific dependencies.

## Notes

- All models support Khmer language
- Audio files use 24kHz sampling rate by default (model-dependent)
- Generated outputs and checkpoints are ignored by `.gitignore`
