# F5-TTS Khmer Fine-Tuning

This directory contains the configuration and tools for fine-tuning F5-TTS (Flow Matching for Text-to-Speech) on Khmer speech datasets.

## Overview

F5-TTS is a state-of-the-art non-autoregressive TTS model based on diffusion and flow matching. This setup enables fine-tuning the model for Khmer language using custom datasets.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- ffmpeg and ffprobe installed
- PyTorch 2.0+

## Installation

### 1. Clone F5-TTS Repository

```bash
git clone https://github.com/SWivid/F5-TTS.git
cd F5-TTS
```

### 2. Install Dependencies

```bash
pip install -e .
```

Or install from PyPI:

```bash
pip install f5-tts
```

### 3. Install Additional Requirements

```bash
pip install hydra-core omegaconf tqdm wandb
```

Ensure ffmpeg is installed:

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

## Data Preparation

### Dataset Format

Your dataset should be in CSV format with one of the following structures:

**Format 1 (Recommended):** Comma-separated with speaker column
```csv
audio_file,text,speaker
wavs/audio_001.wav,ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា,speaker1
wavs/audio_002.wav,សួស្តី,speaker2
```

**Format 2:** Pipe-separated (original F5-TTS format)
```
audio_file|text
wavs/audio_001.wav|ខ្ញុំស្រលាញ់ប្រទេសកម្ពុជា
wavs/audio_002.wav|សួស្តី
```

### Prepare Dataset

Use the `prepare_data.py` script to process your dataset:

```bash
# Using directory (auto-detects CSV file)
python prepare_data.py --input /path/to/dataset --out_dir ./datasets/khmer_dataset

# Using specific CSV file
python prepare_data.py --input /path/to/train.csv --out_dir ./datasets/khmer_dataset

# With custom worker count
python prepare_data.py --input /path/to/dataset --out_dir ./datasets/khmer_dataset --workers 8
```

#### What This Script Does:

1. **Validates audio files**: Checks existence and integrity using ffprobe
2. **Extracts durations**: Computes audio duration for each file
3. **Builds vocabulary**: Creates `vocab.txt` from all unique characters in transcriptions
4. **Generates metadata**: Creates JSON files with audio paths, text, speaker, and duration
5. **Computes statistics**: Provides dataset statistics (total samples, duration, vocab size)

#### Output Files:

- `processed_data.json`: Main dataset file with audio paths, text, speaker, duration
- `duration.json`: List of all audio durations
- `vocab.txt`: Character-level vocabulary extracted from transcriptions
- `speakers.txt`: List of unique speakers
- `metadata_processed.csv`: Human-readable CSV for reference

## Training Configuration

The training configuration is defined in `F5TTS_Khmer_FT.yaml`.

### Key Configuration Sections

#### Dataset Settings

```yaml
datasets:
  name: /path/to/khmer_dataset  # Output from prepare_data.py
  batch_size_per_gpu: 8192      # Frames per batch
  batch_size_type: frame        # Can be 'frame' or 'sample'
  max_samples: 64               # Max samples per batch
  num_workers: 8                # DataLoader workers
  dataset_type: "JSONDataset"   # Use JSON format
  audio_type: "raw"             # Raw waveform processing
```

#### Model Configuration

```yaml
model:
  name: F5TTS_Khmer_FT
  tokenizer: custom             # Use custom tokenizer for Khmer
  tokenizer_path: /path/to/khmer_dataset  # Contains vocab.txt
  backbone: DiT                 # Diffusion Transformer
  mel_spec:
    target_sample_rate: 16000   # Audio sample rate
    n_mel_channels: 100         # Mel spectrogram channels
    hop_length: 256
```

#### Training Hyperparameters

```yaml
optim:
  epochs: 200
  learning_rate: 5e-5
  num_warmup_updates: 20000
  grad_accumulation_steps: 2    # Effective batch size multiplier
  max_grad_norm: 1.0
```

#### Checkpoint Settings

```yaml
ckpts:
  pretrained_checkpoint: /path/to/F5TTS/model_1250000.safetensors
  pretrained_vocab_path: /path/to/F5TTS/vocab.txt
  expand_vocab: true            # CRITICAL: Enable vocab expansion for Khmer
  save_per_updates: 2000        # Save checkpoint every 2k updates
  keep_last_n_checkpoints: 10   # Keep last 10 checkpoints
  logger: wandb                 # wandb | tensorboard | null
```

### Important Notes

- **expand_vocab: true** is critical when fine-tuning with a different vocabulary (e.g., Khmer vs. English)
- Adjust `batch_size_per_gpu` based on your GPU memory (8192 frames ≈ 12-16GB VRAM)
- Use `grad_accumulation_steps` to simulate larger batch sizes on limited VRAM
- The model uses 16kHz audio by default

## Training

### 1. Update Configuration Paths

Edit `F5TTS_Khmer_FT.yaml` and update the following paths:

- `datasets.name`: Path to your prepared dataset directory
- `model.tokenizer_path`: Same as datasets.name (contains vocab.txt)
- `ckpts.pretrained_checkpoint`: Path to pretrained F5-TTS checkpoint
- `ckpts.pretrained_vocab_path`: Path to pretrained vocab.txt

### 2. Start Training

```bash
cd F5-TTS/src/f5_tts/train

# Basic training
python train.py --config /path/to/F5TTS_Khmer_FT.yaml

# With specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config /path/to/F5TTS_Khmer_FT.yaml

# With multiple GPUs (DDP)
torchrun --nproc_per_node=2 train.py --config /path/to/F5TTS_Khmer_FT.yaml
```

### 3. Monitor Training

If using Weights & Biases (recommended):

```bash
wandb login
# Training logs will be available at https://wandb.ai
```

Checkpoints are saved to:
```
ckpts/F5TTS_Khmer_FT_vocos_custom_khmer_dataset/YYYY-MM-DD/HH-MM-SS/
```

## Dataset Statistics

After running `prepare_data.py`, you'll see output similar to:

```
Dataset statistics for khmer_dataset:
  - Total samples: 15000
  - Number of speakers: 3
  - Vocabulary size: 92
  - Total duration: 25.50 hours
  - Average duration: 6.12 seconds
  - Min duration: 1.50 seconds
  - Max duration: 15.00 seconds
```

### Recommended Dataset Specifications

- **Minimum samples**: 5,000+ utterances
- **Minimum duration**: 10+ hours
- **Audio quality**: 16kHz, mono, clean recordings
- **Text quality**: Accurate transcriptions with proper Khmer Unicode
- **Duration per sample**: 2-15 seconds (optimal: 3-10 seconds)

## Inference

After training, use the fine-tuned model for inference:

```python
from f5_tts.api import F5TTS

# Load your fine-tuned model
model = F5TTS.from_pretrained("path/to/checkpoint.safetensors")

# Generate speech
text = "សួស្តី នេះគឺជាការសាកល្បងសំលេង"
reference_audio = "reference.wav"  # Optional: speaker reference
output = model.generate(text, reference_audio=reference_audio)
```

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size_per_gpu` (try 4096 or 2048)
- Increase `grad_accumulation_steps` to maintain effective batch size
- Enable `checkpoint_activations: True` in model config

### Slow Training

- Increase `num_workers` for data loading
- Use `attn_backend: flash_attn` if you have Flash Attention installed
- Verify GPU utilization with `nvidia-smi`

### Vocabulary Issues

- Ensure `expand_vocab: true` in checkpoint settings
- Verify `vocab.txt` contains all Khmer characters from your dataset
- Check that text encoding is UTF-8

## Citation

If you use this setup in your research, please cite:

```bibtex
@article{chen2024f5tts,
  title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching},
  author={Chen, Yushen and others},
  journal={arXiv preprint arXiv:2410.06885},
  year={2024}
}
```

## File Structure

```
f5tts/
├── README.md                   # This file
├── F5TTS_Khmer_FT.yaml        # Training configuration
└── prepare_data.py            # Data preparation script
```

## Additional Resources

- [F5-TTS GitHub Repository](https://github.com/SWivid/F5-TTS)
- [F5-TTS Paper](https://arxiv.org/abs/2410.06885)
- [Khmer Unicode Reference](https://www.unicode.org/charts/PDF/U1780.pdf)

## License

This implementation follows the F5-TTS license. Please refer to the original repository for licensing details.

## Contact

For questions or issues specific to this Khmer fine-tuning setup, please refer to your research paper or contact the authors.
