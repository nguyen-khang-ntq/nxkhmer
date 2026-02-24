# Orpheus Khmer TTS Training

A comprehensive training pipeline for continued pre-training of the Orpheus-3B model on Khmer language text-to-speech tasks using SNAC audio codec.

## Overview

This project implements a continued pre-training (CPT) pipeline for Orpheus-3B, a state-of-the-art text-to-speech model, specifically adapted for Khmer language. The pipeline processes audio data using the SNAC (Scalable Neural Audio Codec) tokenizer and trains the model using efficient distributed training techniques.

## Features

- **SNAC Audio Tokenization**: Efficient audio encoding using the SNAC 24kHz model
- **Streaming Dataset Support**: Handle large-scale datasets without memory overflow
- **Multi-GPU Training**: Distributed training support with gradient accumulation
- **Memory Optimization**: Gradient checkpointing and efficient memory management
- **WandB Integration**: Comprehensive experiment tracking and logging
- **Speaker-specific Processing**: Flexible speaker filtering and multi-speaker support

## Project Structure

```
.
├── train.py                # Main training script
├── snac_tokenizer.py      # SNAC audio tokenization and dataset preprocessing
└── README.md              # This file
```

## Prerequisites

### System Requirements
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- Python 3.8+
- PyTorch 2.0+

### Dependencies

```bash
pip install torch torchaudio transformers datasets accelerate
pip install snac wandb pandas tqdm
pip install flash-attn --no-build-isolation  # Optional but recommended
```

## Quick Start

### 1. Data Preparation

Prepare your dataset in CSV format with the following columns:
- `audio_file`: Path to audio file
- `text`: Transcription text
- `speaker`: Speaker identifier

Example CSV structure:
```csv
audio_file,text,speaker
/path/to/audio1.wav,ជំរាបសួរ,speaker_001
/path/to/audio2.wav,សួស្តី,speaker_002
```

### 2. Tokenize Audio Data

Process your CSV dataset into SNAC tokens:

```bash
python snac_tokenizer.py
```

**Configuration variables** in `snac_tokenizer.py`:
- `INPUT_CSV`: Path to your CSV file
- `AUDIO_BASE_DIR`: Base directory for audio files
- `OUTPUT_DIR`: Output directory for processed parquet files
- `ALLOWED_SPEAKERS`: Set of speaker IDs to process (empty = all speakers)

The script will:
- Load audio files from CSV
- Tokenize audio using SNAC codec (24kHz)
- Remove duplicate audio frames
- Combine text and audio tokens into training format
- Save as Parquet files for efficient loading

### 3. Train the Model

Run the training script:

```bash
# Single GPU
python train.py

# Multi-GPU with accelerate
accelerate config  # Configure once
accelerate launch train.py

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 train.py
```

**Configuration variables** in `train.py`:
- `BASE_MODEL_WITH_TOKENIZER_PATH`: Path to base Orpheus model with tokenizer
- `MODEL_CHECKPOINT_PATH`: Path to model checkpoint to resume from
- `PRETRAINING_CHUNKS_DIR_1`: Path to processed SNAC data directory
- `TRAINING_OUTPUT_DIR`: Output directory for trained model

## Training Configuration

### Default Hyperparameters

```python
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 2e-4
num_train_epochs = 2
warmup_ratio = 0.03
lr_scheduler_type = "cosine"
optimizer = "adamw_torch_fused"
max_grad_norm = 1.0
```

### Memory Optimization Features

- **Gradient Checkpointing**: Reduces memory usage by ~50%
- **Streaming Dataset**: Prevents loading entire dataset into RAM
- **BF16 Mixed Precision**: Faster training with lower memory footprint
- **Dynamic Memory Allocation**: CUDA expandable segments

## Token Structure

The model uses a specialized token structure combining text and audio:

```
[start_of_human] <text_tokens> [end_of_human]
[start_of_ai] [start_of_speech] <audio_tokens> [end_of_speech] [end_of_ai]
```

### Special Tokens
- Text tokens: 0 - 128255
- Special markers: 128256+
- Audio tokens: Start at `tokenizer_length + 10`

### SNAC Audio Encoding
- 7 codes per audio frame
- Multi-level hierarchical encoding
- Duplicate frame removal for efficiency

## Monitoring

Training metrics are logged to Weights & Biases:

```bash
export WANDB_PROJECT="orpheus-khmer-tts"
export WANDB_RUN_NAME="orpheus-3b-khmer-continued-pretraining"
```

View your training progress at: https://wandb.ai/

## Output

After training, the following files will be saved to `TRAINING_OUTPUT_DIR`:

- `pytorch_model.bin` or `model.safetensors`: Trained model weights
- `config.json`: Model configuration
- `training_args.bin`: Training arguments
- `trainer_state.json`: Trainer state for resuming
- Checkpoint directories: `checkpoint-{step}/`

## Advanced Usage

### Resume from Checkpoint

```bash
python train.py --resume_from_checkpoint ./output/checkpoint-1000
```

### Custom Max Steps

If streaming dataset doesn't allow length calculation:

```bash
python train.py --max_steps 50000
```

### Filter by Speaker

Edit `snac_tokenizer.py`:

```python
ALLOWED_SPEAKERS = {
    "speaker_001",
    "speaker_002",
    # Add your speaker IDs
}
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps`
3. Enable gradient checkpointing (already enabled by default)
4. Use lower precision: `--bf16` or `--fp16`

### Dataset Issues

- Ensure audio files exist at specified paths
- Check audio format compatibility (WAV, MP3, FLAC supported)
- Verify CSV encoding (UTF-8 recommended)

### CUDA Errors

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Performance Tips

1. **Use Flash Attention 2**: Significantly faster self-attention
2. **Enable Fused Optimizer**: `adamw_torch_fused` is faster than standard AdamW
3. **Adjust Workers**: Set `dataloader_num_workers` based on CPU cores
4. **Batch Size Tuning**: Find optimal batch size for your GPU


## Acknowledgments

- Orpheus model: [unsloth/orpheus-3b](https://huggingface.co/unsloth/orpheus-3b-0.1-ft)
- SNAC codec: [hubertsiuzdak/snac_24khz](https://huggingface.co/hubertsiuzdak/snac_24khz)
- Transformers library by Hugging Face

