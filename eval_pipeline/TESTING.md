# TTS Evaluation Pipeline - Testing Guide

## Test Files

Individual test files for each component:

1. **test_audio_utils.py** - Test audio loading, MFCC extraction, sequence alignment
2. **test_asr.py** - Test ASR models (OmniASR and Whisper)
3. **test_cer.py** - Test Character Error Rate calculation
4. **test_mos.py** - Test Mean Opinion Score prediction
5. **test_similarity.py** - Test speaker similarity calculation
6. **test_f0.py** - Test F0 extraction and RMSE calculation
7. **test_mcd.py** - Test Mel-Cepstral Distortion calculation

## Running Tests

### Run Individual Tests

```bash
cd /home/coder/data/Speech/TTS/eval_pipeline

# Test audio utilities (no model download required)
python test_audio_utils.py

# Test ASR models
python test_asr.py

# Test specific metrics
python test_cer.py
python test_mos.py
python test_similarity.py
python test_f0.py
python test_mcd.py
```

### Run All Tests

```bash
python run_all_tests.py
```

## Test Requirements

### Minimal Tests (No model download)
- `test_audio_utils.py` - Only needs librosa
- Tests with mock data don't require audio files

### Tests Requiring Audio Files
Prepare some test audio files:
- Generated TTS audio samples
- Reference audio samples
- Any .wav files for testing

### Tests Requiring Model Download
- `test_asr.py` - Downloads Whisper or loads OmniASR
- `test_mos.py` - Downloads UTMOS model (~400MB)
- `test_similarity.py` - Downloads Resemblyzer embeddings

## Quick Start

### 1. Test Basic Functions (No downloads)
```bash
python test_audio_utils.py
```
When prompted, press Enter to skip tests requiring audio files.

### 2. Test with Your Audio Files
```bash
python test_mos.py
# Enter path to your audio file when prompted
```

### 3. Test ASR Models

**For OmniASR:**
```bash
python test_asr.py
# Select OmniASR
# Choose model size (1B recommended for testing)
# Provide audio file path
```

**For Whisper:**
```bash
python test_asr.py
# Select Whisper
# Use whisper-base for faster testing
```

## Expected Outputs

### test_audio_utils.py
```
✓ Audio loaded successfully
  Shape: (160000,)
  Sample rate: 16000 Hz
  Duration: 10.00 seconds
```

### test_cer.py
```
✓ CER Score: 0.0523
  Reference: 'hello world'
  Hypothesis: 'hello word'
```

### test_mos.py
```
✓ MOS Score: 3.85
  Scale: 1.0 (poor) to 5.0 (excellent)
```

### test_similarity.py
```
✓ Similarity Score: 0.82
  → Very similar (likely same speaker)
```

### test_f0.py
```
✓ F0 extracted successfully
  Mean F0: 180.5 Hz
  → Medium pitch
```

### test_mcd.py
```
✓ MCD Score: 5.23 dB
  → Good quality (MCD < 6.0)
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the eval_pipeline directory
cd /home/coder/data/Speech/TTS/eval_pipeline

# Install dependencies
pip install -r requirements.txt
```

### OmniASR Loading Error
- Check if model files exist in `/home/coder/data/Speech/ASR/omnilingual-asr/`
- Try Whisper instead: Use `asr_type="whisper"` in tests
- See INSTALL.md for setup instructions

### CUDA Out of Memory
- Use smaller models (whisper-base, OmniASR 300M)
- Use CPU: Change device to "cpu" in tests
- Close other GPU processes

### Audio File Format Issues
- Convert to WAV: `ffmpeg -i input.mp3 output.wav`
- Ensure 16kHz sample rate: `ffmpeg -i input.wav -ar 16000 output_16k.wav`

## Integration Test

After individual tests pass, test the full pipeline:

```python
from evaluator import TTSEvaluator

evaluator = TTSEvaluator(
    asr_model_name="1B-CTC",
    asr_type="omni",
    device="cuda"
)

metrics = evaluator.evaluate_single(
    generated_audio="path/to/generated.wav",
    reference_text="reference text",
    reference_audio="path/to/reference.wav"
)

print(metrics)
```

## CI/CD Testing

For automated testing without user input, create test fixtures:

```python
# test_fixtures.py
TEST_AUDIO = "test_data/sample.wav"
TEST_TEXT = "hello world"

# Use in tests
if Path(TEST_AUDIO).exists():
    # Run automated tests
    pass
```
