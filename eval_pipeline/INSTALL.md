# Installation Guide

## Basic Dependencies

```bash
pip install -r requirements.txt
```

## ASR Model Setup

### Option 1: Use Whisper (No additional setup)
```python
evaluator = TTSEvaluator(
    asr_model_name="openai/whisper-large-v3",
    asr_type="whisper",
    device="cuda"
)
```

### Option 2: Use OmniASR

#### If OmniASR has official repository:
```bash
# Install OmniASR package (replace with actual repo)
git clone <omniasr-repository-url>
cd omniasr
pip install -e .
```

#### Using pre-trained weights directly:
The current implementation loads OmniASR models directly from `.pt` files.

**Requirements:**
- Model files: `<model_dir>/omniASR-{TYPE}-{SIZE}.pt`
- Tokenizer: `<model_dir>/omniASR_tokenizer_v7.model`

**Usage:**
```python
evaluator = TTSEvaluator(
    asr_model_name="1B-CTC",  # Size: 300M, 1B, 3B, 7B; Type: CTC or LLM
    asr_type="omni",
    device="cuda"
)
```

**Note:** The wrapper in `utils/custom_asr.py` provides basic inference. If OmniASR has specific preprocessing or model architecture requirements, you may need to install the official package.

## Checking Your Setup

```python
# Test if OmniASR loads correctly
from utils import ASRFactory

try:
    asr = ASRFactory.create_omni_asr(
        model_size="1B",
        model_type="CTC",
        device="cuda"
    )
    print("✓ OmniASR loaded successfully")
    
    # Test transcription
    result = asr("path/to/test.wav")
    print(f"Transcription: {result['text']}")
except Exception as e:
    print(f"✗ Error loading OmniASR: {e}")
    print("Consider using Whisper instead or install OmniASR package")
```

## Troubleshooting

### OmniASR Model Loading Issues

If you see errors loading OmniASR models, you might need the official implementation:

1. **Check if official repo exists:**
   - Search for "omnilingual-asr" on GitHub
   - Check ESPnet repositories
   - Look for model cards on HuggingFace

2. **Alternative: Use the official loading method**
   
   Update `utils/custom_asr.py` to use the official OmniASR loader if available:
   ```python
   # Instead of torch.load, use official loader
   from omniasr import OmniASRModel
   self.model = OmniASRModel.from_pretrained(model_path)
   ```

3. **Fallback to Whisper:**
   ```python
   evaluator = TTSEvaluator(asr_type="whisper")
   ```

### Model Architecture Compatibility

The current implementation assumes the model checkpoint contains a complete model. If it's just weights, you may need:

1. The model architecture definition
2. Config files
3. Official loading utilities

**If you have the official OmniASR repository**, please share its location and I can update the integration accordingly.
