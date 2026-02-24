# TTS Evaluation Pipeline

Modular evaluation pipeline for Text-to-Speech (TTS) models with multiple objective metrics.

## Project Structure

```
eval_pipeline/
├── config.py              # Configuration and data classes
├── evaluator.py           # Main orchestrator
├── examples.py            # Usage examples
├── requirements.txt       # Dependencies
├── README.md             # Documentation
├── utils/
│   ├── __init__.py
│   ├── audio.py          # Audio processing utilities
│   └── models.py         # Model loading utilities
└── metrics/
    ├── __init__.py
    ├── cer.py            # Character Error Rate
    ├── mos.py            # Mean Opinion Score
    ├── similarity.py     # Speaker Similarity
    ├── f0.py             # F0 RMSE
    ├── mcd.py            # Mel-Cepstral Distortion
    └── smos.py           # Synthetic MOS
```

## Supported Metrics

1. **CER** - Character Error Rate (transcription accuracy)
2. **MOS** - Mean Opinion Score (predicted quality 1-5)
3. **SIM** - Speaker Similarity (0-1)
4. **RMSE_F0** - F0/Pitch error
5. **MCD** - Mel-Cepstral Distortion (acoustic quality)
6. **SMOS** - Synthetic MOS (combined metric)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from evaluator import TTSEvaluator

# Initialize
evaluator = TTSEvaluator(device="cuda")

# Evaluate single file
metrics = evaluator.evaluate_single(
    generated_audio="generated.wav",
    reference_text="The reference text",
    reference_audio="reference.wav"
)

print(f"CER: {metrics.cer:.4f}")
print(f"MOS: {metrics.mos:.4f}")
print(f"SIM: {metrics.sim:.4f}")
```

### Batch Evaluation

```python
audio_pairs = [
    {
        'id': 'sample_1',
        'generated': 'gen_1.wav',
        'reference_text': 'Hello world',
        'reference_audio': 'ref_1.wav'
    },
    # ... more samples
]

results_df = evaluator.evaluate_batch(
    audio_pairs=audio_pairs,
    output_file="results.csv"
)
```

### Individual Metrics

```python
# Calculate specific metrics only
mos = evaluator.calculate_mos("generated.wav")
sim = evaluator.calculate_similarity("gen.wav", "ref.wav")
cer = evaluator.calculate_cer("gen.wav", "reference text")
mcd = evaluator.calculate_mcd("gen.wav", "ref.wav")
```

## Module Details

### Config (`config.py`)
- `ModelConfig`: Model and device configuration
- `EvalMetrics`: Data class for storing metric results

### Utils (`utils/`)
- `audio.py`: Audio loading, MFCC extraction, sequence alignment
- `models.py`: Model loading for ASR, MOS, and speaker encoder

### Metrics (`metrics/`)
Each metric has its own calculator class:
- `CERCalculator`: Uses Whisper ASR for transcription
- `MOSCalculator`: Uses UTMOS predictor
- `SimilarityCalculator`: Uses Resemblyzer embeddings
- `F0Calculator`: Uses Praat/Parselmouth for pitch
- `MCDCalculator`: MFCC + DTW alignment
- `SMOSCalculator`: Combines MOS and similarity

### Evaluator (`evaluator.py`)
Main orchestrator that:
- Loads all models
- Initializes all calculators
- Provides unified interface for evaluation
- Handles batch processing

## Examples

See `examples.py` for:
1. Single file evaluation
2. Batch evaluation
3. Individual metric calculation

## Configuration

```python
evaluator = TTSEvaluator(
    asr_model_name="openai/whisper-large-v3",
    mos_model_name="cdminix/wav2vec2-base-utmos",
    device="cuda",
    sample_rate=16000
)
```

## Selective Metrics

```python
# Disable specific metrics
metrics = evaluator.evaluate_single(
    generated_audio="gen.wav",
    reference_audio="ref.wav",
    compute_cer=False,      # Skip CER
    compute_mcd=False,      # Skip MCD
    compute_mos=True,
    compute_sim=True,
    compute_f0=True,
    compute_smos=True
)
```

## Output Format

**Single Evaluation:**
```python
EvalMetrics(
    cer=0.05,
    mos=4.2,
    sim=0.87,
    rmse_f0=15.3,
    mcd=4.8,
    smos=4.1
)
```

**Batch Evaluation:**
CSV with columns: `id, cer, mos, sim, rmse_f0, mcd, smos`

## Extending the Pipeline

### Add New Metric

1. Create `metrics/new_metric.py`:
```python
class NewMetricCalculator:
    def __init__(self, required_models):
        self.models = required_models
    
    def calculate(self, audio_path):
        # Your calculation logic
        return score
```

2. Update `metrics/__init__.py`:
```python
from .new_metric import NewMetricCalculator
__all__ = [..., 'NewMetricCalculator']
```

3. Initialize in `evaluator.py`:
```python
self.new_metric_calculator = NewMetricCalculator(...)
```

## Performance Tips

- Use GPU for faster evaluation
- Batch process multiple files
- Disable expensive metrics if not needed
- Use smaller models for faster inference

## License

MIT License
