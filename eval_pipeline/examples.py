"""
Example usage of TTS Evaluation Pipeline
"""

from evaluator import TTSEvaluator
import json
from pathlib import Path


def example_single_file_evaluation():
    """Example: Evaluate a single TTS output"""
    print("="*60)
    print("EXAMPLE 1: Single File Evaluation")
    print("="*60)
    
    # Initialize evaluator with OmniASR
    evaluator = TTSEvaluator(
        asr_model_name="1B-CTC",  # OmniASR model: 300M, 1B, 3B, or 7B with CTC or LLM
        asr_type="omni",  # Use omnilingual-asr
        mos_model_name="cdminix/wav2vec2-base-utmos",
        device="cuda"
    )
    
    # Or use Whisper instead:
    # evaluator = TTSEvaluator(
    #     asr_model_name="openai/whisper-large-v3",
    #     asr_type="whisper",
    #     device="cuda"
    # )
    
    # Evaluate single file
    metrics = evaluator.evaluate_single(
        generated_audio="path/to/your/generated_audio.wav",
        reference_text="This is the reference transcription text",
        reference_audio="path/to/your/reference_audio.wav"
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 60)
    print(f"CER (Character Error Rate):     {metrics.cer:.4f} (lower is better)")
    print(f"MOS (Mean Opinion Score):       {metrics.mos:.4f} (1-5, higher is better)")
    print(f"SIM (Speaker Similarity):       {metrics.sim:.4f} (0-1, higher is better)")
    print(f"RMSE F0 (Pitch Error):          {metrics.rmse_f0:.4f} (lower is better)")
    print(f"MCD (Mel-Cepstral Distortion): {metrics.mcd:.4f} (lower is better)")
    print(f"SMOS (Synthetic MOS):           {metrics.smos:.4f} (1-5, higher is better)")
    print("-" * 60)


def example_batch_evaluation():
    """Example: Evaluate multiple TTS outputs"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*60)
    
    # Use OmniASR by default
    evaluator = TTSEvaluator(
        asr_model_name="1B-CTC",
        asr_type="omni",
        device="cuda"
    )
    
    audio_pairs = [
        {
            'id': 'utterance_001',
            'generated': 'path/to/generated_001.wav',
            'reference_text': 'Hello, how are you today?',
            'reference_audio': 'path/to/reference_001.wav'
        },
        {
            'id': 'utterance_002',
            'generated': 'path/to/generated_002.wav',
            'reference_text': 'The weather is beautiful.',
            'reference_audio': 'path/to/reference_002.wav'
        }
    ]
    
    results_df = evaluator.evaluate_batch(
        audio_pairs=audio_pairs,
        output_file="batch_evaluation_results.csv"
    )


def example_individual_metrics():
    """Example: Calculate ind
        asr_model_name="3B-CTC",  # Can use different sizes
        asr_type="omni",
        device="cuda"
    cs"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Individual Metric Calculation")
    print("="*60)
    
    evaluator = TTSEvaluator(device="cuda")
    
    gen_audio = "path/to/generated.wav"
    ref_audio = "path/to/reference.wav"
    ref_text = "This is the reference text"
    
    # Calculate specific metrics
    print("\nCalculating individual metrics...")
    
    mos = evaluator.calculate_mos(gen_audio)
    print(f"MOS: {mos:.4f}")
    
    sim = evaluator.calculate_similarity(gen_audio, ref_audio)
    print(f"Speaker Similarity: {sim:.4f}")
    
    cer = evaluator.calculate_cer(gen_audio, ref_text)
    print(f"CER: {cer:.4f}")


if __name__ == "__main__":
    print("TTS Evaluation Pipeline - Example Usage\n")
    
    # Uncomment the example you want to run
    # example_single_file_evaluation()
    # example_batch_evaluation()
    # example_individual_metrics()
    
    print("\nPlease uncomment one of the examples to run it.")
