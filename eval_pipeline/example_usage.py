"""
Example usage of TTS Evaluation Pipeline
"""

from tts_evaluator import TTSEvaluator
import json
from pathlib import Path


def example_single_file_evaluation():
    """Example: Evaluate a single TTS output"""
    print("="*60)
    print("EXAMPLE 1: Single File Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = TTSEvaluator(
        asr_model_name="openai/whisper-large-v3",
        mos_model_name="cdminix/wav2vec2-base-utmos",
        device="cuda"  # or "cpu"
    )
    
    # Evaluate single file
    metrics = evaluator.evaluate_single(
        generated_audio="path/to/your/generated_audio.wav",
        reference_text="This is the reference transcription text",
        reference_audio="path/to/your/reference_audio.wav",
        compute_cer=True,
        compute_mos=True,
        compute_sim=True,
        compute_f0=True,
        compute_mcd=True,
        compute_smos=True
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
    
    # Save to JSON
    with open('single_eval_results.json', 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print("\nResults saved to: single_eval_results.json")


def example_batch_evaluation():
    """Example: Evaluate multiple TTS outputs"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Batch Evaluation")
    print("="*60)
    
    # Initialize evaluator
    evaluator = TTSEvaluator(device="cuda")
    
    # Prepare batch data
    # Format: list of dictionaries with generated audio, reference text, and reference audio
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
        },
        {
            'id': 'utterance_003',
            'generated': 'path/to/generated_003.wav',
            'reference_text': 'Machine learning is fascinating.',
            'reference_audio': 'path/to/reference_003.wav'
        }
    ]
    
    # Run batch evaluation
    results_df = evaluator.evaluate_batch(
        audio_pairs=audio_pairs,
        output_file="batch_evaluation_results.csv"
    )
    
    print("\nBatch evaluation complete!")
    print(f"Results shape: {results_df.shape}")
    print("\nFirst few results:")
    print(results_df.head())


def example_with_directory():
    """Example: Evaluate all files in a directory"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Directory-based Evaluation")
    print("="*60)
    
    # Directories
    generated_dir = Path("path/to/generated_audio_dir")
    reference_dir = Path("path/to/reference_audio_dir")
    reference_texts_file = "path/to/reference_texts.json"  # JSON with {filename: text}
    
    # Load reference texts
    with open(reference_texts_file, 'r') as f:
        reference_texts = json.load(f)
    
    # Prepare batch data
    audio_pairs = []
    for gen_file in sorted(generated_dir.glob("*.wav")):
        filename = gen_file.stem
        ref_file = reference_dir / f"{filename}.wav"
        
        if ref_file.exists() and filename in reference_texts:
            audio_pairs.append({
                'id': filename,
                'generated': str(gen_file),
                'reference_text': reference_texts[filename],
                'reference_audio': str(ref_file)
            })
    
    print(f"Found {len(audio_pairs)} audio pairs to evaluate")
    
    # Initialize and evaluate
    evaluator = TTSEvaluator(device="cuda")
    results_df = evaluator.evaluate_batch(
        audio_pairs=audio_pairs,
        output_file="directory_evaluation_results.csv"
    )
    
    print("\nDirectory evaluation complete!")


def example_mos_only():
    """Example: Evaluate only MOS (no reference needed)"""
    print("\n" + "="*60)
    print("EXAMPLE 4: MOS-only Evaluation (No Reference)")
    print("="*60)
    
    evaluator = TTSEvaluator(device="cuda")
    
    # Evaluate only MOS for generated audio
    audio_files = [
        "path/to/generated_001.wav",
        "path/to/generated_002.wav",
        "path/to/generated_003.wav"
    ]
    
    mos_scores = []
    for audio_file in audio_files:
        mos = evaluator.calculate_mos(audio_file)
        mos_scores.append({'file': Path(audio_file).name, 'mos': mos})
        print(f"{Path(audio_file).name}: MOS = {mos:.4f}")
    
    # Save results
    import pandas as pd
    pd.DataFrame(mos_scores).to_csv("mos_only_results.csv", index=False)


def example_custom_metrics():
    """Example: Calculate individual metrics separately"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Metric Selection")
    print("="*60)
    
    evaluator = TTSEvaluator(device="cuda")
    
    gen_audio = "path/to/generated.wav"
    ref_audio = "path/to/reference.wav"
    ref_text = "This is the reference text"
    
    # Calculate only specific metrics
    print("\nCalculating individual metrics...")
    
    # Only MOS
    mos = evaluator.calculate_mos(gen_audio)
    print(f"MOS: {mos:.4f}")
    
    # Only speaker similarity
    sim = evaluator.calculate_similarity(gen_audio, ref_audio)
    print(f"Speaker Similarity: {sim:.4f}")
    
    # Only CER
    cer = evaluator.calculate_cer(gen_audio, ref_text)
    print(f"CER: {cer:.4f}")
    
    # Only F0 RMSE
    rmse_f0 = evaluator.calculate_rmse_f0(gen_audio, ref_audio)
    print(f"RMSE F0: {rmse_f0:.4f}")
    
    # Only MCD
    mcd = evaluator.calculate_mcd(gen_audio, ref_audio)
    print(f"MCD: {mcd:.4f}")


if __name__ == "__main__":
    print("TTS Evaluation Pipeline - Example Usage\n")
    
    # Run examples (uncomment the one you want to try)
    
    # Example 1: Single file evaluation
    # example_single_file_evaluation()
    
    # Example 2: Batch evaluation
    # example_batch_evaluation()
    
    # Example 3: Directory-based evaluation
    # example_with_directory()
    
    # Example 4: MOS-only evaluation
    # example_mos_only()
    
    # Example 5: Custom metric selection
    # example_custom_metrics()
    
    print("\nPlease uncomment one of the examples above to run it.")
    print("Make sure to update the file paths to your actual audio files.")
