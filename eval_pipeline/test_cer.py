"""
Quick CER calculation for Orpheus generated audio
Simple script to calculate CER without full evaluator
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch


def calculate_cer_simple(predicted_text, reference_text):
    """
    Calculate Character Error Rate (CER)
    
    Args:
        predicted_text: Transcribed text from ASR
        reference_text: Ground truth text
        
    Returns:
        float: CER score
    """
    # Remove spaces for character-level comparison
    pred = predicted_text.replace(" ", "")
    ref = reference_text.replace(" ", "")
    
    # Simple Levenshtein distance
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    
    # Dynamic programming for edit distance
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == ref[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n] / len(ref)


def main():
    """Calculate CER for generated audio"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate CER for generated audio')
    parser.add_argument('--csv', required=True, help='Path to CSV file with reference text')
    parser.add_argument('--generated-dir', required=True, help='Directory containing generated audio files')
    parser.add_argument('--output-csv', default='./cer_results_simple.csv', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Paths
    csv_path = args.csv
    generated_audio_dir = args.generated_dir
    output_csv = args.output_csv

    print("="*70)
    print("CER Calculation for Generated Audio")
    print("="*70)
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Get list of generated audio files
    print(f"\nScanning generated audio directory...")
    generated_files = {}
    for audio_file in os.listdir(generated_audio_dir):
        if audio_file.endswith("_generated.wav"):
            base_name = audio_file.replace("_generated.wav", "")
            generated_files[base_name] = os.path.join(generated_audio_dir, audio_file)
    
    print(f"Found {len(generated_files)} generated audio files")
    
    # Match CSV entries with generated audio
    matched_samples = []
    for idx, row in df.iterrows():
        audio_path = row['audio_file']
        text = row['text']
        base_name = Path(audio_path).stem
        
        if base_name in generated_files:
            matched_samples.append({
                'base_name': base_name,
                'generated_audio': generated_files[base_name],
                'reference_text': text,
            })
    
    print(f"Matched {len(matched_samples)} samples")
    
    if len(matched_samples) == 0:
        print("\n⚠ No matching samples found!")
        return
    
    # Load ASR model
    print("\n" + "="*70)
    print("Loading ASR Model...")
    print("="*70)
    
    # Import ASR factory from utils
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import ASRFactory
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model_card = "omniASR_LLM_7B"  # Omni ASR model
    print(f"Loading {model_card}...")
    
    asr_model = ASRFactory.create_omni_asr(
        model_card=model_card,
        device=device
    )
    
    print("✓ ASR model loaded")
    
    # Calculate CER
    print("\n" + "="*70)
    print("Calculating CER...")
    print("="*70)
    
    results = []
    for sample in tqdm(matched_samples):
        try:
            # Load and resample audio to 16kHz if needed
            import torchaudio
            waveform, sr = torchaudio.load(sample['generated_audio'])
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            # Save resampled audio temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
                torchaudio.save(tmp_path, waveform, 16000)
            
            # Transcribe using ASR model
            result = asr_model(tmp_path)
            transcription = result['text']
            
            # Clean up temp file
            os.remove(tmp_path)
            
            # Calculate CER
            cer = calculate_cer_simple(transcription, sample['reference_text'])
            
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'transcription': transcription,
                'cer': cer
            })
            
        except Exception as e:
            print(f"\n⚠ Error: {sample['base_name']}: {e}")
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'transcription': 'ERROR',
                'cer': -1.0
            })
    
    # Results
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['cer'] >= 0]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total: {len(results_df)}, Success: {len(valid_results)}, Failed: {len(results_df) - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nCER Statistics:")
        print(f"  Average: {valid_results['cer'].mean():.4f}")
        print(f"  Median:  {valid_results['cer'].median():.4f}")
        print(f"  Min:     {valid_results['cer'].min():.4f}")
        print(f"  Max:     {valid_results['cer'].max():.4f}")
    
    # Save
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")


if __name__ == "__main__":
    main()
