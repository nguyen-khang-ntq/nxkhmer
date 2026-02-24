"""
Quick MCD (Mel-Cepstral Distortion) calculation for generated audio
Mel-Cepstral Distortion quantifies the spectral difference between
synthesized speech and the ground truth audio. Lower MCD indicates
higher spectral similarity and better quality.

Using pymcd package: https://github.com/chenqi008/pymcd
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from pymcd.mcd import Calculate_MCD


def main():
    """Calculate MCD for generated audio"""
    
    # Paths
    csv_path = "/home/coder/datasets/crawl_datasets/final_dataset/test_clean.csv"
    generated_audio_dir = "/home/coder/data/Speech/TTS/khmer-tts/recipes/interspeech/vits/test_generated_mms"
    reference_audio_base = "/home/coder/datasets/crawl_datasets/final_dataset"  # Base directory for reference audio
    output_csv = "/home/coder/data/Interspeech/eval_pipeline/mcd_results.csv"
    
    print("="*70)
    print("MCD Calculation for Generated Audio")
    print("="*70)
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[:100]  # Limit to first 100 samples
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
            # Construct full path to reference audio
            full_reference_path = os.path.join(reference_audio_base, audio_path)
            
            # Check if reference audio exists
            if os.path.exists(full_reference_path):
                matched_samples.append({
                    'base_name': base_name,
                    'generated_audio': generated_files[base_name],
                    'reference_audio': full_reference_path,
                    'reference_text': text,
                })
            else:
                print(f"\n⚠ Reference audio not found: {full_reference_path}")
    
    print(f"Matched {len(matched_samples)} samples with both generated and reference audio")
    
    if len(matched_samples) == 0:
        print("\n⚠ No matching samples found!")
        return
    
    # Initialize MCD calculator
    print("\n" + "="*70)
    print("Initializing MCD Calculator (pymcd)...")
    print("="*70)
    
    # Initialize MCD calculator with DTW mode (Dynamic Time Warping)
    # Modes: "plain", "dtw", "dtw_sl"
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    print("✓ MCD calculator initialized (mode: dtw)")
    
    # Calculate MCD
    print("\n" + "="*70)
    print("Calculating MCD (Mel-Cepstral Distortion)...")
    print("="*70)
    print("Note: Lower MCD values indicate better spectral similarity")
    
    results = []
    for sample in tqdm(matched_samples, desc="Processing"):
        try:
            # Calculate MCD using pymcd
            mcd_score = mcd_toolbox.calculate_mcd(
                sample['reference_audio'],  # reference (ground-truth)
                sample['generated_audio']    # synthesized
            )
            
            results.append({
                'audio_name': sample['base_name'],
                'generated_audio': sample['generated_audio'],
                'reference_audio': sample['reference_audio'],
                'reference_text': sample['reference_text'],
                'mcd': mcd_score,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n⚠ Error processing {sample['base_name']}: {e}")
            results.append({
                'audio_name': sample['base_name'],
                'generated_audio': sample['generated_audio'],
                'reference_audio': sample['reference_audio'],
                'reference_text': sample['reference_text'],
                'mcd': -1.0,
                'status': f'error: {str(e)}'
            })
    
    # Analyze Results
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['mcd'] >= 0]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total: {len(results_df)}")
    print(f"Success: {len(valid_results)}")
    print(f"Failed: {len(results_df) - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nMCD Statistics (lower is better):")
        print(f"  Average: {valid_results['mcd'].mean():.4f}")
        print(f"  Median:  {valid_results['mcd'].median():.4f}")
        print(f"  Std Dev: {valid_results['mcd'].std():.4f}")
        print(f"  Min:     {valid_results['mcd'].min():.4f}")
        print(f"  Max:     {valid_results['mcd'].max():.4f}")
        
        # Show percentiles
        print(f"\nPercentiles:")
        print(f"  25th: {valid_results['mcd'].quantile(0.25):.4f}")
        print(f"  50th: {valid_results['mcd'].quantile(0.50):.4f}")
        print(f"  75th: {valid_results['mcd'].quantile(0.75):.4f}")
        print(f"  90th: {valid_results['mcd'].quantile(0.90):.4f}")
        
        # Show best and worst samples
        print(f"\nBest samples (lowest MCD):")
        best_samples = valid_results.nsmallest(5, 'mcd')
        for idx, row in best_samples.iterrows():
            print(f"  {row['audio_name']}: {row['mcd']:.4f}")
        
        print(f"\nWorst samples (highest MCD):")
        worst_samples = valid_results.nlargest(5, 'mcd')
        for idx, row in worst_samples.iterrows():
            print(f"  {row['audio_name']}: {row['mcd']:.4f}")
    
    # Save results
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")
    
    # Show failed samples if any
    failed_results = results_df[results_df['mcd'] < 0]
    if len(failed_results) > 0:
        print(f"\nFailed samples:")
        for idx, row in failed_results.iterrows():
            print(f"  {row['audio_name']}: {row['status']}")


if __name__ == "__main__":
    main()
