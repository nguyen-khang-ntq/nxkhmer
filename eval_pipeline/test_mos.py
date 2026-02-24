"""
Quick MOS calculation for Orpheus generated audio
Simple script to calculate MOS (DNSMOS) without full evaluator
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch


def main():
    """Calculate MOS for generated audio"""
    
    # Paths
    csv_path = "/home/coder/datasets/crawl_datasets/final_dataset/test_clean.csv"
    generated_audio_dir = "/home/coder/data/Interspeech/model/XTTSv2-Finetuning-for-New-Languages/output"
    output_csv = "/home/coder/data/Speech/TTS/Orpheus/mos_results_simple.csv"
    model_path = "/home/coder/data/Speech/Data/Amphion/preprocessors/Emilia/pretrained_models/dnsmos/sig_bak_ovr.onnx"
    
    print("="*70)
    print("MOS Calculation for Generated Audio")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n⚠ DNSMOS model not found at: {model_path}")
        print("Please download the model from: https://github.com/microsoft/DNS-Challenge")
        return
    
    # Load CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[:100]
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
    
    # Load DNSMOS Calculator
    print("\n" + "="*70)
    print("Loading DNSMOS Calculator...")
    print("="*70)
    
    # Import DNSMOS calculator from metrics
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from metrics.dnsmos import DNSMOSCalculator
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    dnsmos_calculator = DNSMOSCalculator(
        model_path=model_path,
        device=device,
        is_personalized_MOS=False
    )
    
    print("✓ DNSMOS calculator loaded")
    
    # Calculate MOS
    print("\n" + "="*70)
    print("Calculating MOS Scores...")
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
            
            # Calculate MOS scores
            mos_scores = dnsmos_calculator.calculate(tmp_path)
            
            # Clean up temp file
            os.remove(tmp_path)
            
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'ovrl_mos': mos_scores['OVRL'],
                'sig_mos': mos_scores['SIG'],
                'bak_mos': mos_scores['BAK'],
                'ovrl_mos_raw': mos_scores['OVRL_raw'],
                'sig_mos_raw': mos_scores['SIG_raw'],
                'bak_mos_raw': mos_scores['BAK_raw'],
            })
            
        except Exception as e:
            print(f"\n⚠ Error: {sample['base_name']}: {e}")
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'ovrl_mos': -1.0,
                'sig_mos': -1.0,
                'bak_mos': -1.0,
                'ovrl_mos_raw': -1.0,
                'sig_mos_raw': -1.0,
                'bak_mos_raw': -1.0,
            })
    
    # Results
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['ovrl_mos'] >= 0]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total: {len(results_df)}, Success: {len(valid_results)}, Failed: {len(results_df) - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nMOS Statistics:")
        print(f"  Overall MOS:")
        print(f"    Average: {valid_results['ovrl_mos'].mean():.4f}")
        print(f"    Median:  {valid_results['ovrl_mos'].median():.4f}")
        print(f"    Min:     {valid_results['ovrl_mos'].min():.4f}")
        print(f"    Max:     {valid_results['ovrl_mos'].max():.4f}")
        print(f"\n  Signal MOS:")
        print(f"    Average: {valid_results['sig_mos'].mean():.4f}")
        print(f"    Median:  {valid_results['sig_mos'].median():.4f}")
        print(f"\n  Background MOS:")
        print(f"    Average: {valid_results['bak_mos'].mean():.4f}")
        print(f"    Median:  {valid_results['bak_mos'].median():.4f}")
    
    # Save
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")


if __name__ == "__main__":
    main()
