"""
Quick Speaker Similarity calculation for Orpheus generated audio
Simple script to calculate speaker similarity without full evaluator
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch


def main():
    """Calculate Speaker Similarity for generated audio"""
    
    # Paths
    csv_path = "/home/coder/datasets/crawl_datasets/final_dataset/test_clean_long.csv"
    generated_audio_dir = "/home/coder/data/Speech/TTS/neutts-air/neuttsair/generated_neutts_audio"
    reference_audio_base = "/home/coder/datasets/crawl_datasets/final_dataset"
    output_csv = "/home/coder/data/Speech/TTS/Orpheus/similarity_results_simple.csv"
    
    print("="*70)
    print("Speaker Similarity Calculation for Generated Audio")
    print("="*70)
    
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
        
        # Construct full reference audio path
        if not os.path.isabs(audio_path):
            reference_audio = os.path.join(reference_audio_base, audio_path)
        else:
            reference_audio = audio_path
        
        if base_name in generated_files and os.path.exists(reference_audio):
            matched_samples.append({
                'base_name': base_name,
                'generated_audio': generated_files[base_name],
                'reference_audio': reference_audio,
                'reference_text': text,
            })
    
    print(f"Matched {len(matched_samples)} samples with both generated and reference audio")
    
    if len(matched_samples) == 0:
        print("\n⚠ No matching samples found!")
        return
    
    # Load Speaker Encoder (ECAPA-TDNN)
    print("\n" + "="*70)
    print("Loading ECAPA-TDNN Speaker Encoder...")
    print("="*70)
    
    # Import speechbrain
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    import torch.nn.functional as F
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load ECAPA-TDNN model from speechbrain
    print("Loading speechbrain/spkrec-ecapa-voxceleb...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    print("✓ ECAPA-TDNN speaker encoder loaded")
    
    # Calculate Similarity
    print("\n" + "="*70)
    print("Calculating Speaker Similarity...")
    print("="*70)
    
    results = []
    for sample in tqdm(matched_samples):
        try:
            # Load audio files
            wav1, sr1 = torchaudio.load(sample['generated_audio'])
            wav2, sr2 = torchaudio.load(sample['reference_audio'])
            
            # Resample to 16kHz if needed (speechbrain expects 16kHz)
            if sr1 != 16000:
                resampler1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
                wav1 = resampler1(wav1)
            if sr2 != 16000:
                resampler2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
                wav2 = resampler2(wav2)
            
            # Get embeddings
            emb1 = classifier.encode_batch(wav1)
            emb2 = classifier.encode_batch(wav2)
            
            # Calculate cosine similarity
            similarity = F.cosine_similarity(emb1, emb2, dim=-1).item()
            
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'similarity': similarity,
            })
            
        except Exception as e:
            print(f"\n⚠ Error: {sample['base_name']}: {e}")
            results.append({
                'audio_name': sample['base_name'],
                'reference_text': sample['reference_text'],
                'similarity': -1.0,
            })
    
    # Results
    results_df = pd.DataFrame(results)
    valid_results = results_df[results_df['similarity'] >= 0]
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total: {len(results_df)}, Success: {len(valid_results)}, Failed: {len(results_df) - len(valid_results)}")
    
    if len(valid_results) > 0:
        print(f"\nSpeaker Similarity Statistics:")
        print(f"  Average: {valid_results['similarity'].mean():.4f}")
        print(f"  Median:  {valid_results['similarity'].median():.4f}")
        print(f"  Min:     {valid_results['similarity'].min():.4f}")
        print(f"  Max:     {valid_results['similarity'].max():.4f}")
        print(f"  Std Dev: {valid_results['similarity'].std():.4f}")
    
    # Save
    results_df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to: {output_csv}")


if __name__ == "__main__":
    main()
