"""
Run all tests for TTS Evaluation Pipeline
Consolidated version - all tests in one file
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
import tempfile
import torch.nn.functional as F


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


def test_cer(matched_samples, output_dir):
    """Test CER calculation"""
    print("\n" + "=" * 80)
    print("TEST 1: CER (Character Error Rate)")
    print("=" * 80)
    
    try:
        if len(matched_samples) == 0:
            print("⚠ No matching samples found!")
            return False
        
        # Load ASR model
        print("Loading ASR Model...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import ASRFactory
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        model_card = "omniASR_LLM_7B"
        asr_model = ASRFactory.create_omni_asr(
            model_card=model_card,
            device=device
        )
        print("✓ ASR model loaded")
        
        # Calculate CER
        print("Calculating CER...")
        results = []
        for sample in tqdm(matched_samples, desc="Processing"):  # Only first 10 for testing
            try:
                waveform, sr = torchaudio.load(sample['generated_audio'])
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    torchaudio.save(tmp_path, waveform, 16000)
                
                result = asr_model(tmp_path)
                transcription = result['text']
                os.remove(tmp_path)
                
                cer = calculate_cer_simple(transcription, sample['reference_text'])
                
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'transcription': transcription,
                    'cer': cer
                })
                
            except Exception as e:
                print(f"Error: {sample['base_name']}: {e}")
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'transcription': 'ERROR',
                    'cer': -1.0
                })
        
        # Results
        results_df = pd.DataFrame(results)
        valid_results = results_df[results_df['cer'] >= 0]
        
        print(f"\nResults - Total: {len(results_df)}, Success: {len(valid_results)}")
        if len(valid_results) > 0:
            print(f"Average CER: {valid_results['cer'].mean():.4f}")
            print(f"Median CER: {valid_results['cer'].median():.4f}")
        
        # Save
        output_csv = os.path.join(output_dir, "cer_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}")
        
        return True
        
    except Exception as e:
        print(f"✗ CER Test Failed: {e}")
        return False


def test_mos(matched_samples, output_dir):
    """Test MOS calculation"""
    print("\n" + "=" * 80)
    print("TEST 2: MOS (Mean Opinion Score)")
    print("=" * 80)
    
    try:
        model_path = "/home/coder/data/Speech/Data/Amphion/preprocessors/Emilia/pretrained_models/dnsmos/sig_bak_ovr.onnx"
        
        if not os.path.exists(model_path):
            print(f"⚠ DNSMOS model not found at: {model_path}")
            return False
        
        if len(matched_samples) == 0:
            print("⚠ No matching samples found!")
            return False
        
        # Load DNSMOS
        print("Loading DNSMOS Calculator...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from metrics.dnsmos import DNSMOSCalculator
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dnsmos_calculator = DNSMOSCalculator(
            model_path=model_path,
            device=device,
            is_personalized_MOS=False
        )
        print("✓ DNSMOS calculator loaded")
        
        # Calculate MOS
        print("Calculating MOS...")
        results = []
        for sample in tqdm(matched_samples, desc="Processing"):  # Only first 10 for testing
            try:
                waveform, sr = torchaudio.load(sample['generated_audio'])
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    waveform = resampler(waveform)
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    torchaudio.save(tmp_path, waveform, 16000)
                
                mos_scores = dnsmos_calculator.calculate(tmp_path)
                os.remove(tmp_path)
                
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'ovrl_mos': mos_scores['OVRL'],
                    'sig_mos': mos_scores['SIG'],
                    'bak_mos': mos_scores['BAK'],
                })
                
            except Exception as e:
                print(f"Error: {sample['base_name']}: {e}")
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'ovrl_mos': -1.0,
                    'sig_mos': -1.0,
                    'bak_mos': -1.0,
                })
        
        # Results
        results_df = pd.DataFrame(results)
        valid_results = results_df[results_df['ovrl_mos'] >= 0]
        
        print(f"\nResults - Total: {len(results_df)}, Success: {len(valid_results)}")
        if len(valid_results) > 0:
            print(f"Average Overall MOS: {valid_results['ovrl_mos'].mean():.4f}")
            print(f"Average Signal MOS: {valid_results['sig_mos'].mean():.4f}")
        
        # Save
        output_csv = os.path.join(output_dir, "mos_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}")
        
        return True
        
    except Exception as e:
        print(f"✗ MOS Test Failed: {e}")
        return False


def test_similarity(matched_samples, output_dir):
    """Test Speaker Similarity calculation"""
    print("\n" + "=" * 80)
    print("TEST 3: Speaker Similarity")
    print("=" * 80)
    
    try:
        # Filter samples that have reference audio
        samples_with_ref = [s for s in matched_samples if 'reference_audio' in s and os.path.exists(s['reference_audio'])]
        
        print(f"Samples with reference audio: {len(samples_with_ref)}")
        
        if len(samples_with_ref) == 0:
            print("⚠ No samples with reference audio found!")
            return False
        
        # Load ECAPA-TDNN
        print("Loading ECAPA-TDNN Speaker Encoder...")
        from speechbrain.inference.speaker import EncoderClassifier
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        print("✓ ECAPA-TDNN loaded")
        
        # Calculate Similarity
        print("Calculating Speaker Similarity...")
        results = []
        for sample in tqdm(samples_with_ref, desc="Processing"):  # Only first 10 for testing
            try:
                wav1, sr1 = torchaudio.load(sample['generated_audio'])
                wav2, sr2 = torchaudio.load(sample['reference_audio'])
                
                if sr1 != 16000:
                    resampler1 = torchaudio.transforms.Resample(orig_freq=sr1, new_freq=16000)
                    wav1 = resampler1(wav1)
                if sr2 != 16000:
                    resampler2 = torchaudio.transforms.Resample(orig_freq=sr2, new_freq=16000)
                    wav2 = resampler2(wav2)
                
                emb1 = classifier.encode_batch(wav1)
                emb2 = classifier.encode_batch(wav2)
                
                similarity = F.cosine_similarity(emb1, emb2, dim=-1).item()
                
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'similarity': similarity,
                })
                
            except Exception as e:
                print(f"Error: {sample['base_name']}: {e}")
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'similarity': -1.0,
                })
        
        # Results
        results_df = pd.DataFrame(results)
        valid_results = results_df[results_df['similarity'] >= 0]
        
        print(f"\nResults - Total: {len(results_df)}, Success: {len(valid_results)}")
        if len(valid_results) > 0:
            print(f"Average Similarity: {valid_results['similarity'].mean():.4f}")
            print(f"Median Similarity: {valid_results['similarity'].median():.4f}")
        
        # Save
        output_csv = os.path.join(output_dir, "similarity_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}")
        
        return True
        
    except Exception as e:
        print(f"✗ Similarity Test Failed: {e}")
        return False


def test_mcd(matched_samples, output_dir):
    """Test MCD calculation"""
    print("\n" + "=" * 80)
    print("TEST 4: MCD (Mel-Cepstral Distortion)")
    print("=" * 80)
    
    try:
        from pymcd.mcd import Calculate_MCD
        
        # Filter samples that have reference audio
        samples_with_ref = [s for s in matched_samples if 'reference_audio' in s and os.path.exists(s['reference_audio'])]
        
        print(f"Samples with reference audio: {len(samples_with_ref)}")
        
        if len(samples_with_ref) == 0:
            print("⚠ No samples with reference audio found!")
            return False
        
        # Initialize MCD calculator
        print("Initializing MCD Calculator...")
        mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
        print("✓ MCD calculator initialized")
        
        # Calculate MCD
        print("Calculating MCD...")
        results = []
        for sample in tqdm(samples_with_ref, desc="Processing"):  # Only first 10 for testing
            try:
                mcd_score = mcd_toolbox.calculate_mcd(
                    sample['reference_audio'],
                    sample['generated_audio']
                )
                
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'mcd': mcd_score,
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"Error: {sample['base_name']}: {e}")
                results.append({
                    'audio_name': sample['base_name'],
                    'reference_text': sample['reference_text'],
                    'mcd': -1.0,
                    'status': f'error: {str(e)}'
                })
        
        # Results
        results_df = pd.DataFrame(results)
        valid_results = results_df[results_df['mcd'] >= 0]
        
        print(f"\nResults - Total: {len(results_df)}, Success: {len(valid_results)}")
        if len(valid_results) > 0:
            print(f"Average MCD: {valid_results['mcd'].mean():.4f}")
            print(f"Median MCD: {valid_results['mcd'].median():.4f}")
        
        # Save
        output_csv = os.path.join(output_dir, "mcd_results.csv")
        results_df.to_csv(output_csv, index=False)
        print(f"✓ Saved to: {output_csv}")
        
        return True
        
    except Exception as e:
        print(f"✗ MCD Test Failed: {e}")
        return False


def main():
    print("=" * 80)
    print("TTS EVALUATION PIPELINE - CONSOLIDATED TEST SUITE")
    print("=" * 80)
    
    # Configuration
    csv_path = "/home/coder/datasets/khmer_audio_datasets/fleurs_km_export/test.csv"
    generated_audio_dir = "/home/coder/data/Speech/TTS/Orpheus/generated_final_fleurs"
    reference_audio_base = "/home/coder/datasets/khmer_audio_datasets/fleurs_km_export/"
    output_dir = "/home/coder/data/Interspeech/eval_pipeline/results"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  CSV: {csv_path}")
    print(f"  Generated Audio: {generated_audio_dir}")
    print(f"  Reference Audio: {reference_audio_base}")
    print(f"  Output Directory: {output_dir}")
    
    
    # Load CSV once
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples in CSV: {len(df)}")
    
    # Get generated audio files
    print(f"\nScanning generated audio directory: {generated_audio_dir}")
    generated_files = {}
    if os.path.exists(generated_audio_dir):
        for audio_file in os.listdir(generated_audio_dir):
            if audio_file.endswith("_generated.wav"):
                base_name = audio_file.replace("_generated.wav", "")
                generated_files[base_name] = os.path.join(generated_audio_dir, audio_file)
    
    print(f"Found {len(generated_files)} generated audio files")
    
    # Match samples once
    print(f"\nMatching CSV entries with generated audio...")
    matched_samples = []
    for idx, row in df.iterrows():
        audio_path = row['audio_file']
        text = row['text']
        base_name = Path(audio_path).stem
        
        if base_name in generated_files:
            # Construct reference audio path
            if not os.path.isabs(audio_path):
                reference_audio = os.path.join(reference_audio_base, audio_path)
            else:
                reference_audio = audio_path
            
            matched_samples.append({
                'base_name': base_name,
                'generated_audio': generated_files[base_name],
                'reference_audio': reference_audio,
                'reference_text': text,
            })
    
    print(f"Matched {len(matched_samples)} samples")
    
    if len(matched_samples) == 0:
        print("\n⚠ No matching samples found! Exiting...")
        return
    
    # Run tests with shared data
    results = []
    
    results.append(("CER (Character Error Rate)", test_cer(matched_samples, output_dir)))
    results.append(("MOS (Mean Opinion Score)", test_mos(matched_samples, output_dir)))
    results.append(("Speaker Similarity", test_similarity(matched_samples, output_dir)))
    results.append(("MCD (Mel-Cepstral Distortion)", test_mcd(matched_samples, output_dir)))
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:40} {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{len(results)} test suites passed")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
