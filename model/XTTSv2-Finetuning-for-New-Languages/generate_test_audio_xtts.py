"""
Script to generate audio from test CSV file using XTTS model.
Reads CSV with columns: audio_file, text, speaker (optional: reference_audio)
Generates audio for each text using reference speaker audio.
"""

import os
import pandas as pd
import torch
import torchaudio
from pathlib import Path
from tqdm import tqdm
from underthesea import sent_tokenize
import argparse
import random
import time
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


class XTTSGenerator:
    def __init__(self, checkpoint_path, config_path, vocab_path, device="cuda"):
        """Initialize XTTS model."""
        self.device = device
        self.sample_rate = 24000
        
        print(f"🖥️ Device: {self.device}")
        print("📥 Loading XTTS model...")
        
        # Load config
        self.config = XttsConfig()
        self.config.load_json(config_path)
        
        # Load model
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(
            self.config, 
            checkpoint_path=checkpoint_path, 
            vocab_path=vocab_path, 
            use_deepspeed=False
        )
        self.model.to(self.device)
        
        print("✅ Model loaded successfully!")
    
    def resample_audio(self, audio_path, target_sr=None):
        """Resample audio to target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (default: self.sample_rate)
        
        Returns:
            Path to resampled audio (temporary file if resampled, original if already correct SR)
        """
        if target_sr is None:
            target_sr = self.sample_rate
        
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Check if resampling is needed
        if sr == target_sr:
            return audio_path
        
        # Resample
        print(f"🔄 Resampling {os.path.basename(audio_path)}: {sr}Hz → {target_sr}Hz")
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_resampled = resampler(audio)
        
        # Save to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            torchaudio.save(tmp_path, audio_resampled, target_sr)
        
        return tmp_path
    
    def generate_from_text(
        self, 
        text: str, 
        speaker_audio_file: str,
        language: str = "km",
        temperature: float = 0.1,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 10,
        top_p: float = 0.3,
        resample_input: bool = True,
    ):
        """Generate speech from text using reference speaker audio."""
        try:
            # Resample audio if needed
            audio_to_use = speaker_audio_file
            temp_file = None
            
            if resample_input:
                audio_to_use = self.resample_audio(speaker_audio_file)
                if audio_to_use != speaker_audio_file:
                    temp_file = audio_to_use
            
            # Get conditioning latents from reference audio
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=audio_to_use,
                gpt_cond_len=self.model.config.gpt_cond_len,
                max_ref_length=self.model.config.max_ref_len,
                sound_norm_refs=self.model.config.sound_norm_refs,
            )
            
            # Split text into sentences
            tts_texts = sent_tokenize(text)
            
            # Generate audio for each sentence
            wav_chunks = []
            for sentence in tts_texts:
                wav_chunk = self.model.inference(
                    text=sentence,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    top_k=top_k,
                    top_p=top_p,
                )
                wav_chunks.append(torch.tensor(wav_chunk["wav"]))
            
            # Concatenate all chunks
            out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0)
            
            # Cleanup temporary resampled file
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            
            return out_wav
            
        except Exception as e:
            # Cleanup temporary file on error
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            raise Exception(f"Error generating audio: {str(e)}")


def get_audio_duration(audio_path):
    """Get audio duration in seconds."""
    try:
        audio, sr = torchaudio.load(audio_path)
        duration = audio.shape[1] / sr
        return duration
    except Exception:
        return None


def find_reference_audio(speaker, train_df, audio_base_dir, min_duration=4.0, max_duration=12.0, selection_strategy='shortest'):
    """Find reference audio for a speaker from training set with duration between 4-12 seconds.
    
    Args:
        speaker: Speaker ID
        train_df: Training dataframe
        audio_base_dir: Base directory for audio files
        min_duration: Minimum audio duration in seconds (default: 4.0)
        max_duration: Maximum audio duration in seconds (default: 12.0)
        selection_strategy: How to select audio ('shortest', 'longest', 'first', 'random')
    
    Returns:
        Path to selected reference audio, or None if no suitable audio found
    """
    speaker_files = train_df[train_df['speaker'] == speaker]['audio_file'].values
    
    if len(speaker_files) == 0:
        return None
    
    # Find audios within duration range
    valid_audios = []
    for audio_file in speaker_files:
        # Convert to full path if needed
        if not os.path.isabs(audio_file):
            audio_path = os.path.join(audio_base_dir, audio_file)
        else:
            audio_path = audio_file
        
        if not os.path.exists(audio_path):
            continue
        
        duration = get_audio_duration(audio_path)
        if duration and min_duration <= duration <= max_duration:
            valid_audios.append((audio_path, duration))
    
    if len(valid_audios) == 0:
        return None
    
    # Select based on strategy
    if selection_strategy == 'longest':
        # Select the longest audio within range
        selected_audio = max(valid_audios, key=lambda x: x[1])[0]
    elif selection_strategy == 'shortest':
        # Select the shortest audio within range
        selected_audio = min(valid_audios, key=lambda x: x[1])[0]
    elif selection_strategy == 'random':
        # Random selection
        selected_audio = random.choice(valid_audios)[0]
    else:  # 'first'
        # Use first valid audio
        selected_audio = valid_audios[0][0]
    
    return selected_audio


def main():
    parser = argparse.ArgumentParser(description="Generate audio from test CSV using XTTS")
    parser.add_argument("--root_path", type=str, default="",
                        help="Root directory path for all data files")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test CSV file")
    parser.add_argument("--train_csv", type=str, default=None, 
                        help="Path to train CSV (for finding reference audio by speaker)")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Output directory for generated audio")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to XTTS checkpoint (.pth file)")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to XTTS config.json")
    parser.add_argument("--vocab_path", type=str, required=True,
                        help="Path to vocab.json")
    parser.add_argument("--audio_base_dir", type=str, default="",
                        help="Base directory for audio files (if paths in CSV are relative)")
    parser.add_argument("--language", type=str, default="km", 
                        help="Language code (e.g., 'km', 'vi', 'en')")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    # Reference audio selection parameters
    parser.add_argument("--min_ref_duration", type=float, default=4.0,
                        help="Minimum reference audio duration in seconds")
    parser.add_argument("--max_ref_duration", type=float, default=12.0,
                        help="Maximum reference audio duration in seconds")
    parser.add_argument("--ref_selection_strategy", type=str, default="shortest",
                        choices=['shortest', 'longest', 'first', 'random'],
                        help="Strategy to select reference audio (shortest/longest/first/random)")
    parser.add_argument("--resample_input", action="store_true", default=True,
                        help="Resample reference audio to 24kHz (default: True)")
    parser.add_argument("--no_resample_input", action="store_false", dest="resample_input",
                        help="Disable resampling of reference audio")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=10.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.3)
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Apply root path if provided
    test_csv_path = args.test_csv
    if args.root_path and not os.path.isabs(test_csv_path):
        test_csv_path = os.path.join(args.root_path, test_csv_path)
    
    train_csv_path = args.train_csv
    if args.train_csv and args.root_path and not os.path.isabs(train_csv_path):
        train_csv_path = os.path.join(args.root_path, train_csv_path)
    
    audio_base_dir = args.audio_base_dir
    if args.root_path and audio_base_dir and not os.path.isabs(audio_base_dir):
        audio_base_dir = os.path.join(args.root_path, audio_base_dir)
    elif args.root_path and not audio_base_dir:
        audio_base_dir = args.root_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test CSV
    print(f"📄 Loading test CSV from: {test_csv_path}")
    test_df = pd.read_csv(test_csv_path)
    print(f"Found {len(test_df)} test samples")
    
    # Load train CSV if provided
    train_df = None
    if train_csv_path and os.path.exists(train_csv_path):
        print(f"📄 Loading train CSV from: {train_csv_path}")
        train_df = pd.read_csv(train_csv_path)
        print(f"Found {len(train_df)} training samples")
    
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        print(f"⚠️ Limiting to {args.max_samples} test samples")
    
    # Pre-compute reference audio for each speaker (only once per speaker)
    speaker_ref_audio = {}
    if train_df is not None and 'speaker' in test_df.columns:
        unique_speakers = test_df['speaker'].dropna().unique()
        print(f"\n🔍 Finding reference audio for {len(unique_speakers)} unique speakers...")
        
        for speaker in tqdm(unique_speakers, desc="Finding references"):
            ref_audio = find_reference_audio(
                speaker, 
                train_df, 
                audio_base_dir,
                min_duration=args.min_ref_duration,
                max_duration=args.max_ref_duration,
                selection_strategy=args.ref_selection_strategy
            )
            if ref_audio:
                speaker_ref_audio[speaker] = ref_audio
            else:
                print(f"\n⚠️ No reference audio (4-12s) found for speaker: {speaker}")
        
        print(f"✅ Found reference audio for {len(speaker_ref_audio)}/{len(unique_speakers)} speakers")
    
    # Initialize generator
    generator = XTTSGenerator(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        vocab_path=args.vocab_path,
        device=args.device
    )
    
    successful = 0
    failed = 0
    errors = []
    results = []
    
    print(f"\n🎵 Generating audio for {len(test_df)} test samples...")
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
        start_time = time.time()
        try:
            audio_file = row['audio_file']
            text = row['text']
            
            # Determine reference audio
            if 'reference_audio' in row and pd.notna(row['reference_audio']):
                # Use specified reference audio
                ref_audio = row['reference_audio']
                if not os.path.isabs(ref_audio):
                    ref_audio = os.path.join(audio_base_dir, ref_audio)
            elif 'speaker' in row and pd.notna(row['speaker']):
                # Use pre-computed reference audio for this speaker
                speaker = row['speaker']
                ref_audio = speaker_ref_audio.get(speaker)
                if ref_audio is None:
                    raise ValueError(f"No reference audio found for speaker: {speaker}")
            else:
                raise ValueError("No reference audio or speaker information provided")
            
            if not os.path.exists(ref_audio):
                raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
            
            # Generate audio
            wav = generator.generate_from_text(
                text=text,
                speaker_audio_file=ref_audio,
                language=args.language,
                temperature=args.temperature,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                top_k=args.top_k,
                top_p=args.top_p,
                resample_input=args.resample_input,
            )
            
            # Create output filename
            output_filename = Path(audio_file).stem + "_generated.wav"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Save audio
            torchaudio.save(output_path, wav.cpu(), generator.sample_rate)
            
            # Calculate RTF
            processing_time = time.time() - start_time
            audio_duration = wav.shape[1] / generator.sample_rate
            rtf = processing_time / audio_duration if audio_duration > 0 else 0
            
            results.append({
                'audio_file': audio_file,
                'status': 'success',
                'output_path': output_path,
                'processing_time': round(processing_time, 3),
                'audio_duration': round(audio_duration, 3),
                'rtf': round(rtf, 3)
            })
            
            successful += 1
            
        except Exception as e:
            error_msg = f"{audio_file}: {str(e)}"
            errors.append(error_msg)
            results.append({
                'audio_file': audio_file,
                'status': 'failed',
                'error': str(e)
            })
            print(f"\n❌ Error: {error_msg}")
            failed += 1
    
    # Calculate RTF statistics
    rtf_values = []
    total_audio_duration = 0
    total_processing_time = 0
    
    for result in results:
        if result['status'] == 'success' and 'rtf' in result and result['rtf'] > 0:
            rtf_values.append(result['rtf'])
        if 'audio_duration' in result:
            total_audio_duration += result['audio_duration']
        if 'processing_time' in result:
            total_processing_time += result['processing_time']
    
    # Save results log
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.output_dir, "generation_results.csv")
    results_df.to_csv(results_csv, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print(f"✅ Generation complete!")
    print(f"Total test samples: {len(test_df)}")
    print(f"Successfully generated: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")
    print(f"Results log: {results_csv}")
    
    # RTF Statistics
    if rtf_values:
        avg_rtf = sum(rtf_values) / len(rtf_values)
        min_rtf = min(rtf_values)
        max_rtf = max(rtf_values)
        print(f"\n📊 RTF Statistics:")
        print(f"Average RTF: {avg_rtf:.3f}")
        print(f"Min RTF: {min_rtf:.3f}")
        print(f"Max RTF: {max_rtf:.3f}")
        print(f"Total audio duration: {total_audio_duration:.2f}s")
        print(f"Total processing time: {total_processing_time:.2f}s")
        if total_audio_duration > 0:
            overall_rtf = total_processing_time / total_audio_duration
            print(f"Overall RTF: {overall_rtf:.3f}")
    
    if errors:
        print(f"\n❌ Failed samples (showing first 10):")
        for err in errors[:10]:
            print(f"  {err}")
    
    print("="*80)


if __name__ == "__main__":
    main()
