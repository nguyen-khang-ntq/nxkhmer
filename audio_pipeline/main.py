"""
Main Audio Processing Pipeline
Sequential pipeline: Standardization → Diarization → Quality Filtering → Transcription
"""

import os
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from processors.standardization import AudioStandardizer
from processors.diarization import SpeakerDiarizer
from processors.quality_filter import QualityFilter
from processors.transcription import Transcriber
from processors.sidon_cleaner import SidonCleaner


class AudioPipeline:
    """Complete audio processing pipeline for TTS data preparation"""
    
    def __init__(self, config_path: str = None, device: str = None):
        """
        Initialize pipeline with configuration
        
        Args:
            config_path: Path to config.json
            device: Device for GPU-based models ('cuda' or 'cpu')
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = device
        
        print("="*70)
        print("Audio Processing Pipeline for TTS Data")
        print("="*70)
        print(f"Pipeline: {self.config['pipeline_config']['name']}")
        print(f"Version: {self.config['pipeline_config']['version']}")
        print("="*70)
    
    def run(self, input_folder: str, output_base_folder: str, 
            skip_steps: list = None, process_per_file: bool = True):
        """
        Run the complete pipeline
        
        Args:
            input_folder: Folder containing raw audio files
            output_base_folder: Base folder for all pipeline outputs
            skip_steps: List of steps to skip ['standardization', 'diarization', 'quality_filter', 'transcription', 'sidon']
            process_per_file: If True, process each file through all steps before moving to next file (more efficient)
        """
        if skip_steps is None:
            skip_steps = []
        
        input_folder = Path(input_folder)
        output_base_folder = Path(output_base_folder)
        output_base_folder.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped run folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = output_base_folder / f"run_{timestamp}"
        run_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"\nInput: {input_folder}")
        print(f"Output: {run_folder}")
        print(f"Processing mode: {'Per-file (optimized)' if process_per_file else 'Batch'}")
        print()
        
        if process_per_file:
            self._run_per_file(input_folder, run_folder, skip_steps, timestamp)
        else:
            self._run_batch(input_folder, run_folder, skip_steps, timestamp)
    
    def _run_per_file(self, input_folder: Path, run_folder: Path, skip_steps: list, timestamp: str):
        """Process each audio file through all pipeline steps sequentially"""
        
        # Create temp folders for intermediate processing (will be deleted)
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="audio_pipeline_")
        temp_standardized = Path(temp_dir) / "standardized"
        temp_diarized = Path(temp_dir) / "diarized"
        
        for folder in [temp_standardized, temp_diarized]:
            folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize processors once (expensive operations)
        print("Initializing pipeline processors...")
        standardizer = AudioStandardizer(config=self.config['standardization']) if 'standardization' not in skip_steps else None
        diarizer = SpeakerDiarizer(config=self.config['diarization'], device=self.device) if 'diarization' not in skip_steps else None
        
        # Initialize processors once (expensive operations)
        print("Initializing pipeline processors...")
        standardizer = AudioStandardizer(config=self.config['standardization']) if 'standardization' not in skip_steps else None
        diarizer = SpeakerDiarizer(config=self.config['diarization'], device=self.device) if 'diarization' not in skip_steps else None
        
        # Initialize quality filter (may be disabled if DNSMOS unavailable)
        quality_filter = None
        if 'quality_filter' not in skip_steps:
            quality_filter = QualityFilter(config=self.config['quality_filtering'])
            if not quality_filter.enabled:
                print("  → Quality filtering will be skipped (DNSMOS unavailable)")
                quality_filter = None
        
        transcriber = Transcriber(config=self.config['asr'], device=self.device) if 'transcription' not in skip_steps else None
        
        # Initialize Sidon cleaner (last step)
        sidon_cleaner = None
        if 'sidon' not in skip_steps:
            sidon_cleaner = SidonCleaner(config=self.config['sidon'], device=self.device)
            if not sidon_cleaner.enabled:
                print("  → Sidon cleaning will be skipped (disabled in config)")
                sidon_cleaner = None
        
        print("✓ All processors initialized\n")
        
        # Find all audio files
        audio_extensions = tuple(self.config['standardization'].get('supported_formats', ['.wav', '.mp3', '.flac', '.m4a', '.ogg']))
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(input_folder.rglob(f"*{ext}")))
        
        print(f"Found {len(audio_files)} audio files to process\n")
        print("="*70)
        
        # Process each file through entire pipeline
        all_transcriptions = []
        segments_processed = 0
        segments_filtered = 0
        start_time = time.time()
        
        for idx, audio_path in enumerate(audio_files, 1):
            print(f"\n[{idx}/{len(audio_files)}] Processing: {audio_path.name}")
            print("-"*70)
            file_start = time.time()
            prev_filtered = segments_filtered
            
            try:
                # Create separate folder for this input file
                file_output_folder = run_folder / audio_path.stem
                file_audio_folder = file_output_folder / "audio"
                file_transcription_folder = file_output_folder / "transcriptions"
                file_audio_folder.mkdir(parents=True, exist_ok=True)
                file_transcription_folder.mkdir(parents=True, exist_ok=True)
                
                current_path = audio_path
                
                # Step 1: Standardization
                if standardizer:
                    print("  [1/4] Standardizing audio...", end=" ", flush=True)
                    relative_path = audio_path.relative_to(input_folder)
                    standardized_path = temp_standardized / relative_path.with_suffix('.wav')
                    if standardizer.standardize_audio(current_path, standardized_path):
                        print("✓")
                        current_path = standardized_path
                    else:
                        print("✗ Failed")
                        continue
                
                # Step 2: Diarization (extracts multiple segments from one file)
                if diarizer:
                    print("  [2/4] Diarizing (extracting speaker segments)...", flush=True)
                    segments = diarizer.process_file(current_path, temp_diarized, verbose=True)
                    if segments:
                        segment_paths = [Path(seg['segment_file']) for seg in segments]
                        print(f"        ✓ {len(segments)} segments extracted")
                    else:
                        print("        ✗ No valid segments")
                        continue
                else:
                    segment_paths = [current_path]
                
                # Step 3 & 4: Quality filtering and transcription for each segment
                print(f"  [3/4] Quality filtering & dual ASR transcription ({len(segment_paths)} segments)...")
                
                file_transcriptions = []
                file_segment_count = 0
                cer_filtered = 0
                
                # Add tqdm for segments within this file
                for seg_path in tqdm(segment_paths, desc="    Segments", unit="seg", leave=False):
                    segments_processed += 1
                    
                    # Quality filtering - get MOS scores
                    mos_scores = None
                    if quality_filter:
                        mos_scores = quality_filter.evaluate_audio(seg_path)
                        if not quality_filter.passes_quality_filter(mos_scores):
                            segments_filtered += 1
                            continue
                    
                    # Dual ASR Transcription with CER alignment
                    if transcriber:
                        result = transcriber.transcribe_audio(seg_path)
                        if result and result.get('text', '').strip():
                            # CER check passed - both models agree
                            file_segment_count += 1
                            # Copy audio to this file's output folder
                            final_audio_name = f"seg{file_segment_count:04d}.wav"
                            final_audio_path = file_audio_folder / final_audio_name
                            
                            import shutil
                            shutil.copy2(seg_path, final_audio_path)
                            
                            # Update result with final path and MOS scores
                            result['audio_path'] = str(final_audio_path)
                            if mos_scores:
                                result['mos_ovrl'] = mos_scores.get('OVRL', mos_scores.get('ovrl', 0))
                                result['mos_sig'] = mos_scores.get('SIG', mos_scores.get('sig', 0))
                                result['mos_bak'] = mos_scores.get('BAK', mos_scores.get('bak', 0))
                            
                            file_transcriptions.append(result)
                            all_transcriptions.append(result)
                        else:
                            # CER too high or transcription failed
                            cer_filtered += 1
                
                # Step 5: Sidon audio cleaning (final step)
                if sidon_cleaner and file_transcriptions:
                    print(f"  [4/4] Cleaning audio with Sidon ({len(file_transcriptions)} segments)...")
                    cleaned_audio_folder = file_output_folder / "audio_cleaned"
                    cleaned_audio_folder.mkdir(parents=True, exist_ok=True)
                    
                    cleaned_count = 0
                    for transcription in tqdm(file_transcriptions, desc="    Cleaning", unit="file", leave=False):
                        audio_path = Path(transcription['audio_path'])
                        cleaned_path = cleaned_audio_folder / audio_path.name
                        
                        if sidon_cleaner.clean_audio(audio_path, cleaned_path):
                            # Update transcription to point to cleaned audio
                            transcription['audio_path'] = str(cleaned_path)
                            transcription['audio_path_original'] = str(audio_path)
                            cleaned_count += 1
                    
                    print(f"        ✓ Cleaned {cleaned_count}/{len(file_transcriptions)} audio files")
                
                # Save transcriptions for this file in its own folder
                if file_transcriptions and transcriber:
                    print(f"  [5/5] Saving transcriptions...", end=" ", flush=True)
                    transcriber.save_dataset(file_transcriptions, file_transcription_folder,
                                           csv_filename="dataset.csv",
                                           json_filename="transcriptions.json")
                
                # Summary for this file
                file_elapsed = time.time() - file_start
                file_filtered = segments_filtered - prev_filtered
                print(f"  ✓ Complete: {len(file_transcriptions)} transcriptions, "
                      f"{file_filtered} quality filtered, {cer_filtered} CER filtered ({file_elapsed:.1f}s)")
                print(f"  ✓ Output saved to: {file_output_folder}")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        
        # Save combined dataset at run folder root
        if all_transcriptions and transcriber:
            print("\n" + "="*70)
            print("Saving combined dataset...")
            combined_folder = run_folder / "combined"
            combined_folder.mkdir(parents=True, exist_ok=True)
            transcriber.save_dataset(all_transcriptions, combined_folder)
        
        # Print final summary
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"Total files processed: {len(audio_files)}")
        print(f"Total segments extracted: {segments_processed}")
        print(f"Segments filtered out: {segments_filtered}")
        print(f"Transcriptions created: {len(all_transcriptions)}")
        print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
        print(f"Average time per file: {elapsed_time/len(audio_files):.1f}s")
        print("="*70)
        
        # Save metadata
        self._save_pipeline_metadata(run_folder, input_folder, skip_steps, timestamp, 
                                     mode='per_file', total_files=len(audio_files))
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"✓ Processed {len(audio_files)} audio files")
        print(f"✓ Generated {len(all_transcriptions)} transcribed segments")
        print(f"✓ All outputs saved to: {run_folder}")
    
    def _run_batch(self, input_folder: Path, run_folder: Path, skip_steps: list, timestamp: str):
        """Process all files through each step before moving to next step (original batch mode)"""
        
    def _run_batch(self, input_folder: Path, run_folder: Path, skip_steps: list, timestamp: str):
        """Process all files through each step before moving to next step (original batch mode)"""
        
        # Step 1: Standardization
        if 'standardization' not in skip_steps:
            print("\n" + "="*70)
            print("STEP 1: Audio Standardization")
            print("="*70)
            
            standardized_folder = run_folder / "01_standardized"
            standardizer = AudioStandardizer(config=self.config['standardization'])
            standardizer.process_folder(input_folder, standardized_folder)
            
            current_folder = standardized_folder
        else:
            print("\n⏭ Skipping standardization")
            current_folder = input_folder
        
        # Step 2: Diarization
        if 'diarization' not in skip_steps:
            print("\n" + "="*70)
            print("STEP 2: Speaker Diarization")
            print("="*70)
            
            diarized_folder = run_folder / "02_diarized"
            diarizer = SpeakerDiarizer(config=self.config['diarization'], device=self.device)
            diarizer.process_folder(current_folder, diarized_folder)
            diarized_folder = run_folder / "02_diarized"
            diarizer = SpeakerDiarizer(config=self.config['diarization'], device=self.device)
            diarizer.process_folder(current_folder, diarized_folder)
            
            current_folder = diarized_folder
        else:
            print("\n⏭ Skipping diarization")
        
        # Step 3: Quality Filtering
        if 'quality_filter' not in skip_steps:
            print("\n" + "="*70)
            print("STEP 3: Audio Quality Filtering")
            print("="*70)
            
            filtered_folder = run_folder / "03_filtered"
            quality_filter = QualityFilter(config=self.config['quality_filtering'])
            quality_filter.process_folder(current_folder, filtered_folder)
            
            current_folder = filtered_folder
        else:
            print("\n⏭ Skipping quality filtering")
        
        # Step 4: Transcription
        if 'transcription' not in skip_steps:
            print("\n" + "="*70)
            print("STEP 4: ASR Transcription")
            print("="*70)
            
            transcription_folder = run_folder / "04_transcribed"
            transcriber = Transcriber(config=self.config['asr'], device=self.device)
            transcriber.process_folder(current_folder, transcription_folder)
            
            current_folder = transcription_folder
        else:
            print("\n⏭ Skipping transcription")
        
        # Step 5: Sidon Audio Cleaning (Final Step)
        if 'sidon' not in skip_steps:
            print("\n" + "="*70)
            print("STEP 5: Sidon Audio Cleaning & Resampling")
            print("="*70)
            
            cleaned_folder = run_folder / "05_cleaned"
            sidon_cleaner = SidonCleaner(config=self.config['sidon'], device=self.device)
            if sidon_cleaner.enabled:
                sidon_cleaner.process_folder(current_folder, cleaned_folder)
                current_folder = cleaned_folder
            else:
                print("  Sidon cleaning disabled in config, skipping...")
        else:
            print("\n⏭ Skipping Sidon cleaning")
        
        # Save metadata
        self._save_pipeline_metadata(run_folder, input_folder, skip_steps, timestamp, mode='batch')
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"✓ All outputs saved to: {run_folder}")
    
    def _save_pipeline_metadata(self, run_folder: Path, input_folder: Path, 
                                skip_steps: list, timestamp: str, mode: str = 'per_file',
                                total_files: int = 0):
        """Save pipeline execution metadata"""
        pipeline_metadata = {
            'timestamp': timestamp,
            'processing_mode': mode,
            'input_folder': str(input_folder),
            'output_folder': str(run_folder),
            'total_input_files': total_files,
            'config': self.config,
            'skipped_steps': skip_steps,
            'device': self.device
        }
        
        metadata_path = run_folder / "pipeline_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pipeline_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Pipeline metadata: {metadata_path}")
        
        # Show final dataset location
        final_csv = run_folder / "04_transcribed" / "dataset.csv"
        if final_csv.exists():
            print(f"✓ Final TTS dataset: {final_csv}")
        
        print("\n" + "="*70)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Audio Processing Pipeline for TTS Data Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Run full pipeline
  python main.py --input /path/to/audio --output /path/to/output
  
  # Skip certain steps (if already processed)
  python main.py --input /path/to/audio --output /path/to/output --skip standardization diarization
  
  # Specify device
  python main.py --input /path/to/audio --output /path/to/output --device cuda
        """
    )
    
    parser.add_argument("--input", type=str, required=True,
                       help="Input folder containing raw audio files")
    parser.add_argument("--output", type=str, required=True,
                       help="Output base folder for pipeline results")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config.json (default: ./config.json)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    parser.add_argument("--skip", nargs='+', default=[],
                       choices=['standardization', 'diarization', 'quality_filter', 'transcription', 'sidon'],
                       help="Steps to skip")
    parser.add_argument("--batch", action='store_true',
                       help="Use batch processing mode instead of per-file (less memory efficient)")
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = AudioPipeline(config_path=args.config, device=args.device)
    pipeline.run(args.input, args.output, skip_steps=args.skip, process_per_file=not args.batch)


if __name__ == "__main__":
    main()
