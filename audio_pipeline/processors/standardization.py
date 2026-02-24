"""
Audio Standardization Processor
Converts audio to unified format: 24kHz, mono-channel WAV with volume normalization
"""

import os
from pathlib import Path
from typing import Union
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.audio_utils import AudioProcessor


class AudioStandardizer:
    """Standardize audio format for consistent processing"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: Configuration dictionary (from config.json)
        """
        if config is None:
            config = {
                'target_sample_rate': 24000,
                'target_channels': 1,
                'target_rms_level': -20.0
            }
        
        self.target_sr = config.get('target_sample_rate', 24000)
        self.target_channels = config.get('target_channels', 1)
        self.target_rms_level = config.get('target_rms_level', -20.0)
        
        self.audio_processor = AudioProcessor()
        
    def standardize_audio(self, audio_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Standardize a single audio file
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save standardized audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            audio, sr = self.audio_processor.load_audio(audio_path, sr=None, mono=False)
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = self.audio_processor.to_mono(audio)
            
            # Resample to target sampling rate
            if sr != self.target_sr:
                audio = self.audio_processor.resample(audio, sr, self.target_sr)
            
            # Normalize volume
            audio = self.audio_processor.normalize_volume(audio, self.target_rms_level)
            
            # Save
            self.audio_processor.save_audio(audio, self.target_sr, output_path)
            
            return True
            
        except Exception as e:
            print(f"Error standardizing {audio_path}: {e}")
            return False
    
    def process_folder(self, input_folder: Union[str, Path], output_folder: Union[str, Path],
                      audio_extensions: tuple = ('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        """
        Process all audio files in a folder
        
        Args:
            input_folder: Folder containing input audio files
            output_folder: Folder to save standardized audio
            audio_extensions: Tuple of valid audio file extensions
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(list(input_folder.rglob(f"*{ext}")))
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        success_count = 0
        results = []
        
        for audio_path in tqdm(audio_files, desc="Standardizing audio"):
            # Maintain folder structure
            relative_path = audio_path.relative_to(input_folder)
            output_path = output_folder / relative_path.with_suffix('.wav')
            
            success = self.standardize_audio(audio_path, output_path)
            
            if success:
                success_count += 1
                results.append({
                    'input_file': str(audio_path),
                    'output_file': str(output_path),
                    'status': 'success'
                })
            else:
                results.append({
                    'input_file': str(audio_path),
                    'output_file': str(output_path),
                    'status': 'failed'
                })
        
        # Save metadata
        metadata_path = output_folder / 'standardization_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(audio_files),
                'successful': success_count,
                'failed': len(audio_files) - success_count,
                'config': {
                    'target_sr': self.target_sr,
                    'target_channels': self.target_channels,
                    'target_rms_level': self.target_rms_level
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully standardized {success_count}/{len(audio_files)} files")
        print(f"Output saved to: {output_folder}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize audio files")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for standardized audio")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)['standardization']
    else:
        config = None
    
    standardizer = AudioStandardizer(config=config)
    standardizer.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
