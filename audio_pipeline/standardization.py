"""
Audio Standardization Module
Converts audio to unified format: 24kHz, mono-channel WAV with volume normalization
"""

import os
from pathlib import Path
from typing import Union
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


class AudioStandardizer:
    """Standardize audio format for consistent processing"""
    
    def __init__(self, target_sr: int = 24000, target_channels: int = 1):
        """
        Args:
            target_sr: Target sampling rate (default: 24000 Hz)
            target_channels: Target number of channels (default: 1 for mono)
        """
        self.target_sr = target_sr
        self.target_channels = target_channels
        
    def normalize_volume(self, audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio volume using RMS-based normalization
        
        Args:
            audio: Input audio array
            target_level: Target RMS level in dB (default: -20.0)
            
        Returns:
            Volume-normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms == 0:
            return audio
        
        # Calculate target RMS from dB
        target_rms = 10 ** (target_level / 20.0)
        
        # Apply gain
        gain = target_rms / rms
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.99
            
        return normalized
    
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
            audio, sr = librosa.load(audio_path, sr=None, mono=False)
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
            
            # Resample to target sampling rate
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            
            # Normalize volume
            audio = self.normalize_volume(audio)
            
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save as WAV
            sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
            
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
        for audio_path in tqdm(audio_files, desc="Standardizing audio"):
            # Maintain folder structure
            relative_path = audio_path.relative_to(input_folder)
            output_path = output_folder / relative_path.with_suffix('.wav')
            
            if self.standardize_audio(audio_path, output_path):
                success_count += 1
        
        print(f"Successfully standardized {success_count}/{len(audio_files)} files")
        print(f"Output saved to: {output_folder}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standardize audio files")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for standardized audio")
    parser.add_argument("--sr", type=int, default=24000, help="Target sampling rate (default: 24000)")
    
    args = parser.parse_args()
    
    standardizer = AudioStandardizer(target_sr=args.sr)
    standardizer.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
