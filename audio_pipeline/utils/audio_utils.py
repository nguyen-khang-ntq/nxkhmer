"""
Audio processing utilities
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Union, Tuple


class AudioProcessor:
    """Utility class for audio processing operations"""
    
    @staticmethod
    def load_audio(audio_path: Union[str, Path], sr: int = None, mono: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load audio file
        
        Args:
            audio_path: Path to audio file
            sr: Target sampling rate (None to keep original)
            mono: Convert to mono if True
            
        Returns:
            Tuple of (audio array, sampling rate)
        """
        audio, sample_rate = librosa.load(audio_path, sr=sr, mono=mono)
        return audio, sample_rate
    
    @staticmethod
    def save_audio(audio: np.ndarray, sr: int, output_path: Union[str, Path], 
                   subtype: str = 'PCM_16'):
        """
        Save audio to file
        
        Args:
            audio: Audio array
            sr: Sampling rate
            output_path: Output file path
            subtype: Audio subtype (default: PCM_16)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sr, subtype=subtype)
    
    @staticmethod
    def normalize_volume(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio volume using RMS
        
        Args:
            audio: Input audio
            target_level: Target RMS level in dB
            
        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms == 0:
            return audio
        
        target_rms = 10 ** (target_level / 20.0)
        gain = target_rms / rms
        normalized = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(normalized))
        if max_val > 1.0:
            normalized = normalized / max_val * 0.99
            
        return normalized
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio
        
        Args:
            audio: Input audio
            orig_sr: Original sampling rate
            target_sr: Target sampling rate
            
        Returns:
            Resampled audio
        """
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mono
        
        Args:
            audio: Input audio (can be stereo or multi-channel)
            
        Returns:
            Mono audio
        """
        if audio.ndim > 1:
            return librosa.to_mono(audio)
        return audio
    
    @staticmethod
    def get_duration(audio: np.ndarray, sr: int) -> float:
        """
        Get audio duration in seconds
        
        Args:
            audio: Audio array
            sr: Sampling rate
            
        Returns:
            Duration in seconds
        """
        return len(audio) / sr
