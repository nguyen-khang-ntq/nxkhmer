"""
Utility functions for audio processing
"""

import numpy as np
import librosa
from typing import Tuple, Optional


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file and resample if necessary
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio, sr


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC features from audio
    
    Args:
        audio: Audio array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        MFCC features with shape (time, features)
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # Transpose to (time, features)


def align_sequences(seq1: np.ndarray, seq2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two sequences to the same length using interpolation
    
    Args:
        seq1: First sequence
        seq2: Second sequence
        
    Returns:
        Tuple of aligned sequences
    """
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1_aligned = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(seq1)),
            seq1
        )
        seq2_aligned = np.interp(
            np.linspace(0, 1, min_len),
            np.linspace(0, 1, len(seq2)),
            seq2
        )
        return seq1_aligned, seq2_aligned
    return seq1, seq2
