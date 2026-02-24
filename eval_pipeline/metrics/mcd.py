"""
Mel-Cepstral Distortion (MCD) metric calculator
"""

import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from utils.audio import load_audio, extract_mfcc


class MCDCalculator:
    """Calculate Mel-Cepstral Distortion"""
    
    def __init__(self, n_mfcc: int = 13, sample_rate: int = 16000):
        """
        Initialize MCD calculator
        
        Args:
            n_mfcc: Number of MFCC coefficients
            sample_rate: Audio sample rate
        """
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
    
    def calculate(self, generated_audio: str, reference_audio: str) -> float:
        """
        Calculate Mel-Cepstral Distortion (MCD)
        
        Args:
            generated_audio: Path to generated audio
            reference_audio: Path to reference audio
            
        Returns:
            MCD score (lower is better)
        """
        # Load audio
        gen_audio, sr = load_audio(generated_audio, self.sample_rate)
        ref_audio, _ = load_audio(reference_audio, self.sample_rate)
        
        # Extract MFCCs
        gen_mfcc = extract_mfcc(gen_audio, sr, self.n_mfcc)
        ref_mfcc = extract_mfcc(ref_audio, sr, self.n_mfcc)
        
        # Use DTW to align sequences
        distance, path = fastdtw(gen_mfcc, ref_mfcc, dist=euclidean)
        
        # Calculate MCD
        mcd = (10.0 / np.log(10)) * np.sqrt(2 * distance / len(path))
        
        return float(mcd)
