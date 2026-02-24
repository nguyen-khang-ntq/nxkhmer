"""
F0 (Fundamental Frequency) RMSE metric calculator
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
from utils.audio import align_sequences


class F0Calculator:
    """Calculate F0 RMSE between generated and reference audio"""
    
    def __init__(self, f0_min: float = 75, f0_max: float = 600):
        """
        Initialize F0 calculator
        
        Args:
            f0_min: Minimum F0 frequency in Hz
            f0_max: Maximum F0 frequency in Hz
        """
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def extract_f0(self, audio_path: str) -> np.ndarray:
        """
        Extract F0 (fundamental frequency) contour from audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            F0 contour array (only voiced frames)
        """
        snd = parselmouth.Sound(audio_path)
        pitch = call(snd, "To Pitch", 0.0, self.f0_min, self.f0_max)
        
        # Extract pitch values
        f0_values = []
        for i in range(pitch.get_number_of_frames()):
            f0 = pitch.get_value_in_frame(i)
            if f0 > 0:  # Only voiced frames
                f0_values.append(f0)
        
        return np.array(f0_values)
    
    def calculate(self, generated_audio: str, reference_audio: str) -> float:
        """
        Calculate RMSE of F0 between generated and reference audio
        
        Args:
            generated_audio: Path to generated audio
            reference_audio: Path to reference audio
            
        Returns:
            RMSE_F0 score (lower is better)
        """
        gen_f0 = self.extract_f0(generated_audio)
        ref_f0 = self.extract_f0(reference_audio)
        
        # Align lengths
        gen_f0_aligned, ref_f0_aligned = align_sequences(gen_f0, ref_f0)
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((gen_f0_aligned - ref_f0_aligned) ** 2))
        
        return float(rmse)
