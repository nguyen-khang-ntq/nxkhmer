"""
Synthetic Mean Opinion Score (SMOS) metric calculator
"""

from typing import Optional


class SMOSCalculator:
    """Calculate Synthetic MOS combining multiple metrics"""
    
    def __init__(self, mos_calculator, similarity_calculator=None):
        """
        Initialize SMOS calculator
        
        Args:
            mos_calculator: MOS calculator instance
            similarity_calculator: Optional similarity calculator instance
        """
        self.mos_calculator = mos_calculator
        self.similarity_calculator = similarity_calculator
    
    def calculate(
        self, 
        audio_path: str, 
        reference_audio: Optional[str] = None,
        mos_weight: float = 0.7,
        sim_weight: float = 0.3
    ) -> float:
        """
        Calculate Synthetic MOS (SMOS) - combination of objective metrics
        
        Args:
            audio_path: Path to audio file
            reference_audio: Optional reference audio for comparison
            mos_weight: Weight for MOS component
            sim_weight: Weight for similarity component
            
        Returns:
            SMOS score (1-5, higher is better)
        """
        # Get basic MOS score
        mos = self.mos_calculator.calculate(audio_path)
        
        # If reference audio provided and similarity calculator available
        if reference_audio and self.similarity_calculator:
            sim = self.similarity_calculator.calculate(audio_path, reference_audio)
            # Weighted combination (scale similarity to 0-5)
            smos = mos_weight * mos + sim_weight * (sim * 5)
        else:
            smos = mos
        
        return float(smos)
