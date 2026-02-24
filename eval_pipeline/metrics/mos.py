"""
Mean Opinion Score (MOS) metric calculator
"""

import torch
from utils.audio import load_audio


class MOSCalculator:
    """Calculate predicted Mean Opinion Score"""
    
    def __init__(self, mos_model, mos_feature_extractor, device: str = "cuda"):
        """
        Initialize MOS calculator
        
        Args:
            mos_model: MOS prediction model
            mos_feature_extractor: Feature extractor for MOS model
            device: Computing device
        """
        self.mos_model = mos_model
        self.mos_feature_extractor = mos_feature_extractor
        self.device = device
    
    def calculate(self, audio_path: str) -> float:
        """
        Calculate predicted MOS (Mean Opinion Score)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MOS score (1-5, higher is better)
        """
        audio, sr = load_audio(audio_path, target_sr=16000)
        
        # Prepare input
        inputs = self.mos_feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict MOS
        with torch.no_grad():
            outputs = self.mos_model(**inputs)
            logits = outputs.logits
            
            # Convert logits to MOS score (1-5 scale)
            if logits.shape[-1] == 1:
                mos_score = float(logits.squeeze())
            else:
                # If multi-class, convert to score
                mos_score = float(torch.argmax(logits, dim=-1)) + 1
        
        return mos_score
