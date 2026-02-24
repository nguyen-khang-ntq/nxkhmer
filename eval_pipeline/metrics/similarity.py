"""
Speaker Similarity metric calculator
"""

import numpy as np
from resemblyzer import preprocess_wav


class SimilarityCalculator:
    """Calculate speaker similarity using embeddings"""
    
    def __init__(self, speaker_encoder):
        """
        Initialize similarity calculator
        
        Args:
            speaker_encoder: Resemblyzer VoiceEncoder
        """
        self.speaker_encoder = speaker_encoder
    
    def calculate(self, generated_audio: str, reference_audio: str) -> float:
        """
        Calculate speaker similarity between generated and reference audio
        
        Args:
            generated_audio: Path to generated audio
            reference_audio: Path to reference audio
            
        Returns:
            Cosine similarity score (0-1, higher is better)
        """
        # Load and preprocess audio
        gen_wav = preprocess_wav(generated_audio)
        ref_wav = preprocess_wav(reference_audio)
        
        # Get embeddings
        gen_embed = self.speaker_encoder.embed_utterance(gen_wav)
        ref_embed = self.speaker_encoder.embed_utterance(ref_wav)
        
        # Calculate cosine similarity
        similarity = np.dot(gen_embed, ref_embed) / (
            np.linalg.norm(gen_embed) * np.linalg.norm(ref_embed)
        )
        
        return float(similarity)
