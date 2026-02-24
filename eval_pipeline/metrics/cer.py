"""
Character Error Rate (CER) metric calculator
"""

from jiwer import cer as calculate_cer_score


class CERCalculator:
    """Calculate Character Error Rate using ASR transcription"""
    
    def __init__(self, asr_pipeline):
        """
        Initialize CER calculator
        
        Args:
            asr_pipeline: HuggingFace ASR pipeline
        """
        self.asr_pipeline = asr_pipeline
    
    def calculate(self, generated_audio: str, reference_text: str) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            generated_audio: Path to generated audio
            reference_text: Ground truth text
            
        Returns:
            CER score (lower is better, 0 = perfect)
        """
        # Transcribe generated audio
        result = self.asr_pipeline(generated_audio)
        hypothesis = result["text"]
        
        # Calculate CER
        cer_score = calculate_cer_score(reference_text, hypothesis)
        
        return float(cer_score)
