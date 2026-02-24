"""
Custom ASR models for evaluation pipeline

NOTE: This is a basic wrapper for loading OmniASR model checkpoints.
If OmniASR has an official package/repository with specific model architecture
and loading utilities, please install it and update this implementation accordingly.

For official OmniASR implementation:
    pip install <omniasr-package>
    # or
    git clone <omniasr-repo> && pip install -e omniasr/
"""

import torch
import torchaudio
from typing import Optional, Dict
from pathlib import Path
import numpy as np
from fairseq2.models.hub import load_model
from fairseq2.data.tokenizers.hub import load_tokenizer
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


class OmniASR:
    """
    Wrapper for OmniLingual ASR models
    
    This is a basic implementation that loads .pt checkpoint files directly.
    Update this class if you have access to the official OmniASR implementation.
    """
    
    def __init__(
        self,
        model_card: str = "omniASR_LLM_7B",
        device: str = "cuda"
    ):
        """
        Initialize OmniASR model
        
        Args:
            model_card: Model card name (e.g., "omniASR_LLM_300M", "omniASR_CTC_1B")
            device: Computing device
        """
        self.device = device
        
        print(f"Loading OmniASR model: {model_card}...")
        
        # Create inference pipeline
        try:
            self.pipeline = ASRInferencePipeline(model_card=model_card)
            
            print(f"OmniASR model loaded successfully!")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTIP: Make sure fairseq2 and omnilingual_asr are installed:")
            print("  pip install fairseq2 omnilingual_asr")
            raise
    
    def transcribe(self, audio_path: str, lang: str = "khm_Khmr") -> str:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            lang: Language code (default: "khm_Khmr" for Khmer)
            
        Returns:
            Transcribed text
        """
        # Use pipeline's transcribe method
        # Note: audio_files is a positional argument, not keyword
        transcriptions = self.pipeline.transcribe(
            [audio_path],
            lang=[lang],
            batch_size=1
        )
        
        return transcriptions[0] if transcriptions else ""
    
    def __call__(self, audio_path: str) -> Dict[str, str]:
        """
        Make class callable like HuggingFace pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with 'text' key containing transcription
        """
        text = self.transcribe(audio_path)
        return {"text": text}


class ASRFactory:
    """
    Factory class to create different ASR models
    """
    
    @staticmethod
    def create_omni_asr(
        model_card: str = "omniASR_LLM_300M",
        device: str = "cuda"
    ) -> OmniASR:
        """
        Create OmniASR model
        
        Args:
            model_card: Model card name (e.g., "omniASR_LLM_300M", "omniASR_CTC_1B")
            device: Computing device
            
        Returns:
            OmniASR instance
        """
        return OmniASR(
            model_card=model_card,
            device=device
        )
    
    @staticmethod
    def create_whisper(
        model_name: str = "openai/whisper-large-v3",
        device: str = "cuda"
    ):
        """
        Create Whisper ASR model using HuggingFace
        
        Args:
            model_name: HuggingFace model name
            device: Computing device
            
        Returns:
            HuggingFace pipeline
        """
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device=device,
        )
