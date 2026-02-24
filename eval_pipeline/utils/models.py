"""
Model loading utilities
"""

import torch
from transformers import (
    AutoModelForSpeechSeq2Seq, 
    AutoProcessor, 
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    pipeline
)
from resemblyzer import VoiceEncoder
from .custom_asr import ASRFactory


class ModelLoader:
    """Handles loading of all required models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.asr_pipeline = None
        self.mos_model = None
        self.mos_feature_extractor = None
        self.speaker_encoder = None
    
    def load_asr_model(self, model_name: str, asr_type: str = "whisper"):
        """
        Load ASR model for transcription
        
        Args:
            model_name: Model name or path
            asr_type: Type of ASR - "whisper", "omni", or "custom"
        """
        if asr_type == "omni":
            print(f"Loading OmniASR model: {model_name}...")
            # Parse model_name as "size-type" e.g., "1B-CTC"
            if "-" in model_name:
                size, mtype = model_name.split("-")
            else:
                size, mtype = model_name, "CTC"
            
            self.asr_pipeline = ASRFactory.create_omni_asr(
                model_size=size,
                model_type=mtype,
                device=self.device
            )
        else:
            # Default to Whisper
            print(f"Loading Whisper ASR model: {model_name}...")
            self.asr_pipeline = ASRFactory.create_whisper(
                model_name=model_name,
                device=self.device
            )
        
        return self.asr_pipeline
    
    def load_mos_model(self, model_name: str):
        """Load MOS prediction model"""
        print(f"Loading MOS model: {model_name}...")
        
        self.mos_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.mos_model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.mos_model.to(self.device)
        self.mos_model.eval()
        
        return self.mos_model, self.mos_feature_extractor
    
    def load_speaker_encoder(self):
        """Load speaker encoder for similarity"""
        print("Loading speaker encoder...")
        self.speaker_encoder = VoiceEncoder(device=self.device)
        return self.speaker_encoder
