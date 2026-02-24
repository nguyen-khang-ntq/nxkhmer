"""
Omni ASR wrapper for transcription
"""

import torch
from pathlib import Path
from typing import Union, Dict
import sys


class OmniASR:
    """Wrapper for Omni ASR model"""
    
    def __init__(self, model_card: str = "omniASR_LLM_7B", device: str = None):
        """
        Initialize Omni ASR model
        
        Args:
            model_card: Model identifier (default: omniASR_LLM_7B)
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_card = model_card
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading Omni ASR: {model_card}")
        print(f"Device: {self.device}")
        
        # Import ASR factory from eval_pipeline
        try:
            # Locate eval_pipeline
            eval_pipeline_path = Path(__file__).parent.parent.parent / "eval_pipeline"
            if not eval_pipeline_path.exists():
                raise ImportError(f"Cannot find eval_pipeline at {eval_pipeline_path}")
            
            # Directly load the custom_asr module from eval_pipeline
            import importlib.util
            custom_asr_path = eval_pipeline_path / "utils" / "custom_asr.py"
            
            if not custom_asr_path.exists():
                raise ImportError(f"Cannot find custom_asr.py at {custom_asr_path}")
            
            spec = importlib.util.spec_from_file_location("eval_custom_asr", custom_asr_path)
            custom_asr_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_asr_module)
            
            ASRFactory = custom_asr_module.ASRFactory
            
            self.asr_model = ASRFactory.create_omni_asr(
                model_card=model_card,
                device=self.device
            )
                
        except Exception as e:
            print(f"Error loading ASR model: {e}")
            print("Please ensure the eval_pipeline/utils/custom_asr.py is available")
            raise
        
        print("✓ Omni ASR loaded")
    
    def transcribe(self, audio_path: Union[str, Path]) -> Dict[str, str]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file (should be 16kHz for optimal results)
            
        Returns:
            Dictionary with 'text' key containing transcription
        """
        try:
            # Ensure audio is 16kHz for ASR
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            
            if sr != 16000:
                # Resample to 16kHz
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
                
                # Save temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                    torchaudio.save(tmp_path, waveform, 16000)
                
                result = self.asr_model(tmp_path)
                
                # Clean up
                import os
                os.remove(tmp_path)
            else:
                result = self.asr_model(str(audio_path))
            
            return result
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return {'text': ''}
    
    def __call__(self, audio_path: Union[str, Path]) -> Dict[str, str]:
        """Allow calling the instance directly"""
        return self.transcribe(audio_path)
