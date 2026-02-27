"""
Configuration for TTS Evaluation Pipeline
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    asr_model_name: str = "openai/whisper-large-v3"
    asr_type: str = "whisper"  # "whisper" or "omni"
    mos_model_name: str = "cdminix/wav2vec2-base-utmos"
    dnsmos_model_path: Optional[str] = None  # Path to DNSMOS ONNX model file
    device: str = "cuda"
    sample_rate: int = 16000


@dataclass
class EvalMetrics:
    """Container for evaluation metrics"""
    cer: float = 0.0
    mos: float = 0.0
    dnsmos: float = 0.0
    sim: float = 0.0
    rmse_f0: float = 0.0
    mcd: float = 0.0
    smos: float = 0.0
    
    def to_dict(self):
        return {
            'cer': self.cer,
            'mos': self.mos,
            'dnsmos': self.dnsmos,
            'sim': self.sim,
            'rmse_f0': self.rmse_f0,
            'mcd': self.mcd,
            'smos': self.smos
        }
