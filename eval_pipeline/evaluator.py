"""
Main TTS Evaluator
Orchestrates all metric calculations
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from config import ModelConfig, EvalMetrics
from utils import ModelLoader
from metrics import (
    CERCalculator,
    MOSCalculator,
    DNSMOSCalculator,
    SimilarityCalculator,
    F0Calculator,
    MCDCalculator,
    SMOSCalculator
)


class TTSEvaluator:
    """
    Comprehensive TTS Evaluation Pipeline
    Orchestrates all metric calculators
    """
    
    def __init__(
        self,
        asr_model_name: str = "1B-CTC",
        asr_type: str = "omni",
        mos_model_name: str = "cdminix/wav2vec2-base-utmos",
        dnsmos_model_path: Optional[str] = None,
        device: str = "cuda",
        sample_rate: int = 16000
    ):
        """
        Initialize the TTS evaluator with required models
        
        Args:
            asr_model_name: ASR model name (for omni: "1B-CTC", "3B-LLM", etc; for whisper: HF model name)
            asr_type: Type of ASR - "omni" or "whisper"
            mos_model_name: HuggingFace MOS prediction model
            device: Computing device
            sample_rate: Target sample rate for audio processing
        """
        self.config = ModelConfig(
            asr_model_name=asr_model_name,
            asr_type=asr_type,
            mos_model_name=mos_model_name,
            dnsmos_model_path=dnsmos_model_path,
            device=device,
            sample_rate=sample_rate
        )
        
        print(f"Initializing TTS Evaluator on {device}...")
        
        # Load all models
        self.model_loader = ModelLoader(device=device)
        self._initialize_models()
        
        # Initialize metric calculators
        self._initialize_calculators()
        
        print("TTS Evaluator initialized successfully!")
    
    def _initialize_models(self):
        """Load all required models"""
        # ASR model for CER
        asr_pipeline = self.model_loader.load_asr_model(
            self.config.asr_model_name,
            self.config.asr_type
        )
        
        # MOS model
        mos_model, mos_feature_extractor = self.model_loader.load_mos_model(
            self.config.mos_model_name
        )
        
        # Speaker encoder for similarity
        speaker_encoder = self.model_loader.load_speaker_encoder()
        
        # Store references
        self.asr_pipeline = asr_pipeline
        self.mos_model = mos_model
        self.mos_feature_extractor = mos_feature_extractor
        self.speaker_encoder = speaker_encoder
    
    def _initialize_calculators(self):
        """Initialize all metric calculators"""
        self.cer_calculator = CERCalculator(self.asr_pipeline)
        self.mos_calculator = MOSCalculator(
            self.mos_model,
            self.mos_feature_extractor,
            self.config.device
        )
        
        # Initialize DNSMOS if model path provided
        self.dnsmos_calculator = None
        if self.config.dnsmos_model_path:
            print("Loading DNSMOS model...")
            from metrics.dnsmos import ComputeScore
            device_name = "cuda" if self.config.device == "cuda" else "cpu"
            self.dnsmos_compute_score = ComputeScore(
                self.config.dnsmos_model_path, 
                device_name
            )
            print("DNSMOS model loaded")
        
        self.similarity_calculator = SimilarityCalculator(self.speaker_encoder)
        self.f0_calculator = F0Calculator()
        self.mcd_calculator = MCDCalculator(sample_rate=self.config.sample_rate)
        self.smos_calculator = SMOSCalculator(
            self.mos_calculator,
            self.similarity_calculator
        )
    
    def evaluate_single(
        self,
        generated_audio: str,
        reference_text: Optional[str] = None,
        reference_audio: Optional[str] = None,
        compute_cer: bool = True,
        compute_mos: bool = True,
        compute_dnsmos: bool = True,
        compute_sim: bool = True,
        compute_f0: bool = True,
        compute_mcd: bool = True,
        compute_smos: bool = True
    ) -> EvalMetrics:
        """
        Evaluate a single audio file with all metrics
        
        Args:
            generated_audio: Path to generated audio
            reference_text: Reference text for CER
            reference_audio: Reference audio for similarity, F0, and MCD
            compute_*: Flags to enable/disable specific metrics
            
        Returns:
            EvalMetrics object with all scores
        """
        metrics = EvalMetrics()
        
        try:
            if compute_cer and reference_text:
                print("Computing CER...")
                metrics.cer = self.cer_calculator.calculate(
                    generated_audio, reference_text
                )
            
            if compute_mos:
                print("Computing MOS...")
                metrics.mos = self.mos_calculator.calculate(generated_audio)
            
            if compute_dnsmos and hasattr(self, 'dnsmos_compute_score') and self.dnsmos_compute_score:
                print("Computing DNSMOS...")
                metrics.dnsmos = self.calculate_dnsmos(generated_audio)
            
            if compute_sim and reference_audio:
                print("Computing speaker similarity...")
                metrics.sim = self.similarity_calculator.calculate(
                    generated_audio, reference_audio
                )
            
            if compute_f0 and reference_audio:
                print("Computing RMSE F0...")
                metrics.rmse_f0 = self.f0_calculator.calculate(
                    generated_audio, reference_audio
                )
            
            if compute_mcd and reference_audio:
                print("Computing MCD...")
                metrics.mcd = self.mcd_calculator.calculate(
                    generated_audio, reference_audio
                )
            
            if compute_smos:
                print("Computing SMOS...")
                metrics.smos = self.smos_calculator.calculate(
                    generated_audio, reference_audio
                )
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
        
        return metrics
    
    def evaluate_batch(
        self,
        audio_pairs: List[Dict[str, str]],
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Evaluate multiple audio files in batch
        
        Args:
            audio_pairs: List of dicts with keys:
                - 'generated': path to generated audio
                - 'reference_text': reference text (optional)
                - 'reference_audio': path to reference audio (optional)
                - 'id': unique identifier (optional)
            output_file: Path to save results CSV
            
        Returns:
            DataFrame with all evaluation results
        """
        results = []
        
        for item in tqdm(audio_pairs, desc="Evaluating TTS samples"):
            generated = item['generated']
            ref_text = item.get('reference_text')
            ref_audio = item.get('reference_audio')
            sample_id = item.get('id', os.path.basename(generated))
            
            print(f"\nEvaluating: {sample_id}")
            
            metrics = self.evaluate_single(
                generated_audio=generated,
                reference_text=ref_text,
                reference_audio=ref_audio
            )
            
            result = {'id': sample_id, **metrics.to_dict()}
            results.append(result)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate average metrics
        print("\n" + "="*50)
        print("AVERAGE METRICS:")
        print("="*50)
        for col in df.columns:
            if col != 'id':
                avg = df[col].mean()
                print(f"{col.upper()}: {avg:.4f}")
        
        # Save to file if specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        
        return df
    
    # Convenience methods for individual metrics
    def calculate_cer(self, generated_audio: str, reference_text: str) -> float:
        """Calculate CER only"""
        return self.cer_calculator.calculate(generated_audio, reference_text)
    
    def calculate_mos(self, audio_path: str) -> float:
        """Calculate MOS only"""
        return self.mos_calculator.calculate(audio_path)
    
    def calculate_dnsmos_with_vad(self, audio_path: str, vad_list: List[Dict]) -> tuple:
        """
        Calculate DNSMOS with VAD segments
        
        Args:
            audio_path: Path to audio file
            vad_list: List of VAD segments with 'start' and 'end' times in seconds
            
        Returns:
            tuple: (average_dnsmos, updated_vad_list_with_scores)
        """
        import librosa
        import numpy as np
        from tqdm import tqdm
        
        if not hasattr(self, 'dnsmos_compute_score') or self.dnsmos_compute_score is None:
            raise RuntimeError("DNSMOS model not initialized. Provide dnsmos_model_path during initialization.")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        sample_rate = 16000
        
        # Resample if needed
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        
        # Calculate DNSMOS for each VAD segment
        for index, vad in enumerate(tqdm(vad_list, desc="DNSMOS")):
            start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
            segment = audio[start:end]
            
            # Compute DNSMOS for this segment
            dnsmos_score = self.dnsmos_compute_score(segment, sample_rate, False)["OVRL"]
            vad_list[index]["dnsmos"] = dnsmos_score
        
        # Calculate average DNSMOS
        avg_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])
        
        print(f"Average DNSMOS for whole audio: {avg_dnsmos:.4f}")
        
        return avg_dnsmos, vad_list
    
    def calculate_dnsmos(self, audio_path: str) -> float:
        """Calculate DNSMOS only (without VAD)"""
        if not hasattr(self, 'dnsmos_compute_score') or self.dnsmos_compute_score is None:
            raise RuntimeError("DNSMOS model not initialized. Provide dnsmos_model_path during initialization.")
        
        import librosa
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Compute DNSMOS
        result = self.dnsmos_compute_score(audio, sr, False)
        return result["OVRL"]
    
    def calculate_similarity(self, generated_audio: str, reference_audio: str) -> float:
        """Calculate speaker similarity only"""
        return self.similarity_calculator.calculate(generated_audio, reference_audio)
    
    def calculate_rmse_f0(self, generated_audio: str, reference_audio: str) -> float:
        """Calculate RMSE F0 only"""
        return self.f0_calculator.calculate(generated_audio, reference_audio)
    
    def calculate_mcd(self, generated_audio: str, reference_audio: str) -> float:
        """Calculate MCD only"""
        return self.mcd_calculator.calculate(generated_audio, reference_audio)
    
    def calculate_smos(self, audio_path: str, reference_audio: Optional[str] = None) -> float:
        """Calculate SMOS only"""
        return self.smos_calculator.calculate(audio_path, reference_audio)
