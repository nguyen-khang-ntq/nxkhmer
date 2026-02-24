"""
TTS Evaluation Pipeline
Supports multiple metrics: CER, MOS, SIM, RMSE_F0, MCD, SMOS
"""

import os
import numpy as np
import librosa
import torch
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

# Audio processing
import soundfile as sf

# For CER calculation
from jiwer import cer

# For speaker similarity
from resemblyzer import VoiceEncoder, preprocess_wav

# For F0 extraction
import parselmouth
from parselmouth.praat import call

# For MCD calculation
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


@dataclass
class EvalMetrics:
    """Container for evaluation metrics"""
    cer: float = 0.0
    mos: float = 0.0
    sim: float = 0.0
    rmse_f0: float = 0.0
    mcd: float = 0.0
    smos: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class TTSEvaluator:
    """
    Comprehensive TTS Evaluation Pipeline
    """
    
    def __init__(
        self,
        asr_model_name: str = "openai/whisper-large-v3",
        mos_model_name: str = "cdminix/wav2vec2-base-utmos",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 16000
    ):
        """
        Initialize the TTS evaluator with required models
        
        Args:
            asr_model_name: HuggingFace ASR model for CER calculation
            mos_model_name: HuggingFace MOS prediction model
            device: Computing device
            sample_rate: Target sample rate for audio processing
        """
        self.device = device
        self.sample_rate = sample_rate
        
        print(f"Initializing TTS Evaluator on {device}...")
        
        # Load ASR model for CER
        self._load_asr_model(asr_model_name)
        
        # Load MOS prediction model
        self._load_mos_model(mos_model_name)
        
        # Load speaker encoder for similarity
        print("Loading speaker encoder...")
        self.speaker_encoder = VoiceEncoder(device=device)
        
        print("TTS Evaluator initialized successfully!")
    
    def _load_asr_model(self, model_name: str):
        """Load ASR model for transcription"""
        print(f"Loading ASR model: {model_name}...")
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        model.to(self.device)
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device=self.device,
        )
    
    def _load_mos_model(self, model_name: str):
        """Load MOS prediction model"""
        print(f"Loading MOS model: {model_name}...")
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        
        self.mos_feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.mos_model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.mos_model.to(self.device)
        self.mos_model.eval()
    
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (if None, use self.sample_rate)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if target_sr is None:
            target_sr = self.sample_rate
            
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio, sr
    
    def calculate_cer(self, generated_audio: str, reference_text: str) -> float:
        """
        Calculate Character Error Rate
        
        Args:
            generated_audio: Path to generated audio
            reference_text: Ground truth text
            
        Returns:
            CER score (lower is better)
        """
        # Transcribe generated audio
        result = self.asr_pipeline(generated_audio)
        hypothesis = result["text"]
        
        # Calculate CER
        cer_score = cer(reference_text, hypothesis)
        
        return cer_score
    
    def calculate_mos(self, audio_path: str) -> float:
        """
        Calculate predicted MOS (Mean Opinion Score)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            MOS score (1-5, higher is better)
        """
        audio, sr = self.load_audio(audio_path, target_sr=16000)
        
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
            # Assuming the model outputs a single value or class probabilities
            if logits.shape[-1] == 1:
                mos_score = float(logits.squeeze())
            else:
                # If multi-class, convert to score
                mos_score = float(torch.argmax(logits, dim=-1)) + 1
        
        return mos_score
    
    def calculate_similarity(self, generated_audio: str, reference_audio: str) -> float:
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
    
    def extract_f0(self, audio_path: str) -> np.ndarray:
        """
        Extract F0 (fundamental frequency) contour from audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            F0 contour array
        """
        snd = parselmouth.Sound(audio_path)
        pitch = call(snd, "To Pitch", 0.0, 75, 600)  # Time step, min F0, max F0
        
        # Extract pitch values
        f0_values = []
        for i in range(pitch.get_number_of_frames()):
            f0 = pitch.get_value_in_frame(i)
            if f0 > 0:  # Only voiced frames
                f0_values.append(f0)
        
        return np.array(f0_values)
    
    def calculate_rmse_f0(self, generated_audio: str, reference_audio: str) -> float:
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
        
        # Align lengths using interpolation
        if len(gen_f0) != len(ref_f0):
            min_len = min(len(gen_f0), len(ref_f0))
            gen_f0 = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(gen_f0)),
                gen_f0
            )
            ref_f0 = np.interp(
                np.linspace(0, 1, min_len),
                np.linspace(0, 1, len(ref_f0)),
                ref_f0
            )
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((gen_f0 - ref_f0) ** 2))
        
        return float(rmse)
    
    def extract_mfcc(self, audio: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features"""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T  # Transpose to (time, features)
    
    def calculate_mcd(self, generated_audio: str, reference_audio: str) -> float:
        """
        Calculate Mel-Cepstral Distortion (MCD)
        
        Args:
            generated_audio: Path to generated audio
            reference_audio: Path to reference audio
            
        Returns:
            MCD score (lower is better)
        """
        # Load audio
        gen_audio, sr = self.load_audio(generated_audio)
        ref_audio, _ = self.load_audio(reference_audio)
        
        # Extract MFCCs
        gen_mfcc = self.extract_mfcc(gen_audio, sr)
        ref_mfcc = self.extract_mfcc(ref_audio, sr)
        
        # Use DTW to align sequences
        distance, path = fastdtw(gen_mfcc, ref_mfcc, dist=euclidean)
        
        # Calculate MCD
        mcd = (10.0 / np.log(10)) * np.sqrt(2 * distance / len(path))
        
        return float(mcd)
    
    def calculate_smos(self, audio_path: str, reference_audio: Optional[str] = None) -> float:
        """
        Calculate Synthetic MOS (SMOS) - combination of objective metrics
        
        Args:
            audio_path: Path to audio file
            reference_audio: Optional reference audio for comparison
            
        Returns:
            SMOS score (1-5, higher is better)
        """
        # Get basic MOS score
        mos = self.calculate_mos(audio_path)
        
        # If reference audio provided, incorporate similarity
        if reference_audio:
            sim = self.calculate_similarity(audio_path, reference_audio)
            # Weighted combination
            smos = 0.7 * mos + 0.3 * (sim * 5)  # Scale similarity to 0-5
        else:
            smos = mos
        
        return float(smos)
    
    def evaluate_single(
        self,
        generated_audio: str,
        reference_text: Optional[str] = None,
        reference_audio: Optional[str] = None,
        compute_cer: bool = True,
        compute_mos: bool = True,
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
                metrics.cer = self.calculate_cer(generated_audio, reference_text)
            
            if compute_mos:
                print("Computing MOS...")
                metrics.mos = self.calculate_mos(generated_audio)
            
            if compute_sim and reference_audio:
                print("Computing speaker similarity...")
                metrics.sim = self.calculate_similarity(generated_audio, reference_audio)
            
            if compute_f0 and reference_audio:
                print("Computing RMSE F0...")
                metrics.rmse_f0 = self.calculate_rmse_f0(generated_audio, reference_audio)
            
            if compute_mcd and reference_audio:
                print("Computing MCD...")
                metrics.mcd = self.calculate_mcd(generated_audio, reference_audio)
            
            if compute_smos:
                print("Computing SMOS...")
                metrics.smos = self.calculate_smos(generated_audio, reference_audio)
        
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


def main():
    """Example usage"""
    # Initialize evaluator
    evaluator = TTSEvaluator(
        asr_model_name="openai/whisper-large-v3",
        mos_model_name="cdminix/wav2vec2-base-utmos",
        device="cuda"
    )
    
    # Example 1: Evaluate single file
    metrics = evaluator.evaluate_single(
        generated_audio="path/to/generated.wav",
        reference_text="This is the reference text",
        reference_audio="path/to/reference.wav"
    )
    
    print("\nSingle file metrics:")
    print(json.dumps(metrics.to_dict(), indent=2))
    
    # Example 2: Batch evaluation
    audio_pairs = [
        {
            'id': 'sample_1',
            'generated': 'path/to/generated1.wav',
            'reference_text': 'Hello world',
            'reference_audio': 'path/to/reference1.wav'
        },
        {
            'id': 'sample_2',
            'generated': 'path/to/generated2.wav',
            'reference_text': 'Testing TTS',
            'reference_audio': 'path/to/reference2.wav'
        }
    ]
    
    results_df = evaluator.evaluate_batch(
        audio_pairs=audio_pairs,
        output_file="tts_evaluation_results.csv"
    )


if __name__ == "__main__":
    main()
