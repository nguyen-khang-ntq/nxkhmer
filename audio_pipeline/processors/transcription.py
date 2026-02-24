"""
Audio Transcription using Omni ASR
Transcribes filtered audio segments for TTS training
"""

import os
from pathlib import Path
from typing import Union, List, Dict
import json
from tqdm import tqdm
import pandas as pd
import sys
import jiwer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.omni_asr import OmniASR


class Transcriber:
    """Transcribe audio segments using dual ASR models with CER alignment"""
    
    def __init__(self, config: dict = None, device: str = None):
        """
        Args:
            config: Configuration dictionary (from config.json)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if config is None:
            config = {
                'model_card_ctc': 'omniASR_CTC_300M',
                'model_card_llm': 'omniASR_LLM_7B',
                'target_sample_rate': 16000,
                'cer_threshold': 0.01
            }
        
        self.model_card_ctc = config.get('model_card_ctc', 'omniASR_CTC_300M')
        self.model_card_llm = config.get('model_card_llm', 'omniASR_LLM_7B')
        self.target_sr = config.get('target_sample_rate', 16000)
        self.cer_threshold = config.get('cer_threshold', 0.01)
        
        # Initialize both ASR models
        print("Initializing CTC ASR model...")
        self.asr_ctc = OmniASR(model_card=self.model_card_ctc, device=device)
        print("✓ CTC ASR ready")
        
        print("Initializing LLM ASR model...")
        self.asr_llm = OmniASR(model_card=self.model_card_llm, device=device)
        print("✓ LLM ASR ready")
    
    def transcribe_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Transcribe a single audio file using dual ASR models with CER alignment
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio path, transcription, and CER score
            Returns None if CER > threshold
        """
        try:
            # Transcribe with both models
            result_ctc = self.asr_ctc(audio_path)
            result_llm = self.asr_llm(audio_path)
            
            text_ctc = result_ctc.get('text', '').strip()
            text_llm = result_llm.get('text', '').strip()
            
            # Skip if either transcription is empty
            if not text_ctc or not text_llm:
                return None
            
            # Calculate CER between the two transcriptions
            cer = jiwer.cer(text_llm, text_ctc)
            
            # Only keep if CER is below threshold
            if cer >= self.cer_threshold:
                return None
            
            duration = self._get_duration(audio_path)
            
            return {
                'audio_path': str(audio_path),
                'text': text_llm,  # Use LLM transcription as primary
                'text_ctc': text_ctc,  # Keep CTC for reference
                'cer': cer,
                'duration': duration
            }
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return None
    
    def _get_duration(self, audio_path: Union[str, Path]) -> float:
        """Get audio duration"""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            return waveform.shape[1] / sr
        except:
            return 0.0
    
    def transcribe_batch(self, audio_files: List[Union[str, Path]]) -> List[Dict]:
        """
        Transcribe a batch of audio files
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of transcription results
        """
        results = []
        
        for audio_path in tqdm(audio_files, desc="Transcribing audio"):
            audio_path = Path(audio_path)
            result = self.transcribe_audio(audio_path)
            
            if result and result.get('text', '').strip():
                results.append(result)
            else:
                results.append({
                    'audio_file': str(audio_path),
                    'text': '',
                    'status': 'failed'
                })
        
        return results
    
    def save_dataset(self, transcriptions: List[Dict], output_folder: Union[str, Path],
                    csv_filename: str = "dataset.csv", json_filename: str = "transcriptions.json"):
        """
        Save transcriptions as dataset files
        
        Args:
            transcriptions: List of transcription dictionaries
            output_folder: Folder to save dataset
            csv_filename: Name of CSV file
            json_filename: Name of JSON file
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Filter valid transcriptions
        valid_transcriptions = [t for t in transcriptions if t and t.get('text', '').strip()]
        
        # Save as JSON
        json_path = output_folder / json_filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(valid_transcriptions, f, indent=2, ensure_ascii=False)
        
        # Save as CSV for easy use in TTS training
        csv_path = output_folder / csv_filename
        
        # Determine columns based on what data is available
        columns = ['audio_path', 'text', 'duration']
        if valid_transcriptions and 'cer' in valid_transcriptions[0]:
            columns.append('cer')
        if valid_transcriptions and 'mos_ovrl' in valid_transcriptions[0]:
            columns.extend(['mos_ovrl', 'mos_sig', 'mos_bak'])
        
        df = pd.DataFrame(valid_transcriptions)
        # Select only available columns
        available_cols = [col for col in columns if col in df.columns]
        df[available_cols].to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✓ Saved {len(valid_transcriptions)} transcriptions")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def process_folder(self, input_folder: Union[str, Path], 
                      output_folder: Union[str, Path],
                      metadata_file: str = "transcription_metadata.json",
                      csv_file: str = "dataset.csv"):
        """
        Process all audio files in a folder
        
        Args:
            input_folder: Folder containing input audio files
            output_folder: Folder to save transcription results
            metadata_file: JSON file to save transcription metadata
            csv_file: CSV file in TTS training format
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all WAV files
        audio_files = list(input_folder.rglob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        # Transcribe all files
        results = self.transcribe_batch(audio_files)
        
        # Filter successful transcriptions
        successful = [r for r in results if r['status'] == 'success' and r['text']]
        
        # Save JSON metadata
        metadata_path = output_folder / metadata_file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(audio_files),
                'successful': len(successful),
                'failed': len(audio_files) - len(successful),
                'config': {
                    'model_card': self.model_card,
                    'target_sr': self.target_sr
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Create CSV for TTS training
        csv_path = output_folder / csv_file
        df = pd.DataFrame([
            {
                'audio_file': r['audio_file'],
                'text': r['text']
            }
            for r in successful
        ])
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"\nTranscription Results:")
        print(f"  Total: {len(audio_files)}")
        print(f"  Successful: {len(successful)} ({len(successful)/len(audio_files)*100:.1f}%)")
        print(f"  Failed: {len(audio_files) - len(successful)}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Dataset CSV saved to: {csv_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe audio using Omni ASR")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for transcriptions")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda' or 'cpu'")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)['asr']
    else:
        config = None
    
    transcriber = Transcriber(config=config, device=args.device)
    transcriber.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
