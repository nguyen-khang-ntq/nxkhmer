"""
Speaker Diarization Module
Uses pyannote/speaker-diarization-3.1 to extract single-speaker segments (2-40s)
"""

import os
from pathlib import Path
from typing import Union, List, Dict
import torch
import torchaudio
from pyannote.audio import Pipeline
from tqdm import tqdm
import json


class SpeakerDiarizer:
    """Extract single-speaker segments using pyannote diarization"""
    
    def __init__(self, 
                 model_name: str = "pyannote/speaker-diarization-community-1",
                 min_duration: float = 2.0,
                 max_duration: float = 40.0,
                 device: str = None):
        """
        Args:
            model_name: Pyannote model identifier
            min_duration: Minimum segment duration in seconds (default: 2.0)
            max_duration: Maximum segment duration in seconds (default: 40.0)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading diarization model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load pipeline
        # Note: Requires HuggingFace token for pyannote models
        # Set token via: huggingface-cli login
        self.pipeline = Pipeline.from_pretrained(
            model_name,
            use_auth_token=True
        )
        self.pipeline.to(self.device)
        
        print("✓ Diarization model loaded")
    
    def diarize_audio(self, audio_path: Union[str, Path]) -> List[Dict]:
        """
        Perform speaker diarization on a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of segment dictionaries with speaker labels and timestamps
        """
        try:
            # Run diarization
            diarization = self.pipeline(audio_path)
            
            # Extract segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                
                # Filter by duration
                if self.min_duration <= duration <= self.max_duration:
                    segments.append({
                        'speaker': speaker,
                        'start': turn.start,
                        'end': turn.end,
                        'duration': duration
                    })
            
            return segments
            
        except Exception as e:
            print(f"Error diarizing {audio_path}: {e}")
            return []
    
    def extract_segment(self, audio_path: Union[str, Path], 
                       start_time: float, end_time: float,
                       output_path: Union[str, Path]) -> bool:
        """
        Extract audio segment between start and end times
        
        Args:
            audio_path: Path to source audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to save extracted segment
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert time to samples
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save segment
            torchaudio.save(output_path, segment, sr)
            
            return True
            
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return False
    
    def process_file(self, audio_path: Union[str, Path], output_folder: Union[str, Path]) -> List[Dict]:
        """
        Process a single audio file: diarize and extract segments
        
        Args:
            audio_path: Path to input audio file
            output_folder: Folder to save extracted segments
            
        Returns:
            List of segment metadata
        """
        audio_path = Path(audio_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Diarize
        segments = self.diarize_audio(audio_path)
        
        if not segments:
            print(f"No valid segments found in {audio_path.name}")
            return []
        
        # Extract each segment
        segment_metadata = []
        base_name = audio_path.stem
        
        for idx, segment in enumerate(segments):
            # Create output filename
            output_name = f"{base_name}_spk{segment['speaker']}_seg{idx:04d}.wav"
            output_path = output_folder / output_name
            
            # Extract segment
            if self.extract_segment(audio_path, segment['start'], segment['end'], output_path):
                segment_metadata.append({
                    'segment_file': str(output_path),
                    'source_file': str(audio_path),
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'duration': segment['duration']
                })
        
        return segment_metadata
    
    def process_folder(self, input_folder: Union[str, Path], 
                      output_folder: Union[str, Path],
                      metadata_file: str = "diarization_metadata.json"):
        """
        Process all audio files in a folder
        
        Args:
            input_folder: Folder containing input audio files
            output_folder: Folder to save extracted segments
            metadata_file: JSON file to save segment metadata
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all WAV files
        audio_files = list(input_folder.rglob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        # Process each file
        all_metadata = []
        for audio_path in tqdm(audio_files, desc="Diarizing audio"):
            segments = self.process_file(audio_path, output_folder)
            all_metadata.extend(segments)
        
        # Save metadata
        metadata_path = output_folder / metadata_file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\nExtracted {len(all_metadata)} segments from {len(audio_files)} files")
        print(f"Segments saved to: {output_folder}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker diarization and segment extraction")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for segments")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Minimum segment duration (default: 2.0s)")
    parser.add_argument("--max-duration", type=float, default=40.0, help="Maximum segment duration (default: 40.0s)")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda' or 'cpu'")
    
    args = parser.parse_args()
    
    diarizer = SpeakerDiarizer(
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        device=args.device
    )
    diarizer.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
