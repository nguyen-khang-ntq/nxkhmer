"""
Speaker Diarization Processor
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
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SpeakerDiarizer:
    """Extract single-speaker segments using pyannote diarization"""
    
    def __init__(self, config: dict = None, device: str = None):
        """
        Args:
            config: Configuration dictionary (from config.json)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        if config is None:
            config = {
                'model': 'pyannote/speaker-diarization-3.1',
                'min_segment_duration': 2.0,
                'max_segment_duration': 40.0
            }
        
        self.model_name = config.get('model', 'pyannote/speaker-diarization-3.1')
        self.min_duration = config.get('min_segment_duration', 2.0)
        self.max_duration = config.get('max_segment_duration', 40.0)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Loading diarization model: {self.model_name}")
        print(f"Device: {self.device}")
        
        # Load pipeline (use 'token' instead of deprecated 'use_auth_token')
        use_token = config.get('use_auth_token', True) if config else True
        if use_token:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=True
            )
        else:
            self.pipeline = Pipeline.from_pretrained(self.model_name)
        
        self.pipeline.to(self.device)
        
        print("✓ Diarization model loaded")
    
    def diarize_audio(self, audio_path: Union[str, Path], verbose: bool = False) -> List[Dict]:
        """
        Perform speaker diarization on a single audio file
        
        Args:
            audio_path: Path to audio file
            verbose: If True, print progress messages
            
        Returns:
            List of segment dictionaries with speaker labels and timestamps
        """
        try:
            # Get audio duration for progress estimation
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            duration_sec = waveform.shape[1] / sr
            
            if verbose:
                print(f"Running diarization on {duration_sec:.1f}s audio...", end=" ", flush=True)
            
            # Run diarization (this is the slow part - deep learning inference)
            diarization = self.pipeline(audio_path)
            
            # Extract segments - handle different pyannote API versions
            segments = []
            
            # Check for new pyannote-audio 3.x API (DiarizeOutput)
            if hasattr(diarization, 'speaker_diarization'):
                # pyannote-audio 3.x+ returns DiarizeOutput object
                # Extract the actual Annotation from it
                annotation = diarization.speaker_diarization
                for segment, track, speaker in annotation.itertracks(yield_label=True):
                    duration = segment.end - segment.start
                    
                    # Filter by duration
                    if self.min_duration <= duration <= self.max_duration:
                        segments.append({
                            'speaker': speaker,
                            'start': segment.start,
                            'end': segment.end,
                            'duration': duration
                        })
            elif hasattr(diarization, 'itertracks'):
                # pyannote-audio < 3.0 - Annotation object directly
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
            else:
                # Unknown format
                print(f"  Warning: Unknown diarization output type: {type(diarization)}")
                print(f"  Available methods: {[x for x in dir(diarization) if not x.startswith('_')][:10]}")
                return []
            
            if verbose:
                print(f"Found {len(segments)} segments", flush=True)
            
            return segments
            
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            print(f"Error diarizing {audio_path}: {e}")
            print(f"  Diarization object type: {type(diarization) if 'diarization' in locals() else 'N/A'}")
            if 'diarization' in locals():
                print(f"  Available attributes: {[x for x in dir(diarization) if not x.startswith('_')][:10]}")
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
    
    def process_file(self, audio_path: Union[str, Path], output_folder: Union[str, Path], verbose: bool = False) -> List[Dict]:
        """
        Process a single audio file: diarize and extract segments
        
        Args:
            audio_path: Path to input audio file
            output_folder: Folder to save extracted segments
            verbose: If True, print progress messages
            
        Returns:
            List of segment metadata
        """
        audio_path = Path(audio_path)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Diarize
        segments = self.diarize_audio(audio_path, verbose=verbose)
        
        if not segments:
            return []
        
        # Extract each segment (this is usually fast)
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
            json.dump({
                'total_input_files': len(audio_files),
                'total_segments': len(all_metadata),
                'config': {
                    'model': self.model_name,
                    'min_duration': self.min_duration,
                    'max_duration': self.max_duration
                },
                'segments': all_metadata
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nExtracted {len(all_metadata)} segments from {len(audio_files)} files")
        print(f"Segments saved to: {output_folder}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speaker diarization and segment extraction")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for segments")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cuda' or 'cpu'")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)['diarization']
    else:
        config = None
    
    diarizer = SpeakerDiarizer(config=config, device=args.device)
    diarizer.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
