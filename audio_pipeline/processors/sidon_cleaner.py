"""
Sidon Audio Cleaner Processor
Speech restoration and cleaning using Sidon model as final pipeline step
"""

from pathlib import Path
from typing import Optional

import torch
import torchaudio
from peft import PeftModel
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from tqdm import tqdm
import dac
from huggingface_hub import hf_hub_download


class SidonCleaner:
    """Sidon audio cleaner processor for pipeline integration"""
    
    def __init__(
        self,
        config: dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize Sidon cleaner for audio pipeline
        
        Args:
            config: Sidon configuration dict from config.json
            device: Device to run inference on
        """
        self.config = config
        self.device = device
        self.enabled = config.get('enabled', True)
        
        if not self.enabled:
            print("  Sidon cleaning disabled in config")
            return
        
        self.target_sr = config.get('target_sample_rate', 48000)
        self.input_sr = 16000  # Model expects 16kHz input
        self.use_fp16 = config.get('use_fp16', True) and device == "cuda"
        
        model_name = config.get('model_name', 'facebook/w2v-bert-2.0')
        adapter_path = config.get('adapter_path', 'sarulab-speech/sidon_raw_weight')
        decoder_path = config.get('decoder_path', None)
        
        print(f"  Loading Sidon model on {device}...")
        print(f"  Base model: {model_name}")
        print(f"  Adapter: {adapter_path}")
        
        # Load encoder with adapter
        base_model = Wav2Vec2BertModel.from_pretrained(
            model_name,
            num_hidden_layers=8,
            layerdrop=0.0
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        self.model.to(device)
        
        # Load decoder
        self.decoder = dac.model.dac.Decoder(
            input_channel=self.model.config.hidden_size,
            channels=1536,
            rates=[8, 5, 4, 3, 2],
        )
        
        if decoder_path is None:
            decoder_path = hf_hub_download(
                'sarulab-speech/sidon_raw_weight',
                'decoder_state_dict.pt'
            )
        
        self.decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        self.decoder.eval()
        self.decoder.to(device)
        
        # Load preprocessor
        self.preprocessor = SeamlessM4TFeatureExtractor()
        
        print(f"  ✓ Sidon model loaded (FP16: {self.use_fp16})")
    
    @torch.inference_mode()
    def clean_audio(self, audio_path: Path, output_path: Path) -> bool:
        """
        Clean a single audio file
        
        Args:
            audio_path: Path to input audio file
            output_path: Path to save cleaned audio
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load audio
            wav, sr = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Resample to 16kHz for model input
            if sr != self.input_sr:
                wav = torchaudio.functional.resample(wav, sr, self.input_sr)
            
            # Store original length for cropping later
            original_length = wav.shape[-1]
            
            # Add padding to the end (0.5 seconds = 8000 samples at 16kHz)
            padding_samples = int(0.5 * self.input_sr)
            wav_padded = torch.nn.functional.pad(wav, (0, padding_samples), mode='constant', value=0)
            
            # Preprocess
            ssl_input = self.preprocessor(
                wav_padded.numpy(),
                sampling_rate=self.input_sr,
                return_tensors='pt'
            )
            
            # Move to device
            ssl_input = {k: v.to(self.device) for k, v in ssl_input.items()}
            
            # Predict features and restore audio
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    predicted_feature = self.model(**ssl_input).last_hidden_state
                    restored_wav = self.decoder(predicted_feature.transpose(1, 2))
            else:
                predicted_feature = self.model(**ssl_input).last_hidden_state
                restored_wav = self.decoder(predicted_feature.transpose(1, 2))
            
            # Crop back to original length (accounting for 16kHz -> target_sr upsampling)
            output_length = int(original_length * (self.target_sr / self.input_sr))
            restored_wav = restored_wav[..., :output_length]
            
            # Convert to CPU and ensure float32
            restored_wav = restored_wav.cpu()
            if restored_wav.dtype == torch.float16:
                restored_wav = restored_wav.float()
            
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cleaned audio
            torchaudio.save(
                str(output_path),
                restored_wav.squeeze(0),  # Remove batch dimension
                self.target_sr
            )
            
            return True
            
        except Exception as e:
            print(f"  Error cleaning {audio_path.name}: {e}")
            return False
    
    def process_folder(self, input_folder: Path, output_folder: Path, verbose: bool = True):
        """
        Process all audio files in a folder
        
        Args:
            input_folder: Path to folder containing audio files
            output_folder: Path to output folder for cleaned audio
            verbose: Print progress information
        """
        if not self.enabled:
            if verbose:
                print("  Sidon cleaning disabled, skipping...")
            return
        
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all WAV files
        audio_files = list(input_folder.rglob("*.wav"))
        
        if not audio_files:
            if verbose:
                print("  No audio files found to clean")
            return
        
        if verbose:
            print(f"  Cleaning {len(audio_files)} audio files with Sidon...")
        
        # Process each file
        success_count = 0
        failed_count = 0
        
        for audio_file in tqdm(audio_files, desc="  Sidon cleaning", disable=not verbose):
            # Maintain folder structure
            relative_path = audio_file.relative_to(input_folder)
            output_path = output_folder / relative_path
            
            if self.clean_audio(audio_file, output_path):
                success_count += 1
            else:
                failed_count += 1
        
        if verbose:
            print(f"  ✓ Cleaned: {success_count} files")
            if failed_count > 0:
                print(f"  ✗ Failed: {failed_count} files")


def main():
    """Main entry point for standalone use"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Sidon Audio Cleaner - Speech restoration for pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Process with default config
  python sidon_cleaner.py --input /path/to/audio --output /path/to/cleaned
  
  # Specify config file
  python sidon_cleaner.py --input /path/to/audio --output /path/to/cleaned --config ../config.json
  
  # Specify device
  python sidon_cleaner.py --input /path/to/audio --output /path/to/cleaned --device cuda
        """
    )
    
    parser.add_argument("--input", type=str, required=True,
                       help="Input folder containing audio files to clean")
    parser.add_argument("--output", type=str, required=True,
                       help="Output folder for cleaned audio files")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config.json (default: ../config.json)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Load config
    if args.config is None:
        args.config = Path(__file__).parent.parent / "config.json"
    
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    
    sidon_config = full_config.get('sidon', {})
    
    # Override device if specified
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize cleaner
    cleaner = SidonCleaner(config=sidon_config, device=device)
    
    # Process folder
    cleaner.process_folder(
        input_folder=Path(args.input),
        output_folder=Path(args.output),
        verbose=True
    )
    
    print("\n✓ Sidon cleaning complete!")


if __name__ == "__main__":
    main()
