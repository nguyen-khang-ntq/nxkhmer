"""
Audio Quality Filter using DNSMOS P.835
Filters out low-quality segments based on perceptual quality scores
"""

import os
from pathlib import Path
from typing import Union, List, Dict
import json
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import DNSMOS from eval_pipeline using direct file loading to avoid conflicts
DNSMOS = None
try:
    eval_pipeline_path = Path(__file__).parent.parent.parent / "eval_pipeline"
    dnsmos_path = eval_pipeline_path / "metrics" / "dnsmos.py"
    
    if dnsmos_path.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("eval_dnsmos", dnsmos_path)
        dnsmos_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dnsmos_module)
        DNSMOS = dnsmos_module.DNSMOSCalculator  # Use DNSMOSCalculator, not DNSMOS
    else:
        print(f"Warning: Could not find DNSMOS at {dnsmos_path}")
except Exception as e:
    print(f"Warning: Could not import DNSMOS: {e}")
    DNSMOS = None


class QualityFilter:
    """Filter audio segments based on DNSMOS P.835 quality scores"""
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: Configuration dictionary (from config.json)
        """
        if config is None:
            config = {
                'dnsmos_threshold': 3.0,
                'model_path': None,  # Path to DNSMOS ONNX model
                'primary_model': 'sig_bak_ovr.onnx',
                'p808_model': 'model_v8.onnx',
                'filter_criteria': {
                    'min_ovrl': 3.0,
                    'min_sig': 3.0,
                    'min_bak': 3.5
                }
            }
        
        self.threshold = config.get('dnsmos_threshold', 3.0)
        self.model_path = config.get('model_path', None)  # Path to DNSMOS ONNX model
        self.criteria = config.get('filter_criteria', {
            'min_ovrl': 3.0,
            'min_sig': 3.0,
            'min_bak': 3.5
        })
        
        # Initialize DNSMOS
        self.dnsmos = None
        self.enabled = False
        
        if DNSMOS is None:
            print("⚠ DNSMOS not available. Quality filtering will be skipped.")
            print("  To enable: ensure eval_pipeline/metrics/dnsmos.py is available")
        else:
            try:
                print("Initializing DNSMOS P.835...")
                # DNSMOSCalculator takes model_path (ONNX file) and device
                # Determine device
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.dnsmos = DNSMOS(
                    model_path=self.model_path,
                    device=device
                )
                self.enabled = True
                print(f"✓ DNSMOS initialized (device: {device})")
            except Exception as e:
                print(f"⚠ Failed to initialize DNSMOS: {e}")
                print("  Quality filtering will be skipped.")
                import traceback
                traceback.print_exc()
    
    def evaluate_audio(self, audio_path: Union[str, Path]) -> Dict:
        """
        Evaluate audio quality using DNSMOS
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with quality scores (ovrl, sig, bak, p808_mos)
        """
        if not self.enabled or self.dnsmos is None:
            return None
        
        try:
            scores = self.dnsmos.calculate(str(audio_path))
            return scores
        except Exception as e:
            print(f"Error evaluating {audio_path}: {e}")
            return None
    
    def passes_quality_filter(self, scores: Dict) -> bool:
        """
        Check if audio passes quality criteria
        
        Args:
            scores: DNSMOS scores dictionary
            
        Returns:
            True if passes all criteria, False otherwise
        """
        # If DNSMOS is disabled, pass all files
        if not self.enabled:
            return True
        
        if scores is None:
            return False
        
        # DNSMOSCalculator returns uppercase keys: OVRL, SIG, BAK
        # Check each criterion
        ovrl = scores.get('OVRL', scores.get('ovrl', 0))
        sig = scores.get('SIG', scores.get('sig', 0))
        bak = scores.get('BAK', scores.get('bak', 0))
        
        if ovrl < self.criteria['min_ovrl']:
            return False
        if sig < self.criteria['min_sig']:
            return False
        if bak < self.criteria['min_bak']:
            return False
        
        return True
    
    def filter_audio_list(self, audio_files: List[Union[str, Path]], 
                         output_folder: Union[str, Path] = None) -> List[Dict]:
        """
        Filter a list of audio files based on quality
        
        Args:
            audio_files: List of audio file paths
            output_folder: Optional folder to copy passed files to
            
        Returns:
            List of metadata for files that passed the filter
        """
        passed_files = []
        failed_files = []
        
        for audio_path in tqdm(audio_files, desc="Filtering audio quality"):
            audio_path = Path(audio_path)
            
            # Evaluate quality
            scores = self.evaluate_audio(audio_path)
            passes = self.passes_quality_filter(scores)
            
            metadata = {
                'audio_file': str(audio_path),
                'scores': scores,
                'passed': passes
            }
            
            if passes:
                passed_files.append(metadata)
                
                # Copy to output folder if specified
                if output_folder is not None:
                    import shutil
                    output_folder = Path(output_folder)
                    output_folder.mkdir(parents=True, exist_ok=True)
                    output_path = output_folder / audio_path.name
                    shutil.copy2(audio_path, output_path)
                    metadata['filtered_file'] = str(output_path)
            else:
                failed_files.append(metadata)
        
        return passed_files, failed_files
    
    def process_folder(self, input_folder: Union[str, Path], 
                      output_folder: Union[str, Path],
                      metadata_file: str = "quality_filter_metadata.json"):
        """
        Process all audio files in a folder
        
        Args:
            input_folder: Folder containing input audio files
            output_folder: Folder to save filtered audio
            metadata_file: JSON file to save filtering results
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all WAV files
        audio_files = list(input_folder.rglob("*.wav"))
        print(f"Found {len(audio_files)} audio files")
        
        # Filter files
        passed_files, failed_files = self.filter_audio_list(audio_files, output_folder)
        
        # Save metadata
        metadata_path = output_folder / metadata_file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(audio_files),
                'passed': len(passed_files),
                'failed': len(failed_files),
                'pass_rate': len(passed_files) / len(audio_files) if audio_files else 0,
                'config': {
                    'threshold': self.threshold,
                    'criteria': self.criteria
                },
                'passed_files': passed_files,
                'failed_files': failed_files
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\nQuality Filter Results:")
        print(f"  Total: {len(audio_files)}")
        print(f"  Passed: {len(passed_files)} ({len(passed_files)/len(audio_files)*100:.1f}%)")
        print(f"  Failed: {len(failed_files)} ({len(failed_files)/len(audio_files)*100:.1f}%)")
        print(f"Output saved to: {output_folder}")
        print(f"Metadata saved to: {metadata_path}")


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter audio by quality using DNSMOS")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing audio files")
    parser.add_argument("--output", type=str, required=True, help="Output folder for filtered audio")
    parser.add_argument("--config", type=str, default=None, help="Path to config.json")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)['quality_filtering']
    else:
        config = None
    
    filter = QualityFilter(config=config)
    filter.process_folder(args.input, args.output)


if __name__ == "__main__":
    main()
