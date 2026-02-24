"""
Test Pipeline Components Individually
Quick test to verify each component works before running full pipeline
"""

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_config():
    """Test 1: Load configuration"""
    print("="*70)
    print("TEST 1: Configuration Loading")
    print("="*70)
    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("✓ Config loaded successfully")
        print(f"  Pipeline: {config['pipeline_config']['name']}")
        return config
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return None

def test_standardizer(config):
    """Test 2: Audio Standardizer"""
    print("\n" + "="*70)
    print("TEST 2: Audio Standardization")
    print("="*70)
    try:
        from processors.standardization import AudioStandardizer
        standardizer = AudioStandardizer(config=config['standardization'])
        print("✓ Standardizer initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize standardizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_diarizer(config):
    """Test 3: Speaker Diarizer"""
    print("\n" + "="*70)
    print("TEST 3: Speaker Diarization")
    print("="*70)
    try:
        from processors.diarization import SpeakerDiarizer
        diarizer = SpeakerDiarizer(config=config['diarization'], device='cuda')
        print("✓ Diarizer initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize diarizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quality_filter(config):
    """Test 4: Quality Filter"""
    print("\n" + "="*70)
    print("TEST 4: Quality Filter (DNSMOS)")
    print("="*70)
    try:
        from processors.quality_filter import QualityFilter
        quality_filter = QualityFilter(config=config['quality_filtering'])
        if quality_filter.enabled:
            print("✓ Quality filter initialized successfully")
            return True
        else:
            print("⚠ Quality filter initialized but DNSMOS disabled")
            return True
    except Exception as e:
        print(f"✗ Failed to initialize quality filter: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_transcriber(config):
    """Test 5: ASR Transcriber"""
    print("\n" + "="*70)
    print("TEST 5: ASR Transcription")
    print("="*70)
    try:
        from processors.transcription import Transcriber
        transcriber = Transcriber(config=config['asr'], device='cuda')
        print("✓ Transcriber initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize transcriber: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_audio_file(test_audio_path):
    """Test 6: Process a single audio file through standardization"""
    print("\n" + "="*70)
    print("TEST 6: Process Single Audio File")
    print("="*70)
    
    if not test_audio_path or not Path(test_audio_path).exists():
        print("⚠ No test audio file provided, skipping file processing test")
        print(f"  Provide path as argument: python test_pipeline_components.py <audio_file>")
        return True
    
    try:
        from processors.standardization import AudioStandardizer
        import tempfile
        
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test standardization on single file
        standardizer = AudioStandardizer(config=config['standardization'])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.wav"
            success = standardizer.standardize_audio(test_audio_path, output_path)
            
            if success and output_path.exists():
                print(f"✓ Successfully processed test file: {test_audio_path}")
                print(f"  Output: {output_path}")
                
                # Get duration
                import torchaudio
                waveform, sr = torchaudio.load(str(output_path))
                duration = waveform.shape[1] / sr
                print(f"  Duration: {duration:.2f}s")
                print(f"  Sample rate: {sr} Hz")
                print(f"  Channels: {waveform.shape[0]}")
                return True
            else:
                print(f"✗ Failed to process test file")
                return False
                
    except Exception as e:
        print(f"✗ Error processing test file: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test pipeline components")
    parser.add_argument("--audio", type=str, default=None, 
                       help="Path to test audio file (optional)")
    parser.add_argument("--skip-models", action='store_true',
                       help="Skip heavy model loading tests")
    
    args = parser.parse_args()
    
    print("\n")
    print("*" * 70)
    print("  AUDIO PIPELINE COMPONENT TESTS")
    print("*" * 70)
    print()
    
    results = {}
    
    # Test 1: Config
    config = test_config()
    results['config'] = config is not None
    
    if not config:
        print("\n✗ Cannot proceed without config")
        return
    
    # Test 2: Standardizer
    results['standardizer'] = test_standardizer(config)
    
    if not args.skip_models:
        # Test 3: Diarizer (heavy)
        results['diarizer'] = test_diarizer(config)
        
        # Test 4: Quality Filter
        results['quality_filter'] = test_quality_filter(config)
        
        # Test 5: Transcriber (heavy)
        results['transcriber'] = test_transcriber(config)
    else:
        print("\n⚠ Skipping model loading tests (use without --skip-models to test)")
    
    # Test 6: File processing
    if args.audio:
        results['file_processing'] = test_audio_file(args.audio)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("  You can now run the full pipeline")
    else:
        print("✗ SOME TESTS FAILED")
        print("  Fix the issues above before running full pipeline")
    print("="*70)
    print()

if __name__ == "__main__":
    main()
