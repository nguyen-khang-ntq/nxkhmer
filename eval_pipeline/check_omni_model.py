"""
Check OmniASR model checkpoint structure
"""

import torch
from pathlib import Path

model_dir = Path("/home/coder/data/Speech/ASR/omnilingual-asr")

# Check one model file
model_path = model_dir / "omniASR-CTC-1B.pt"

if model_path.exists():
    print(f"Loading {model_path}...")
    print("=" * 60)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Checkpoint type: {type(checkpoint)}")
    print()
    
    if isinstance(checkpoint, dict):
        print("Checkpoint keys:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            print(f"  - {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    shape: {value.shape}")
        print()
        
        # Check if it's a model or state_dict
        if 'model' in checkpoint:
            print("✓ Contains 'model' key")
            print(f"  Model type: {type(checkpoint['model'])}")
        elif 'state_dict' in checkpoint:
            print("✓ Contains 'state_dict' key")
            print("  → Need model architecture to load this")
            print("  → Install official OmniASR package required")
        else:
            print("⚠ Unknown checkpoint format")
    else:
        print(f"✓ Direct model object: {type(checkpoint)}")
        if hasattr(checkpoint, 'eval'):
            print("  → Has eval() method")
        if hasattr(checkpoint, 'forward'):
            print("  → Has forward() method")
    
    print()
    print("=" * 60)
    print("\nConclusion:")
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("❌ This checkpoint requires official OmniASR package")
        print("   You need to install OmniASR to use these models")
        print("   OR use Whisper instead: asr_type='whisper'")
    else:
        print("✓ Checkpoint can be loaded directly")
        print("  Current wrapper should work")
else:
    print(f"❌ Model file not found: {model_path}")
