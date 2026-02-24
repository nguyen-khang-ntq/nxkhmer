"""
Example: Using DNSMOS with VAD segments
Demonstrates the pattern for loading and computing DNSMOS scores
"""

import numpy as np
import librosa
import tqdm
from pathlib import Path
from typing import List, Dict
from metrics import dnsmos


def time_logger(func):
    """Decorator to log execution time"""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[TIME] {func.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


# Configuration
cfg = {
    "mos_model": {
        "primary_model_path": "path/to/dnsmos_model.onnx"  # Update this path
    },
    "entrypoint": {
        "SAMPLE_RATE": 16000
    }
}

# Device setup
device_name = "cpu"  # or "cuda" if available


# Initialize DNSMOS Model (do this once at startup)
print("Loading DNSMOS Model...")
primary_model_path = cfg["mos_model"]["primary_model_path"]
dnsmos_compute_score = dnsmos.ComputeScore(primary_model_path, device_name)
print("All models loaded")


@time_logger
def mos_prediction(audio, vad_list):
    """
    Predict the Mean Opinion Score (MOS) for the given audio and VAD segments.

    Args:
        audio (dict): A dictionary containing the audio waveform and sample rate.
        vad_list (list): List of VAD segments with start and end times.

    Returns:
        tuple: A tuple containing the average MOS and the updated VAD segments with MOS scores.
    """
    audio_waveform = audio["waveform"]
    sample_rate = 16000

    # Resample audio if necessary
    audio_resampled = librosa.resample(
        audio_waveform, 
        orig_sr=cfg["entrypoint"]["SAMPLE_RATE"], 
        target_sr=sample_rate
    )

    # Process each VAD segment
    for index, vad in enumerate(tqdm.tqdm(vad_list, desc="DNSMOS")):
        start, end = int(vad["start"] * sample_rate), int(vad["end"] * sample_rate)
        segment = audio_resampled[start:end]

        # Compute DNSMOS for this segment
        dnsmos_result = dnsmos_compute_score(segment, sample_rate, False)["OVRL"]

        vad_list[index]["dnsmos"] = dnsmos_result

    # Calculate average DNSMOS across all segments
    predict_dnsmos = np.mean([vad["dnsmos"] for vad in vad_list])

    print(f"Average predict_dnsmos for whole audio: {predict_dnsmos:.4f}")

    return predict_dnsmos, vad_list


def example_usage():
    """Example demonstrating DNSMOS with VAD"""
    
    # Example: Load audio file
    audio_path = "path/to/your/audio.wav"  # Update this
    
    print(f"\nProcessing: {audio_path}")
    
    # Load audio
    waveform, sr = librosa.load(audio_path, sr=cfg["entrypoint"]["SAMPLE_RATE"])
    
    audio_dict = {
        "waveform": waveform,
        "sample_rate": sr
    }
    
    # Example VAD list (normally you'd get this from a VAD model)
    # Format: list of dicts with 'start' and 'end' times in seconds
    vad_list = [
        {"start": 0.0, "end": 2.5},
        {"start": 2.5, "end": 5.0},
        {"start": 5.5, "end": 8.0},
    ]
    
    # Calculate DNSMOS
    avg_dnsmos, vad_with_scores = mos_prediction(audio_dict, vad_list)
    
    # Display results
    print("\n" + "="*60)
    print("DNSMOS Results:")
    print("="*60)
    print(f"Overall Average: {avg_dnsmos:.4f}")
    print("\nPer-segment scores:")
    for i, vad in enumerate(vad_with_scores):
        print(f"  Segment {i+1} ({vad['start']:.1f}s - {vad['end']:.1f}s): {vad['dnsmos']:.4f}")
    print("="*60)


def simple_example_without_vad():
    """Simple example without VAD segments"""
    
    audio_path = "path/to/your/audio.wav"  # Update this
    
    print(f"\nProcessing (no VAD): {audio_path}")
    
    # Direct computation without VAD
    # The ComputeScore callable handles file path or numpy array
    result = dnsmos_compute_score(audio_path, None, False)
    
    print("\n" + "="*60)
    print("DNSMOS Results (Full Audio):")
    print("="*60)
    print(f"Overall (OVRL): {result['OVRL']:.4f}")
    print(f"Signal (SIG):   {result['SIG']:.4f}")
    print(f"Background (BAK): {result['BAK']:.4f}")
    print(f"Audio length:   {result['len_in_sec']:.2f}s")
    print("="*60)


if __name__ == "__main__":
    print("DNSMOS with VAD Example")
    print("="*60)
    print("\nNote: Update the model path and audio path before running")
    print(f"Model path: {cfg['mos_model']['primary_model_path']}")
    print("\nDownload DNSMOS model from:")
    print("https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS")
    print("="*60)
    
    # Uncomment to run examples:
    # example_usage()
    # simple_example_without_vad()
