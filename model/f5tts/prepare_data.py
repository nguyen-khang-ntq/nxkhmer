import concurrent.futures
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import argparse
import csv
import json
from tqdm import tqdm

# Configuration constants
BATCH_SIZE = 1000
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
THREAD_NAME_PREFIX = "AudioProcessor"
CHUNK_SIZE = 1000

executor = None


@contextmanager
def graceful_exit():
    """Context manager for graceful shutdown on signals"""

    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Cleaning up...")
        if executor is not None:
            print("Shutting down executor...")
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        if executor is not None:
            executor.shutdown(wait=False)


def process_audio_file(audio_path, text, speaker):
    """Process a single audio file by checking its existence and extracting duration."""
    if not Path(audio_path).exists():
        print(f"audio {audio_path} not found, skipping")
        return None
    try:
        audio_duration = get_audio_duration(audio_path)
        if audio_duration <= 0:
            raise ValueError(f"Duration {audio_duration} is non-positive.")
        return (audio_path, text, speaker, audio_duration)
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping corrupt file.")
        return None


def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffmpeg's ffprobe.
    Falls back to loading with subprocess if ffprobe fails.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError) as e:
        print(f"Warning: ffprobe failed for {audio_path} with error: {e}.")
        raise RuntimeError(f"ffprobe failed for {audio_path}: {e}")


def read_audio_text_speaker_csv(csv_file_path):
    """Read CSV file with format: audio_file,text,speaker or audio_file|text"""
    audio_text_speaker_list = []
    
    parent = Path(csv_file_path).parent
    with open(csv_file_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
        # Read first line to detect delimiter
        first_line = csvfile.readline()
        csvfile.seek(0)
        
        # Detect delimiter and check for speaker column
        if ',' in first_line and 'speaker' in first_line.lower():
            # Format: audio_file,text,speaker
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                audio_file = row.get("audio_file", "").strip()
                text = row.get("text", "").strip()
                speaker = row.get("speaker", "").strip() or "default_speaker"
                
                if audio_file and text:
                    audio_file_path = parent / audio_file
                    audio_text_speaker_list.append((audio_file_path.as_posix(), text, speaker))
        else:
            # Format: audio_file|text (original format)
            reader = csv.reader(csvfile, delimiter='|')
            next(reader, None)  # Skip header if exists
            for row in reader:
                if len(row) >= 2:
                    audio_file = row[0].strip()
                    text = row[1].strip()
                    speaker = "default_speaker"
                    
                    if audio_file and text:
                        audio_file_path = parent / audio_file
                        audio_text_speaker_list.append((audio_file_path.as_posix(), text, speaker))
    
    return audio_text_speaker_list


def get_csv_path(input_path):
    """Get CSV path from input (can be directory or specific CSV file)"""
    input_path = Path(input_path)
    
    if input_path.is_file() and input_path.suffix == '.csv':
        # Input is CSV file directly
        return input_path
    elif input_path.is_dir():
        # Input is directory, look for metadata_train_all.csv or metadata.csv
        for csv_name in ['metadata_train_all.csv', 'train.csv', 'metadata.csv']:
            csv_path = input_path / csv_name
            if csv_path.exists():
                return csv_path
        raise FileNotFoundError(f"No CSV file found in {input_path}. Looking for: metadata_train_all.csv, train.csv, or metadata.csv")
    else:
        raise ValueError(f"Input must be a CSV file or directory: {input_path}")


def prepare_khmer_dataset_from_csv(csv_path, num_workers=None):
    """Process the Khmer dataset from CSV file"""
    global executor
    
    audio_text_speaker_list = read_audio_text_speaker_csv(csv_path.as_posix())
    
    total_files = len(audio_text_speaker_list)
    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")
    
    with graceful_exit():
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            results = []
            
            for i in range(0, len(audio_text_speaker_list), CHUNK_SIZE):
                chunk = audio_text_speaker_list[i : i + CHUNK_SIZE]
                chunk_futures = [
                    executor.submit(process_audio_file, item[0], item[1], item[2]) 
                    for item in chunk
                ]
                
                for future in tqdm(
                    chunk_futures,
                    total=len(chunk),
                    desc=f"Processing chunk {i // CHUNK_SIZE + 1}/{(total_files + CHUNK_SIZE - 1) // CHUNK_SIZE}",
                ):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing file: {e}")
            
            executor = None
    
    processed = [res for res in results if res is not None]
    if not processed:
        raise RuntimeError("No valid audio files were processed!")
    
    # Prepare final results
    sub_result = []
    durations = []
    speakers_set = set()
    vocab_set = set()
    
    for audio_path, text, speaker, duration in processed:
        sub_result.append({
            "audio_path": audio_path,
            "text": text,
            "speaker": speaker,
            "duration": duration
        })
        durations.append(duration)
        speakers_set.add(speaker)
        vocab_set.update(list(text))
    
    return sub_result, durations, speakers_set, vocab_set


def save_prepped_dataset(out_dir, result, duration_list, speakers_set, vocab_set):
    """Save the prepared dataset to output directory"""
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir} ...")
    
    # Save processed data as JSON
    data_json_path = out_dir / "processed_data.json"
    with open(data_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Save durations to JSON
    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path.as_posix(), "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)
    
    # Save speakers list
    speakers_path = out_dir / "speakers.txt"
    with open(speakers_path.as_posix(), "w", encoding="utf-8") as f:
        for speaker in sorted(speakers_set):
            f.write(speaker + "\n")
    
    # Save vocabulary
    vocab_path = out_dir / "vocab.txt"
    with open(vocab_path.as_posix(), "w", encoding="utf-8") as f:
        for char in sorted(vocab_set):
            f.write(char + "\n")
    
    # Save metadata CSV for reference
    metadata_path = out_dir / "metadata_processed.csv"
    with open(metadata_path.as_posix(), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "text", "speaker", "duration"])
        for item in result:
            writer.writerow([item["audio_path"], item["text"], item["speaker"], item["duration"]])
    
    dataset_name = out_dir.stem
    print(f"\nDataset statistics for {dataset_name}:")
    print(f"  - Total samples: {len(result)}")
    print(f"  - Number of speakers: {len(speakers_set)}")
    print(f"  - Vocabulary size: {len(vocab_set)}")
    print(f"  - Total duration: {sum(duration_list) / 3600:.2f} hours")
    print(f"  - Average duration: {sum(duration_list) / len(duration_list):.2f} seconds")
    print(f"  - Min duration: {min(duration_list):.2f} seconds")
    print(f"  - Max duration: {max(duration_list):.2f} seconds")
    print(f"\nSpeakers: {', '.join(sorted(speakers_set))}")


def prepare_and_save(input_path, out_dir, num_workers=None):
    """Main function to prepare and save dataset"""
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe is not available. Please install ffmpeg.")
        return
    
    # Get CSV path (flexible input)
    csv_path = get_csv_path(input_path)
    print(f"Using CSV file: {csv_path}")
    
    sub_result, durations, speakers_set, vocab_set = prepare_khmer_dataset_from_csv(csv_path, num_workers=num_workers)
    save_prepped_dataset(out_dir, sub_result, durations, speakers_set, vocab_set)


def cli():
    try:
        parser = argparse.ArgumentParser(
            description="Prepare Khmer speech dataset from CSV file or directory",
            epilog="""
Examples:
    # Using directory (will auto-detect CSV file):
    python prepare_khmer.py /path/to/dataset ./output
    
    # Using specific CSV file:
    python prepare_khmer.py /path/to/train.csv ./output
    python prepare_khmer.py /path/to/metadata_train_all.csv ./output
    
    # With custom worker count:
    python prepare_khmer.py /path/to/dataset ./output --workers 4

Input format:
    - Can be a directory (will look for: metadata_train_all.csv, train.csv, or metadata.csv)
    - Can be a specific CSV file path
    
CSV format support:
    - Format 1: audio_file,text,speaker (comma-separated with speaker column)
    - Format 2: audio_file|text (pipe-separated, original F5-TTS format)
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument("--input", type=str, help="Input CSV file or directory")
        parser.add_argument("--out_dir", type=str, help="Output directory to save the prepared data")
        parser.add_argument("--workers", type=int, help=f"Number of worker threads (default: {MAX_WORKERS})")
        args = parser.parse_args()
        
        prepare_and_save(args.input, args.out_dir, num_workers=args.workers)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Cleaning up...")
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
