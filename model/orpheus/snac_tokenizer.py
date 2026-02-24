import os
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import Dataset, Features, Audio, Value
from transformers import AutoTokenizer
from snac import SNAC
from tqdm import tqdm
import warnings
import pandas as pd

# Suppress unnecessary warnings from torchaudio
warnings.filterwarnings("ignore", category=UserWarning, module='torchaudio')

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
INPUT_CSV = "/home/coder/datasets/crawl_datasets/final_dataset/train_clean.csv"  # Path to CSV file
AUDIO_BASE_DIR = "./"  # Base directory containing audio files (if CSV paths are relative)
OUTPUT_DIR = "./train_processed_snac_data_clean"  # Directory to save processed results
TOKENIZER_NAME = "unsloth/orpheus-3b-0.1-ft"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# List of speakers to process (leave empty or None to process all)
ALLOWED_SPEAKERS = {
    # Example:
    # "speaker_000",
    # "speaker_001",
    # Leave empty to process all speakers
}

# ==============================================================================
# 2. INITIALIZE MODEL AND TOKENIZER
# ==============================================================================
print(f"Using device: {DEVICE}")
print(f"Loading SNAC model: {SNAC_MODEL_NAME}...")
snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).to(DEVICE)
snac_model.eval()

print(f"Loading text tokenizer: {TOKENIZER_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
print("Initialization complete.")
print("-" * 50)
if ALLOWED_SPEAKERS:
    print(f"NOTE: Will only process samples from {len(ALLOWED_SPEAKERS)} specified speakers.")
else:
    print("NOTE: Will process ALL speakers in CSV.")
print("-" * 50)

# ==============================================================================
# 3. DEFINE SPECIAL TOKENS
# ==============================================================================
tokeniser_length = 128256
start_of_text = 128000
end_of_text = 128009
start_of_speech = tokeniser_length + 1
end_of_speech = tokeniser_length + 2
start_of_human = tokeniser_length + 3
end_of_human = tokeniser_length + 4
start_of_ai = tokeniser_length + 5
end_of_ai = tokeniser_length + 6
pad_token = tokeniser_length + 7
audio_tokens_start = tokeniser_length + 10

# ==============================================================================
# 4. PROCESSING FUNCTIONS
# ==============================================================================

def tokenise_audio(waveform, orig_freq):
    """Tokenize audio into SNAC codes"""
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    if orig_freq != 24000:
        resample_transform = T.Resample(orig_freq=orig_freq, new_freq=24000)
        waveform = resample_transform(waveform)
    waveform = waveform.unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        codes = snac_model.encode(waveform)
    all_codes = []
    base_offset = audio_tokens_start
    for i in range(codes[0].shape[1]):
        all_codes.append(codes[0][0][i].item() + base_offset)
        all_codes.append(codes[1][0][2*i].item() + base_offset + 4096)
        all_codes.append(codes[2][0][4*i].item() + base_offset + (2*4096))
        all_codes.append(codes[2][0][(4*i)+1].item() + base_offset + (3*4096))
        all_codes.append(codes[1][0][(2*i)+1].item() + base_offset + (4*4096))
        all_codes.append(codes[2][0][(4*i)+2].item() + base_offset + (5*4096))
        all_codes.append(codes[2][0][(4*i)+3].item() + base_offset + (6*4096))
    return all_codes

def remove_duplicate_frames(codes_list):
    """Remove duplicate audio frames"""
    if not codes_list or len(codes_list) % 7 != 0:
        return []
    result = codes_list[:7]
    for i in range(7, len(codes_list), 7):
        if codes_list[i] != result[-7]:
            result.extend(codes_list[i:i+7])
    return result

def load_audio_file(audio_path):
    """Load audio file and return waveform + sampling rate"""
    full_path = os.path.join(AUDIO_BASE_DIR, audio_path) if not os.path.isabs(audio_path) else audio_path
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Không tìm thấy file audio: {full_path}")
    
    waveform, sample_rate = torchaudio.load(full_path)
    # Chuyển về mono nếu là stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform.squeeze(0).numpy(), sample_rate

def process_batch(batch):
    """Process a batch of samples from CSV"""
    new_input_ids = []
    new_labels = []
    new_attention_masks = []

    for i in range(len(batch['text'])):
        try:
            speaker = batch["speaker"][i]
            audio_path = batch["audio_file"][i]
            text = batch["text"][i]

            # Filter by speaker if specified
            if ALLOWED_SPEAKERS and speaker not in ALLOWED_SPEAKERS:
                continue

            # Load audio file
            audio_array, sampling_rate = load_audio_file(audio_path)

            # Tokenize audio
            audio_codes = tokenise_audio(audio_array, sampling_rate)
            if not audio_codes:
                raise ValueError("Audio tokenization failed or audio is empty.")

            audio_codes = remove_duplicate_frames(audio_codes)
            if not audio_codes:
                raise ValueError("Audio has no content after removing duplicate frames.")

            # Tokenize text
            text_prompt = f"{speaker}: {text}"
            text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
            text_ids.append(end_of_text)

            # Create complete input_ids
            input_ids = (
                [start_of_human] + text_ids + [end_of_human] +
                [start_of_ai] + [start_of_speech] + audio_codes + [end_of_speech] + [end_of_ai]
            )

            new_input_ids.append(input_ids)
            new_labels.append(input_ids)
            new_attention_masks.append([1] * len(input_ids))

        except Exception as e:
            # Skip failed samples
            print(f"  Error processing sample {i}: {e}")
            continue

    return {
        "input_ids": new_input_ids,
        "labels": new_labels,
        "attention_mask": new_attention_masks
    }

# ==============================================================================
# 5. MAIN PROCESSING PIPELINE
# ==============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check CSV file
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: CSV file not found: {INPUT_CSV}")
        return

    print(f"Reading data from CSV: {INPUT_CSV}...")
    
    try:
        # Read CSV
        df = pd.read_csv(INPUT_CSV)
        
        # Check required columns
        required_columns = ['audio_file', 'text', 'speaker']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"ERROR: CSV is missing columns: {missing_columns}")
            return
        
        print(f"-> Found {len(df)} samples in CSV.")
        print(f"-> Columns: {list(df.columns)}")
        
        # Filter by speaker if specified
        if ALLOWED_SPEAKERS:
            df = df[df['speaker'].isin(ALLOWED_SPEAKERS)]
            print(f"-> After speaker filtering: {len(df)} samples.")
        
        if len(df) == 0:
            print("No samples to process after filtering.")
            return
        
        # Convert DataFrame to Hugging Face Dataset
        dataset_dict = {
            'audio_file': df['audio_file'].tolist(),
            'text': df['text'].tolist(),
            'speaker': df['speaker'].tolist()
        }
        
        raw_dataset = Dataset.from_dict(dataset_dict)
        original_count = len(raw_dataset)
        
        print(f"\nStarting to process {original_count} samples...")
        
        # Process dataset
        processed_dataset = raw_dataset.map(
            process_batch,
            batched=True,
            batch_size=32,
            remove_columns=raw_dataset.column_names,
            writer_batch_size=1000,
            load_from_cache_file=False,
            desc="Tokenizing audio and text"
        )
        
        processed_count = len(processed_dataset)
        filtered_count = original_count - processed_count
        
        print(f"\n-> Successfully processed: {processed_count}/{original_count} samples")
        if filtered_count > 0:
            print(f"-> Filtered out: {filtered_count} samples (due to errors or not in allowed speakers)")
        
        # Save results
        if processed_count > 0:
            output_path = os.path.join(OUTPUT_DIR, "processed_data.parquet")
            processed_dataset.to_parquet(output_path)
            print(f"\n✓ Processed data saved to: {output_path}")
            
            # Display statistics
            print("\n" + "="*50)
            print("STATISTICS:")
            print(f"  - Total input samples: {original_count}")
            print(f"  - Processed samples: {processed_count}")
            print(f"  - Success rate: {processed_count/original_count*100:.2f}%")
            print("="*50)
        else:
            print("\n⚠ No valid samples after processing.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\nComplete!")

if __name__ == "__main__":
    main()