"""
Script to prepare training data with GPU batch processing for faster encoding.
Reads from CSV file and outputs multiple parquet files (5k samples each).
CSV format: audio_file, text, speaker
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import librosa
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import warnings
import math
from neucodec import NeuCodec, DistillNeuCodec
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer

try:
    from khmerphonemizer import phonemize as khmer_phonemize
    KHMER_PHONEMIZER_AVAILABLE = True
except ImportError:
    KHMER_PHONEMIZER_AVAILABLE = False
    print("⚠️  khmerphonemizer not installed. Install with: pip install khmerphonemizer")


class TTSDatasetPreprocessor:
    """Preprocessor for creating TTS training data with tokenized inputs."""

    def __init__(
        self,
        tokenizer_repo: str = "neuphonic/neutts-air",
        codec_repo: str = "neuphonic/neucodec",
        codec_device: str = "cuda",
        max_context: int = 2048*2,
        target_sample_rate: int = 16000,
        batch_size: int = 4,
        language: str = "km",  # NEW: language parameter for phonemizer
        use_khmer_phonemizer: bool = True  # NEW: flag to use khmer_phonemizer
    ):
        """
        Initialize the preprocessor.

        Args:
            tokenizer_repo: Repository for the tokenizer
            codec_repo: Repository or path for the codec model
            codec_device: Device to run codec on ('cpu' or 'cuda')
            max_context: Maximum sequence length
            target_sample_rate: Target sample rate for audio (default: 16000 for codec)
            batch_size: Number of audio files to encode in parallel on GPU
            language: Language code for phonemizer (default: 'km' for Khmer)
            use_khmer_phonemizer: Use khmer_phonemizer instead of espeak (default: True)
        """
        self.target_sample_rate = target_sample_rate
        self.max_context = max_context
        self.batch_size = batch_size
        self.codec_device = codec_device
        self.language = language
        self.use_khmer_phonemizer = use_khmer_phonemizer
        self._is_onnx_codec = False

        # Load tokenizer
        print(f"Loading tokenizer from: {tokenizer_repo}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo)
        # Set model_max_length to avoid warnings about long sequences
        if self.tokenizer.model_max_length > 1000000:  # Some tokenizers have default 1e30
            self.tokenizer.model_max_length = max_context

        # Load phonemizer based on language and preference
        if language == "km" and use_khmer_phonemizer:
            if not KHMER_PHONEMIZER_AVAILABLE:
                raise ImportError(
                    "khmerphonemizer is not installed. "
                    "Install it with: pip install khmerphonemizer"
                )
            print(f"Using khmerphonemizer for Khmer language")
            self.phonemizer = None  # We'll use the function directly
        else:
            print(f"Loading phonemizer for language: {language}...")
            self.phonemizer = EspeakBackend(
                language=language,
                preserve_punctuation=True,
                with_stress=True
            )

        # Load codec
        self._load_codec(codec_repo, codec_device)

    def _load_codec(self, codec_repo: str, codec_device: str):
        """Load the codec model."""
        print(f"Loading codec from: {codec_repo} on {codec_device}...")

        if codec_repo.endswith(".onnx") and Path(codec_repo).is_file():
            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import NeuCodecOnnxDecoder. "
                    "Make sure `neucodec` and `onnxruntime` are installed."
                ) from e

            self.codec = NeuCodecOnnxDecoder(codec_repo)
            self._is_onnx_codec = True

        elif codec_repo == "neuphonic/neucodec":
            self.codec = NeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)

        elif codec_repo == "neuphonic/distill-neucodec":
            self.codec = DistillNeuCodec.from_pretrained(codec_repo)
            self.codec.eval().to(codec_device)

        elif codec_repo == "neuphonic/neucodec-onnx-decoder":
            if codec_device != "cpu":
                raise ValueError("ONNX decoder only runs on CPU.")

            try:
                from neucodec import NeuCodecOnnxDecoder
            except ImportError as e:
                raise ImportError(
                    "Failed to import ONNX decoder. "
                    "Ensure you have onnxruntime and neucodec >= 0.0.4."
                ) from e

            self.codec = NeuCodecOnnxDecoder.from_pretrained(codec_repo)
            self._is_onnx_codec = True

        else:
            raise ValueError(
                f"Invalid codec repo: {codec_repo}. Must be one of: "
                "'neuphonic/neucodec', 'neuphonic/distill-neucodec', "
                "'neuphonic/neucodec-onnx-decoder', or a local .onnx file."
            )

    def _resample_audio(self, audio_data: np.ndarray, original_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            audio_data: Audio waveform
            original_sr: Original sample rate

        Returns:
            Resampled audio at target_sample_rate
        """
        if original_sr != self.target_sample_rate:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=original_sr,
                target_sr=self.target_sample_rate
            )
        return audio_data

    def _load_audio_batch(self, audio_paths: List[Path]) -> List[torch.Tensor]:
        """
        Load and preprocess a batch of audio files.

        Args:
            audio_paths: List of audio file paths

        Returns:
            List of audio tensors
        """
        audio_tensors = []
        for audio_path in audio_paths:
            # Load audio with original sample rate
            wav, sr = librosa.load(audio_path, sr=None, mono=True)

            # Resample to target sample rate
            wav = self._resample_audio(wav, sr)

            # Convert to tensor
            wav_tensor = torch.from_numpy(wav).float()
            audio_tensors.append(wav_tensor)

        return audio_tensors

    def _encode_audio_batch(self, audio_tensors: List[torch.Tensor]) -> List[np.ndarray]:
        """
        Encode a batch of audio tensors to codec codes.

        Args:
            audio_tensors: List of audio tensors

        Returns:
            List of codec codes as numpy arrays
        """
        outputs = []

        for wav_tensor in audio_tensors:
            # Add batch and channel dimensions [1, 1, T]
            # Keep on CPU - neucodec's feature extractor requires CPU tensors
            wav_tensor = wav_tensor.cpu().unsqueeze(0).unsqueeze(0)

            # Encode (codec internally needs CPU tensors for feature extraction)
            with torch.no_grad():
                codes = self.codec.encode_code(audio_or_path=wav_tensor)

            # Remove batch/channel dims and convert to numpy
            codes = codes.squeeze(0).squeeze(0)
            if isinstance(codes, torch.Tensor):
                codes = codes.cpu().numpy()

            outputs.append(codes)

            # Clean up
            del wav_tensor
            torch.cuda.empty_cache()

        return outputs

    def _encode_audio(self, audio_path: str | Path) -> np.ndarray:
        """
        Encode single audio file to codec codes (fallback for ONNX or single files).

        Args:
            audio_path: Path to audio file

        Returns:
            Codec codes as numpy array
        """
        # Load audio with original sample rate
        wav, sr = librosa.load(audio_path, sr=None, mono=True)

        # Resample to target sample rate
        wav = self._resample_audio(wav, sr)

        # Convert to tensor [1, 1, T]
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)

        if not self._is_onnx_codec:
            wav_tensor = wav_tensor.to(self.codec_device)

        # Encode
        with torch.no_grad():
            ref_codes = self.codec.encode_code(
                audio_or_path=wav_tensor
            ).squeeze(0).squeeze(0)

        if isinstance(ref_codes, torch.Tensor):
            ref_codes = ref_codes.cpu().numpy()

        return ref_codes

    def _to_phones(self, text: str) -> str:
        """Convert text to phonemes using the configured language."""
        if not text or not text.strip():
            return ""
            
        if self.language == "km" and self.use_khmer_phonemizer:
            # Use khmer_phonemizer
            try:
                # Returns: (words, phonemes) where phonemes is list of list of phones
                words, phonemes = khmer_phonemize(text)
                
                # Flatten phonemes and join with spaces
                # Each word's phonemes are joined together, then words are separated by spaces
                phones_str = " ".join([
                    "".join(phone_list) for phone_list in phonemes
                ])
                return phones_str
            except Exception as e:
                warnings.warn(f"Khmer phonemization failed for text: {text[:50]}... Error: {e}")
                return ""
        else:
            # Use espeak backend
            try:
                phones = self.phonemizer.phonemize([text])
                if not phones or not phones[0]:
                    return ""
                phones = phones[0].split()
                phones = " ".join(phones)
                return phones
            except Exception as e:
                warnings.warn(f"Phonemization failed for text: {text[:50]}... Error: {e}")
                return ""

    def _apply_chat_template(
        self,
        ref_codes: np.ndarray,
        text: str
    ) -> Dict[str, List[int]]:
        """
        Apply chat template and create input_ids, labels, attention_mask.
        Optimized version for faster preprocessing.

        Args:
            ref_codes: Reference audio codec codes
            text: Text to generate speech for

        Returns:
            Dict with 'input_ids', 'labels', 'attention_mask' or None if sequence is too long
        """
        # Get special token IDs
        speech_gen_start = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")
        speech_gen_end = self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")
        ignore_index = -100  # Standard ignore index for loss computation

        # Convert text to phonemes
        phones = self._to_phones(text)

        # Create codes string
        codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes])

        # Create chat format directly (matching neutts inference format)
        # NOTE: Do NOT use tokenizer.apply_chat_template() here!
        # The model was trained with this simple format, not with <|im_start|>/<|im_end|> tokens
        chat = (
            f"user: Convert the text to speech:<|TEXT_PROMPT_START|>{phones}<|TEXT_PROMPT_END|>\n"
            f"assistant:<|SPEECH_GENERATION_START|>{codes_str}<|SPEECH_GENERATION_END|>"
        )

        # Encode chat
        # Note: add_special_tokens defaults to True, which is what we want
        # But we need to disable truncation to handle long sequences ourselves
        ids = self.tokenizer.encode(chat)

        # Check if sequence is too long (skip if it is)
        if len(ids) > self.max_context:
            return None

        # Pad to max_context (all three should have same length)
        if len(ids) < self.max_context:
            pad_length = self.max_context - len(ids)
            ids = ids + [self.tokenizer.pad_token_id] * pad_length

        # Convert to tensor
        input_ids = torch.tensor(ids, dtype=torch.long)

        # Create labels (mask everything before speech generation)
        # Labels should have the SAME length as input_ids
        labels = torch.full_like(input_ids, ignore_index)
        speech_gen_start_idx = (input_ids == speech_gen_start).nonzero(as_tuple=True)[0]
        
        if len(speech_gen_start_idx) > 0:
            speech_gen_start_idx = speech_gen_start_idx[0]
            # Copy from speech_gen_start to the end (including the END token if present)
            labels[speech_gen_start_idx:] = input_ids[speech_gen_start_idx:]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Verify all have same length
        assert len(input_ids) == len(labels) == len(attention_mask) == self.max_context, \
            f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}, attention_mask={len(attention_mask)}"

        return {
            'input_ids': input_ids.tolist(),
            'labels': labels.tolist(),
            'attention_mask': attention_mask.tolist()
        }

    def process_csv_row(
        self,
        audio_path: str | Path,
        text: str,
        speaker: str,
        codes: np.ndarray,
    ) -> Dict | None:
        """
        Process a single CSV row with pre-encoded codes.

        Args:
            audio_path: Path to audio file
            text: Text content
            speaker: Speaker ID
            codes: Pre-encoded audio codes

        Returns:
            Dict with input_ids, labels, attention_mask, and metadata or None if failed
        """
        # Apply chat template
        result = self._apply_chat_template(codes, text)
        
        # Return None if sequence was too long
        if result is None:
            return None

        # Add metadata
        result['audio_file'] = str(audio_path)
        result['text'] = text
        result['speaker'] = speaker
        result['num_codes'] = len(codes)

        return result

    def process_csv(
        self,
        csv_path: str | Path,
        output_dir: str | Path,
        audio_base_dir: Optional[str | Path] = None,
        samples_per_file: int = 50,
        skip_errors: bool = True,
        debug: bool = False
    ):
        """
        Process CSV file and create multiple parquet files with GPU batch processing.

        Args:
            csv_path: Path to input CSV file
            output_dir: Directory to save parquet files
            audio_base_dir: Base directory for audio files (if CSV has relative paths)
            samples_per_file: Number of samples per parquet file
            skip_errors: Whether to skip failed samples or raise error
            debug: Print detailed error information
        """
        # Read CSV
        print(f"Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Validate columns
        required_cols = ['audio_file', 'text', 'speaker']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        print(f"Total samples: {len(df)}")
        print(f"Using device: {self.codec_device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Language: {self.language}")
        if self.language == "km" and self.use_khmer_phonemizer:
            print(f"Phonemizer: khmer_phonemizer")
        else:
            print(f"Phonemizer: espeak ({self.language})")

        # Setup paths
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if audio_base_dir:
            audio_base_dir = Path(audio_base_dir)

        # Calculate number of output files
        num_files = math.ceil(len(df) / samples_per_file)
        print(f"Will create {num_files} parquet files ({samples_per_file} samples each)")

        # Process in chunks
        for file_idx in range(num_files):
            start_idx = file_idx * samples_per_file
            end_idx = min(start_idx + samples_per_file, len(df))
            chunk_df = df.iloc[start_idx:end_idx]

            print(f"\nProcessing file {file_idx + 1}/{num_files} (samples {start_idx}-{end_idx})...")

            results = []
            failed_count = 0
            error_summary = {}  # Track error types
            
            # Track progress
            processed_in_chunk = 0
            total_in_chunk = len(chunk_df)

            # Process in batches
            pbar = tqdm(range(0, len(chunk_df), self.batch_size), desc=f"File {file_idx + 1}/{num_files}")
            for batch_start in pbar:
                batch_end = min(batch_start + self.batch_size, len(chunk_df))
                batch_rows = chunk_df.iloc[batch_start:batch_end]

                # Collect audio paths for this batch
                batch_audio_paths = []
                batch_metadata = []

                for idx, row in batch_rows.iterrows():
                    audio_path = Path(row['audio_file'])
                    if audio_base_dir and not audio_path.is_absolute():
                        audio_path = audio_base_dir / audio_path

                    if audio_path.exists():
                        batch_audio_paths.append(audio_path)
                        batch_metadata.append({
                            'text': row['text'],
                            'speaker': row['speaker'],
                            'audio_path': audio_path,
                            'idx': idx
                        })
                    else:
                        failed_count += 1
                        error_type = "FileNotFound"
                        error_summary[error_type] = error_summary.get(error_type, 0) + 1
                        if debug:
                            print(f"\n❌ Row {idx}: Audio file not found: {audio_path}")
                        if not skip_errors:
                            raise FileNotFoundError(f"Audio file not found: {audio_path}")

                if not batch_audio_paths:
                    continue

                try:
                    # Load audio files
                    audio_tensors = self._load_audio_batch(batch_audio_paths)

                    # Encode batch on GPU
                    codes_batch = self._encode_audio_batch(audio_tensors)

                    # Clean up audio tensors immediately
                    del audio_tensors

                    # Process each sample in the batch
                    for codes, metadata in zip(codes_batch, batch_metadata):
                        try:
                            result = self.process_csv_row(
                                audio_path=metadata['audio_path'],
                                text=metadata['text'],
                                speaker=metadata['speaker'],
                                codes=codes
                            )
                            
                            # Only append if result is not None (i.e., sequence wasn't too long)
                            if result is not None:
                                results.append(result)
                                processed_in_chunk += 1
                            else:
                                failed_count += 1
                                error_type = "SequenceTooLong"
                                error_summary[error_type] = error_summary.get(error_type, 0) + 1

                        except Exception as e:
                            failed_count += 1
                            error_type = type(e).__name__
                            error_summary[error_type] = error_summary.get(error_type, 0) + 1
                            if debug:
                                print(f"\n❌ Row {metadata['idx']} processing failed:")
                                print(f"   Audio: {metadata['audio_path']}")
                                print(f"   Text: {metadata['text'][:50]}...")
                                print(f"   Error: {type(e).__name__}: {str(e)}")
                            if skip_errors:
                                warnings.warn(f"Failed to process row {metadata['idx']}: {e}")
                            else:
                                raise

                    # Clean up codes batch
                    del codes_batch
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'success': processed_in_chunk,
                        'failed': failed_count,
                        'total': total_in_chunk
                    })

                except Exception as e:
                    failed_count += len(batch_audio_paths)
                    error_type = type(e).__name__
                    error_summary[error_type] = error_summary.get(error_type, 0) + len(batch_audio_paths)
                    if debug:
                        print(f"\n❌ Batch encoding failed:")
                        print(f"   Batch size: {len(batch_audio_paths)}")
                        print(f"   First file: {batch_audio_paths[0]}")
                        print(f"   Error: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    if skip_errors:
                        warnings.warn(f"Failed to process batch: {e}")
                    else:
                        raise

                # Periodic GPU memory cleanup
                if torch.cuda.is_available() and (batch_start // self.batch_size) % 10 == 0:
                    torch.cuda.empty_cache()

            # Save to parquet
            if results:
                result_df = pd.DataFrame(results)
                output_file = output_dir / f"train_{file_idx:04d}.parquet"
                result_df.to_parquet(output_file, index=False)
                print(f"✅ Saved {len(result_df)} samples to: {output_file}")
                if failed_count > 0:
                    print(f"⚠️  Skipped {failed_count} failed samples")
                    if error_summary:
                        print(f"   Error breakdown:")
                        for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
                            print(f"     {error_type}: {count}")
            else:
                print(f"❌ No successful samples in chunk {file_idx + 1}")
                if error_summary:
                    print(f"   Error breakdown:")
                    for error_type, count in sorted(error_summary.items(), key=lambda x: -x[1]):
                        print(f"     {error_type}: {count}")
                if debug:
                    print(f"   Enable --debug flag to see detailed errors")

        print(f"\n✅ Processing complete! Output saved to: {output_dir}")

        # Print summary
        parquet_files = list(output_dir.glob("train_*.parquet"))
        total_samples = sum(len(pd.read_parquet(f)) for f in parquet_files)
        print(f"\nSummary:")
        print(f"  Total parquet files: {len(parquet_files)}")
        print(f"  Total processed samples: {total_samples}")
        print(f"  Failed samples: {len(df) - total_samples}")


def main():
    """CLI for batch processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TTS Dataset Preprocessor - Convert CSV to Parquet files with GPU acceleration"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="Input CSV file path (columns: audio_file, text, speaker)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for parquet files"
    )
    parser.add_argument(
        "--audio_base_dir",
        type=str,
        default=None,
        help="Base directory for audio files (if CSV has relative paths)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="km",
        help="Language code for phonemizer (default: km for Khmer)"
    )
    parser.add_argument(
        "--use_espeak",
        action="store_true",
        help="Use espeak instead of khmer_phonemizer for Khmer (default: use khmer_phonemizer)"
    )
    parser.add_argument(
        "--samples_per_file",
        type=int,
        default=50,
        help="Number of samples per parquet file (default: 5000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for GPU encoding (default: 8)"
    )
    parser.add_argument(
        "--tokenizer_repo",
        type=str,
        default="neuphonic/neutts-air",
        help="Tokenizer repository"
    )
    parser.add_argument(
        "--codec_repo",
        type=str,
        default="neuphonic/neucodec",
        help="Codec repository or path"
    )
    parser.add_argument(
        "--codec_device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for codec (default: cuda)"
    )
    parser.add_argument(
        "--max_context",
        type=int,
        default=2048*2,
        help="Maximum sequence length (default: 4096)"
    )
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=16000,
        help="Target sample rate for audio (default: 16000)"
    )
    parser.add_argument(
        "--skip_errors",
        action="store_true",
        help="Skip failed samples instead of stopping"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed error information for debugging"
    )

    args = parser.parse_args()

    # Check CUDA availability
    if args.codec_device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available! Falling back to CPU.")
        args.codec_device = "cpu"

    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = TTSDatasetPreprocessor(
        tokenizer_repo=args.tokenizer_repo,
        codec_repo=args.codec_repo,
        codec_device=args.codec_device,
        max_context=args.max_context,
        target_sample_rate=args.target_sample_rate,
        batch_size=args.batch_size,
        language=args.language,
        use_khmer_phonemizer=not args.use_espeak  # Invert the flag
    )

    # Process CSV
    preprocessor.process_csv(
        csv_path=args.csv_file,
        output_dir=args.output_dir,
        audio_base_dir=args.audio_base_dir,
        samples_per_file=args.samples_per_file,
        skip_errors=args.skip_errors,
        debug=args.debug
    )


if __name__ == "__main__":
    main()