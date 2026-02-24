import torch
import os
import math
import wandb
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- WANDB CONFIGURATION ---
os.environ["WANDB_PROJECT"] = "neu-khmer-tts_v2"
os.environ["WANDB_RUN_NAME"] = "neu-air-khmer"

# --- PATH CONFIGURATION ---
BASE_MODEL_WITH_TOKENIZER_PATH = "/home/coder/data/Speech/TTS/neutts-air/neuttsair/final_khmer_full_v2/checkpoint-5722"
MODEL_CHECKPOINT_PATH = "/home/coder/data/Speech/TTS/neutts-air/neutts-air"
PRETRAINING_CHUNKS_DIR_1 = "/home/coder/datasets/crawl_datasets/final_dataset/neu_codec_clean_parquet"
TRAINING_OUTPUT_DIR = "./final_khmer_full_v2"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=MODEL_CHECKPOINT_PATH,
        metadata={"help": "Path to model checkpoint to continue training from."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=BASE_MODEL_WITH_TOKENIZER_PATH,
        metadata={"help": "Path to tokenizer."},
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16",
        metadata={"help": "Torch data type: auto, bfloat16, float16, float32."},
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "flash_attention_2 or sdpa."},
    )


@dataclass
class DataArguments:
    dataset_paths: List[str] = field(
        default_factory=lambda: [PRETRAINING_CHUNKS_DIR_1],
        metadata={"help": "List of paths to directories containing .parquet data files."},
    )


def main():
    # 1. PARSE ARGUMENTS
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        args=[
            f"--output_dir={TRAINING_OUTPUT_DIR}",
            "--do_train",
            "--per_device_train_batch_size=1",
            "--gradient_accumulation_steps=16",  # Increase to maintain effective batch size
            "--learning_rate=4e-5",
            "--num_train_epochs=2",
            "--logging_steps=5",
            "--save_steps=1000",
            "--warmup_ratio=0.03",
            "--lr_scheduler_type=cosine",
            "--report_to=wandb",
            "--bf16",
            "--dataloader_num_workers=0",
            "--gradient_checkpointing",  # Significantly reduces memory usage
            "--optim=adamw_torch_fused",  # More efficient optimizer
            "--max_grad_norm=1.0",  # Prevent gradient explosion
            # "--max_steps=50000",  # You can uncomment and set value if you want to manually specify
        ]
    )

    # 2. CHECK PATHS
    model_path = Path(model_args.model_name_or_path)
    tokenizer_path_str = model_args.tokenizer_name_or_path or model_args.model_name_or_path
    tokenizer_path = Path(tokenizer_path_str)

    if not model_path.exists():
        raise FileNotFoundError(f"Lỗi: Thư mục model checkpoint '{model_path}' không tồn tại.")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Lỗi: Thư mục tokenizer '{tokenizer_path}' không tồn tại.")

    for path_str in data_args.dataset_paths:
        data_path = Path(path_str)
        if not data_path.exists():
            raise FileNotFoundError(f"Lỗi: Thư mục dữ liệu '{data_path}' không tồn tại.")
        if not any(data_path.glob("*.parquet")):
            print(f"Cảnh báo: Thư mục dữ liệu '{data_path}' không chứa bất kỳ file .parquet nào.")

    # 3. LOAD TOKENIZER & MODEL
    print(f"--- Loading tokenizer from '{tokenizer_path_str}' ---")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_str, trust_remote_code=True)
    print("Tokenizer loaded successfully.")

    print(f"--- Loading model from '{model_args.model_name_or_path}' ---")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        trust_remote_code=True,
        use_cache=True,  # Enable cache to speed up inference
    )
    
    # Enable gradient checkpointing if model supports it
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")
    
    print("Model loaded successfully.")

    print(f"Model vocabulary size: {model.get_input_embeddings().weight.shape[0]}")
    print(f"Tokenizer size: {len(tokenizer)}")
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise ValueError("Model and tokenizer vocabulary sizes do not match!")

    # 4. COLLECT PARQUET FILES
    all_parquet_files = []
    for path_str in data_args.dataset_paths:
        data_path = Path(path_str)
        files_in_dir = [str(p) for p in data_path.glob("*.parquet")]
        all_parquet_files.extend(files_in_dir)

    if not all_parquet_files:
        raise FileNotFoundError("No .parquet files found in any directory.")

    print(f"Found {len(all_parquet_files)} parquet files.")

    # 5. ATTEMPT TO CALCULATE TOTAL NUMBER OF SAMPLES (NUM_SAMPLES) WITH ONE NON-STREAMING LOAD
    num_samples = None
    try:
        print("Attempting to load dataset temporarily (non-streaming) to count samples (may be slow)...")
        temp_ds = load_dataset("parquet", data_files={"train": all_parquet_files}, split="train")
        num_samples = len(temp_ds)
        del temp_ds
        print(f"Total number of samples: {num_samples}")
    except Exception as e:
        print("Cannot load dataset in non-streaming mode for counting (dataset too large or error).")
        print("Counting error:", e)
        num_samples = None

    # If num_samples was calculated, set max_steps based on batch configuration
    if num_samples is not None:
        total_train_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
        if total_train_batch_size == 0:
            raise ValueError("Total train batch size is zero — check per_device_train_batch_size and gradient_accumulation_steps.")
        num_update_steps_per_epoch = math.ceil(num_samples / total_train_batch_size)
        training_args.max_steps = int(num_update_steps_per_epoch * training_args.num_train_epochs)
        print(f"Total batch size (per_device * grad_accum * world_size): {total_train_batch_size}")
        print(f"Number of update steps per epoch (estimated): {num_update_steps_per_epoch}")
        print(f"⇒ Setting training_args.max_steps = {training_args.max_steps}")
    else:
        # If counting failed, check if user passed max_steps via CLI
        if getattr(training_args, "max_steps", 0) > 0:
            print(f"max_steps already set from args: {training_args.max_steps}")
        else:
            raise ValueError(
                "train_dataset does not support __len__ (streaming). You must specify `--max_steps` when running the script "
                "or allow the script to load dataset in non-streaming mode to auto-calculate. Example: "
                "python multi_gpu_train.py --max_steps 50000 ..."
            )

    # 6. LOAD STREAMING DATASET (for actual training)
    train_dataset = load_dataset(
        "parquet",
        data_files={"train": all_parquet_files},
        split="train",
        streaming=True,
    )
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=training_args.seed)
    
    # Filter to keep only necessary columns for model (input_ids, attention_mask, labels)
    def preprocess_function(examples):
        # Keep only fields that model needs
        result = {}
        if "input_ids" in examples:
            result["input_ids"] = examples["input_ids"]
        if "attention_mask" in examples:
            result["attention_mask"] = examples["attention_mask"]
        if "labels" in examples:
            result["labels"] = examples["labels"]
        elif "input_ids" in examples:
            # If no labels, use input_ids as labels (causal LM)
            result["labels"] = examples["input_ids"]
        return result
    
    train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
    print("Data loaded in streaming mode and columns filtered.")

    # 7. Custom data collator for TTS model
    def custom_data_collator(features):
        # Get maximum length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Manual padding
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for f in features:
            input_ids = f["input_ids"]
            attention_mask = f.get("attention_mask", [1] * len(input_ids))
            labels = f.get("labels", input_ids)
            
            # Padding
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # -100 will be ignored in loss
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }
    
    data_collator = custom_data_collator

    class StreamingTrainer(Trainer):
        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: train_dataset is not set")
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                drop_last=True,
            )

    print("--- Initializing StreamingTrainer ---")
    trainer = StreamingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # 8. TRAIN
    print("--- Starting Continued Pre-Training (CTP, streaming) ---")
    train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # 9. SAVE
    print("--- Training completed. Saving final model. ---")
    trainer.save_model()
    trainer.save_state()
    print(f"Model saved to: {training_args.output_dir}")


if __name__ == "__main__":
    main()
