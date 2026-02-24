"""
Train single-speaker VITS model with the speaker that has the most data.
This script is optimized for single-speaker TTS training.
"""
import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# ================= CONFIG PATHS =================
output_path = os.path.dirname(os.path.abspath(__file__))

# Dataset config for single speaker
dataset_config = BaseDatasetConfig(
    formatter="ntq_khmer",
    meta_file_train="/home/coder/datasets/khmer_audio_datasets/km_kh_male/train.csv",
    meta_file_val="/home/coder/datasets/khmer_audio_datasets/km_kh_male/val.csv",
    path='/home/coder/datasets/khmer_audio_datasets/km_kh_male/'
)

# ================= AUDIO CONFIG =================
audio_config = VitsAudioConfig(
    sample_rate=22050,
    resample=True,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

# ================= MODEL ARGS (SINGLE SPEAKER) =================
vitsArgs = VitsArgs(
    use_language_embedding=False,
    embedded_language_dim=0,
    use_speaker_embedding=False,  # Disable speaker embedding for single speaker
    use_sdp=True,
)

# ================= CHARACTER SET =================
_pad         = '_'
_punctuation = ';:,.!?¡¿—…"«»"" %\'+-_'
_letters     = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_vi  = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ'
_numbers     = '0123456789'
_khmer_phonetic_chars = 'abcdefghijklmnopqrstuvwxyកខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអឥឧឪឫឬឮឯឱឲាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័្ '

# Use Khmer phonetic characters
combined_chars = _khmer_phonetic_chars

_symbols = sorted(list(set(combined_chars)))
symbols_string = ''.join(_symbols)

# ================= MAIN CONFIG =================
config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_khmer_single_speaker_openslr",
    use_speaker_embedding=False,  # Single speaker - no speaker embedding needed
    batch_size=32,                # Adjust based on your GPU memory
    eval_batch_size=32,
    batch_group_size=0,
    num_loader_workers=16,
    num_eval_loader_workers=16,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=500,                   # More epochs for single speaker training
    text_cleaner=None,
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    min_audio_len=16000 * 1,      # 1 second minimum
    max_audio_len=16000 * 40,     # 40 seconds maximum
    output_path=output_path,
    datasets=[],
    
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLNK>",
        characters=symbols_string,
        punctuations="!?,.;- ",
        phonemes=None,
    ),
    
    # Test sentences (will use the single speaker automatically)
    test_sentences=[
        ["កុំ កំណត់ ព្រំដែន ថា ខ្លួន មាន អាយុ យូរ អង្វែង ប៉ុណ្ណា"],
        ["ម៉ាទីន លូធើឃីង បាននិយាយថា ភាពស្អប់ ធ្វើឱ្យជីវិតជាប់គាំង"],
        ["ស្ថានភាពថៃកម្ពុជាចាប់ផ្ដើមងាកផ្ដល់ផលប្រយោជន៍អោយថៃច្រើន"],
    ],
)

# sync dict
config.from_dict(config.to_dict())

# ================= DATA PREP =================
ap = AudioProcessor(**config.audio.to_dict())

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

print(f"\n{'='*60}")
print(f"SINGLE SPEAKER TRAINING")
print(f"{'='*60}")
print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(eval_samples)}")
if train_samples:
    print(f"Speaker: {train_samples[0]['speaker_name']}")
print(f"{'='*60}\n")

# No speaker manager needed for single speaker
# No language manager needed
tokenizer, config = TTSTokenizer.init_from_config(config)
model = Vits(config, ap, tokenizer, speaker_manager=None)

# ================= TRAIN =================
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
