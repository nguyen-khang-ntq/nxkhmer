import torch
import torchaudio
from tqdm import tqdm
from underthesea import sent_tokenize

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model paths
xtts_checkpoint = "/home/coder/data/Interspeech/model/XTTSv2-Finetuning-for-New-Languages/checkpoints/GPT_XTTS_FT-January-24-2026_05+46PM-8e59ec3/best_model_80101.pth"
xtts_config = "/home/coder/data/Interspeech/model/XTTSv2-Finetuning-for-New-Languages/checkpoints/GPT_XTTS_FT-January-24-2026_05+46PM-8e59ec3/config.json"
xtts_vocab = "/home/coder/data/Interspeech/model/XTTSv2-Finetuning-for-New-Languages/checkpoints/XTTS_v2.0_original_model_files/vocab.json"

# Load model
config = XttsConfig()
config.load_json(xtts_config)
XTTS_MODEL = Xtts.init_from_config(config)
XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
XTTS_MODEL.to(device)

print("Model loaded successfully!")

# Inference
tts_text = "សួស្តី ខ្ញុំជាជនជាតិវៀតណាម។"
speaker_audio_file = "/home/coder/datasets/crawl_datasets/final_dataset/prompt_2.wav"
lang = "km"

gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(
    audio_path=speaker_audio_file,
    gpt_cond_len=XTTS_MODEL.config.gpt_cond_len,
    max_ref_length=XTTS_MODEL.config.max_ref_len,
    sound_norm_refs=XTTS_MODEL.config.sound_norm_refs,
)

tts_texts = sent_tokenize(tts_text)

wav_chunks = []
for text in tqdm(tts_texts):
    wav_chunk = XTTS_MODEL.inference(
        text=text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.1,
        length_penalty=1.0,
        repetition_penalty=10.0,
        top_k=10,
        top_p=0.3,
    )
    wav_chunks.append(torch.tensor(wav_chunk["wav"]))

out_wav = torch.cat(wav_chunks, dim=0).unsqueeze(0).cpu()

# Save audio to file
output_path = "output.wav"
torchaudio.save(output_path, out_wav, 24000)
print(f"Audio saved to {output_path}")