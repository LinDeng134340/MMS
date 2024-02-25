import os
import subprocess

# Get the access token from environment
access_token = os.getenv("HUGGINGFACE_TOKEN")

if not access_token:
    raise Exception("you need to set the HUGGINGFACE_TOKEN environment variable")

# Use subprocess to execute the shell command
subprocess.run(['huggingface-cli', 'login', '--token', access_token])


from datasets import load_dataset, Audio
from transformers import Wav2Vec2ForCTC, AutoProcessor
import torch

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True, use_auth_token=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

# French
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "fr", split="test", streaming=True, use_auth_token=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
fr_sample = next(iter(stream_data))["audio"]["array"]


model_id = "/kaggle/input/mms/pytorch/300m/1"

processor = AutoProcessor.from_pretrained(model_id)#,repo_type='model')
model = Wav2Vec2ForCTC.from_pretrained(model_id)#,repo_type='model')

inputs = processor(en_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

processor.tokenizer.set_target_lang("fra")
model.load_adapter("fra")

inputs = processor(fr_sample, sampling_rate=16_000, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).logits

ids = torch.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)

transcription

