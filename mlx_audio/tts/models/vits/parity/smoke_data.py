"""A stand-in dataset: the stock blt voice reading verses, as the audiofolder
finetune-hf-vits and the MLX trainer both read (clips and a metadata.jsonl).

    python smoke_data.py <work dir>
"""

import json
import os
import sys

import soundfile as sf
import torch
from text import normalise
from transformers import VitsModel, VitsTokenizer

work = sys.argv[1]
verses = json.load(open(f"{work}/text/verses.json"))
tok = VitsTokenizer.from_pretrained("facebook/mms-tts-blt")
model = VitsModel.from_pretrained("facebook/mms-tts-blt").eval()
for split, rows in [("train", range(20, 60)), ("test", range(0, 8))]:
    d = f"{work}/smoke_data/{split}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/metadata.jsonl", "w") as f:
        for i in rows:
            text = normalise(verses[i])
            torch.manual_seed(i)
            with torch.no_grad():
                wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
            name = f"clip_{i:03}.wav"
            sf.write(f"{d}/{name}", wav, 16000)
            f.write(json.dumps({"file_name": name, "text": text}, ensure_ascii=False))
            f.write("\n")
print("smoke data ready")
