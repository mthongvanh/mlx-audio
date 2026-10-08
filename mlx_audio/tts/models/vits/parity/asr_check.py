"""A voice made with MLX, read back by MMS's Tai Dam recogniser
(facebook/mms-1b-all with its blt adapter, in PyTorch): character and word
error over 20 verses, and how fast the voice spoke them.

Runs in mlx-audio's own environment with torch, jiwer and soundfile added,
after setup.sh has fetched the verses:

    python asr_check.py <work> [--model facebook/mms-tts-blt] [--max-cer 0.03]

The recogniser is about 4 GB. Measured on an M2 Max: the stock voice at
1.8% character error, about 18 times faster than real time.
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from text import normalise, without_marks
from transformers import AutoProcessor, Wav2Vec2ForCTC

from mlx_audio.tts.utils import load

parser = argparse.ArgumentParser()
parser.add_argument("work", type=Path)
parser.add_argument("--model", default="facebook/mms-tts-blt")
parser.add_argument("--verses", type=int, default=20)
parser.add_argument("--max-cer", type=float, default=0.03)
parser.add_argument("--save", type=Path, help="write the first three clips here")
args = parser.parse_args()
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

verses = [normalise(v) for v in json.load(open(args.work / "text/verses.json"))]
verses = verses[: args.verses]
m = load(args.model)
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="blt")
asr = Wav2Vec2ForCTC.from_pretrained(
    "facebook/mms-1b-all", target_lang="blt", ignore_mismatched_sizes=True
).eval()
if args.save:
    args.save.mkdir(parents=True, exist_ok=True)

refs, hyps, t_tts, secs = [], [], 0.0, 0.0
for i, v in enumerate(verses):
    t0 = time.time()
    r = next(m.generate(v, seed=i))
    a = np.array(r.audio)
    t_tts += time.time() - t0
    secs += len(a) / 16000
    if args.save and i < 3:
        sf.write(args.save / f"verse_{i}.wav", a, 16000)
    x = proc(a, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        ids = torch.argmax(asr(**x).logits, -1)[0]
    refs.append(without_marks(v))
    hyps.append(without_marks(proc.decode(ids)))
cer = jiwer.cer(refs, hyps)
print(
    f"cer {cer:.3f}  wer {jiwer.wer(refs, hyps):.3f}  "
    f"tts {t_tts:.1f}s for {secs:.0f}s of speech (rtf {t_tts/secs:.3f})"
)
print("PASS" if cer <= args.max_cer else f"FAIL: cer above {args.max_cer}")
sys.exit(0 if cer <= args.max_cer else 1)
