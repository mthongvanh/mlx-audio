"""Speaking: the MLX port against transformers' VitsModel, noise off, layer
by layer, then the waveform, then a padded batch against single items.

Runs in mlx-audio's own environment with torch added; needs no setup.sh.

    python parity_infer.py [--repo facebook/mms-tts-blt]

Exits 1 if any figure is outside its tolerance.
"""

import argparse
import sys
import warnings

import mlx.core as mx
import numpy as np
import torch
from transformers import VitsModel as TorchVits
from transformers import VitsTokenizer as TorchTok

from mlx_audio.tts.utils import load

# Measured on an M2 Max: 1.9e-6, 2.6e-5, 6.0e-5 and 3.8e-5.
TOLERANCE = {
    "text encoder hidden": 1e-4,
    "prior means": 1e-4,
    "prior log var": 1e-4,
    "log durations (sdp)": 1e-3,
    "waveform": 1e-3,
    "batched item 0 vs single": 1e-3,
}

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="facebook/mms-tts-blt")
parser.add_argument(
    "--text",
    default="kháo 'hung saư 'viạk chảu 'giê‐'su 'kha‐'lịt phủ pên 'lụk 'chái chảu pua 'phạ",
)
parser.add_argument("--short", default="té pang 'chạu")
args = parser.parse_args()
warnings.simplefilter("ignore", FutureWarning)

tt = TorchTok.from_pretrained(args.repo)
tm = TorchVits.from_pretrained(args.repo).eval()
m = load(args.repo)
failures = []

ids_t = tt(args.text, return_tensors="pt").input_ids
ids_m = m.tokenizer.encode(args.text)
same_tokens = ids_t[0].tolist() == ids_m
print(f"tokens equal: {same_tokens} ({len(ids_m)})")
if not same_tokens:
    failures.append("tokens")


def cmp(name, a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.shape != b.shape:
        print(f"{name}: SHAPE {a.shape} vs {b.shape}")
        failures.append(name)
        return
    d = float(np.abs(a - b).max())
    ok = d <= TOLERANCE[name]
    print(
        f"{name:28s} max|diff| {d:.2e}  (scale {np.abs(a).max():.2e}){'' if ok else '  FAIL'}"
    )
    if not ok:
        failures.append(name)


with torch.no_grad():
    mask = torch.ones_like(ids_t).unsqueeze(-1).float()
    te = tm.text_encoder(input_ids=ids_t, padding_mask=mask)
    h_t, pm_t, pv_t = te.last_hidden_state, te.prior_means, te.prior_log_variances
ids = mx.array([ids_m])
pmask = mx.ones((1, len(ids_m), 1))
h_m, pm_m, pv_m = m.text_encoder(ids, pmask, mx.ones((1, len(ids_m))))
cmp("text encoder hidden", h_t, h_m)
cmp("prior means", pm_t, pm_m)
cmp("prior log var", pv_t, pv_m)

with torch.no_grad():
    ld_t = tm.duration_predictor(
        h_t.transpose(1, 2), mask.transpose(1, 2), None, reverse=True, noise_scale=0.0
    )
ld_m = m.duration_predictor(h_m, pmask, None, reverse=True, noise_scale=0.0)
cmp("log durations (sdp)", ld_t[0, 0], ld_m[0, :, 0])
frames_t = int(torch.ceil(torch.exp(ld_t)).sum())
frames_m = int(mx.ceil(mx.exp(ld_m)).sum().item())
print(f"frames torch/mlx: {frames_t} {frames_m}")

tm.noise_scale = 0.0
tm.noise_scale_duration = 0.0
with torch.no_grad():
    wav_t = tm(ids_t).waveform[0].numpy()
wav_m, lens = m(ids, noise_scale=0.0, noise_scale_duration=0.0)
wav_m = np.array(wav_m[0])
print(f"samples torch/mlx: {len(wav_t)} {int(lens[0].item())}")
if len(wav_t) != int(lens[0].item()):
    failures.append("samples")
cmp("waveform", wav_t, wav_m[: len(wav_t)])

# A batch of two, padded, gives the same first item.
ids2 = tt([args.text, args.short], return_tensors="pt", padding=True)
wav_b, lens_b = m(
    mx.array(ids2.input_ids.numpy()),
    mx.array(ids2.attention_mask.numpy()),
    noise_scale=0.0,
    noise_scale_duration=0.0,
)
cmp("batched item 0 vs single", wav_m, np.array(wav_b[0, : int(lens_b[0].item())]))

print("dropped-character warning, expected below:")
with warnings.catch_warnings():
    warnings.simplefilter("always")
    m.tokenizer.encode("ꞌha#ꞌnị, ê‑sa")

print("PASS" if not failures else f"FAIL: {', '.join(failures)}")
sys.exit(1 if failures else 0)
