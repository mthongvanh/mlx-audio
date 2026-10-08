"""Fixtures for checking another port of this model (a Swift one, say)
against this one: inputs and outputs at each stage of speaking, saved as
safetensors, plus the tokenizer's ids for a few texts as JSON.

Runs in mlx-audio's own environment; needs no PyTorch.

    python export_fixtures.py <out dir> [--repo facebook/mms-tts-blt]

Writes <out dir>/vits_fixtures.safetensors and vits_tokenizer.json. Every
stage is given its inputs from this port, so a mismatch shows where it
starts rather than everything after it.
"""

import argparse
import json
import warnings
from pathlib import Path

import mlx.core as mx

from mlx_audio.tts.utils import load

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path)
parser.add_argument("--repo", default="facebook/mms-tts-blt")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

TEXT = "kháo 'hung saư 'viạk chảu 'giê‐'su 'kha‐'lịt phủ pên 'lụk 'chái chảu pua 'phạ"
SHORT = "té pang 'chạu"
TOKENIZER_CASES = [
    TEXT,
    SHORT,
    "ꞌha#ꞌnị, ê‑sa",  # marks the voice lacks: dropped
    "Kháo HUNG",  # upper case: lowered
    "",
    "  pên  ",
]

m = load(args.repo)
m.eval()
mx.random.seed(0)
out = {}

# Tokens.
ids = m.tokenizer.encode(TEXT)
out["ids"] = mx.array([ids], dtype=mx.int32)
cases = []
for text in TOKENIZER_CASES:
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        cases.append({"text": text, "ids": m.tokenizer.encode(text)})
(args.out / "vits_tokenizer.json").write_text(
    json.dumps({"repo": args.repo, "cases": cases}, ensure_ascii=False, indent=1)
)

# Text encoder.
length = len(ids)
pmask = mx.ones((1, length, 1))
hidden, prior_means, prior_log_var = m.text_encoder(
    out["ids"], pmask, mx.ones((1, length))
)
out["text_encoder.hidden"] = hidden
out["text_encoder.prior_means"] = prior_means
out["text_encoder.prior_log_variances"] = prior_log_var

# Durations: noise off, then a fixed draw at the default scale.
out["duration.log_duration_quiet"] = m.duration_predictor(
    hidden, pmask, None, reverse=True, noise_scale=0.0
)
duration_noise = mx.random.normal((1, length, 2))
out["duration.noise"] = duration_noise
out["duration.log_duration"] = m.duration_predictor(
    hidden,
    pmask,
    None,
    reverse=True,
    noise_scale=m.config.noise_scale_duration,
    noise=duration_noise,
)

# The flow, reversed, on a fixed latent.
frames = 96
flow_input = mx.random.normal((1, frames, m.config.flow_size))
flow_mask = mx.ones((1, frames, 1))
out["flow.input"] = flow_input
out["flow.output"] = m.flow(flow_input, flow_mask, None, reverse=True)

# The decoder, on the flow's output.
out["decoder.output"] = m.decoder(out["flow.output"])

# The whole model: noise off, then fixed draws at the default scales.
wav, lengths = m(out["ids"], noise_scale=0.0, noise_scale_duration=0.0)
out["model.waveform_quiet"] = wav
out["model.lengths_quiet"] = lengths
# As many frames as the durations drawn with `duration_noise` give.
prior_frames = int(mx.ceil(mx.exp(out["duration.log_duration"])).sum().item())
prior_noise = mx.random.normal((1, prior_frames, m.config.flow_size))
out["model.prior_noise"] = prior_noise
wav, lengths = m(out["ids"], duration_noise=duration_noise, prior_noise=prior_noise)
out["model.waveform"] = wav
out["model.lengths"] = lengths

# A padded batch of two, noise off.
short = m.tokenizer.encode(SHORT)
batch = mx.zeros((2, length), dtype=mx.int32)
batch[0, :] = mx.array(ids)
batch[1, : len(short)] = mx.array(short)
mask = mx.zeros((2, length))
mask[0, :] = 1
mask[1, : len(short)] = 1
out["batch.ids"] = batch
out["batch.attention_mask"] = mask
wav, lengths = m(batch, mask, noise_scale=0.0, noise_scale_duration=0.0)
out["batch.waveform"] = wav
out["batch.lengths"] = lengths

mx.eval(out)
mx.save_safetensors(str(args.out / "vits_fixtures.safetensors"), out)
print(f"{len(out)} arrays to {args.out}")
for k, v in out.items():
    print(f"  {k:36s} {tuple(v.shape)} {v.dtype}")
