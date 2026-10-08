"""Fixtures for checking another port's trainer against this one: one batch
of two clips, the noise for one step, and what this port makes of them (the
spectrograms, the training forward pass, every loss, the gradients' norms,
and some parameters after one full step), as safetensors.

Runs in mlx-audio's own environment, after setup.sh has built the
reference (for the training checkpoint and the clips):

    python export_train_fixtures.py <work> <out dir>

Writes <out dir>/vits_train_fixtures.safetensors. Dropout and layer drop
are off, so a port that matches gives the same numbers.
"""

import argparse
import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.tts.models.vits import train as T

parser = argparse.ArgumentParser()
parser.add_argument("work", type=Path)
parser.add_argument("out", type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

model, disc, _, _ = T.load_for_training(str(args.work / "mms-tts-blt-train"))
model.eval()
disc.eval()
config = T.TrainConfig(batch_size=2)
trainer = T.Trainer(model, disc, config)
mx.random.seed(0)
out = {}

# One batch: the first two clips, of different lengths.
data = args.work / "smoke_data/train"
rows = [json.loads(l) for l in (data / "metadata.jsonl").read_text().splitlines()][:2]
clips = T.load_clips(data, model, trainer.spectrogram, config)[:2]
batch = T.collate(clips)
for key, value in batch.items():
    out[f"batch.{key}"] = value
out["batch.input_ids"] = out["batch.input_ids"].astype(mx.int32)

# The spectrogram of the first clip on its own.
magnitudes, mel = trainer.spectrogram(mx.array(clips[0]["waveform"])[None])
out["spectrogram.magnitudes"] = magnitudes
out["spectrogram.mel"] = mel

# The step's noise.
noise = trainer._draw_noise(batch)
for key, value in noise.items():
    out[f"noise.{key}"] = value

# The training forward pass.
outputs = T.training_forward(model, batch, trainer.segment_frames, **noise)
for key in (
    "waveform",
    "log_duration",
    "attn",
    "prior_latents",
    "prior_means",
    "prior_log_variances",
    "posterior_log_variances",
):
    out[f"forward.{key}"] = outputs[key]

# Losses and gradients, before any update.
fake = mx.stop_gradient(outputs["waveform"])
_, wave_target = trainer._targets(batch, outputs)
(_, (loss_disc, real_disc, fake_disc)), disc_grads = trainer.disc_grad(
    fake, wave_target
)
out["loss.disc"] = loss_disc
out["loss.real_disc"] = real_disc
out["loss.fake_disc"] = fake_disc
(total, parts), gen_grads = trainer.gen_grad(batch, noise)
out["loss.total"] = total
for key, value in parts.items():
    out[f"loss.{key}"] = value


def global_norm(grads):
    return mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))


out["grad_norm.disc"] = global_norm(disc_grads)
out["grad_norm.gen"] = global_norm(gen_grads)
# Each generator parameter's gradient norm, to find where a port parts.
for key, g in tree_flatten(gen_grads):
    out[f"gen_grad_norm.{key}"] = mx.sqrt(mx.sum(g * g))

# What the mel loss, and the discriminator's losses, send back into the
# generated audio: where a port's gradients part, if they do.
mel_target, wave_target = trainer._targets(batch, outputs)


def mel_part(w):
    _, m = trainer.spectrogram(w[..., 0])
    return mx.mean(mx.abs(mel_target - m))


def adversarial_part(w):
    _, fmaps_target = disc(wave_target)
    generated, fmaps_generated = disc(w)
    return T.feature_loss(fmaps_target, fmaps_generated) + T.generator_loss(generated)


latents, _, _ = model.posterior_encoder(
    batch["labels"],
    batch["labels_attention_mask"][..., None],
    noise=noise["posterior_noise"],
)
out["decoder.input"] = T.slice_segments(
    latents, noise["slice_starts"], trainer.segment_frames
)
out["grad_wave.mel"] = mx.grad(mel_part)(outputs["waveform"])
out["grad_wave.adversarial"] = mx.grad(adversarial_part)(outputs["waveform"])

# One full step: the parameters it moves, a few of them whole.
before = dict(tree_flatten(model.parameters()))
trainer.set_epoch(0)
losses = trainer.step(batch, noise)
after = dict(tree_flatten(model.parameters()))
for key in (
    "text_encoder.encoder.layers.0.attention.q_proj.weight",
    "duration_predictor.flows.1.conv_proj.weight",
    "flow.flows.0.conv_pre.weight_v",
    "decoder.upsampler.0.weight_v",
    "decoder.conv_post.weight",
):
    out[f"step.{key}"] = after[key]
num = sum(mx.sum((after[k] - before[k]) ** 2) for k in after)
out["step.update_norm"] = mx.sqrt(num)
for key, value in losses.items():
    out[f"step.loss.{key}"] = mx.array(value)

mx.eval(out)
mx.save_safetensors(str(args.out / "vits_train_fixtures.safetensors"), out)
print(f"{len(out)} arrays to {args.out}")
for k, v in out.items():
    print(f"  {k:60s} {tuple(v.shape)} {v.dtype}")
