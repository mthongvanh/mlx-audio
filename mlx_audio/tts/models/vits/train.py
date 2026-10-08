"""Fine-tuning VITS on MLX: MMS-TTS voices, or any VITS checkpoint in
transformers' format.

A port of finetune-hf-vits (github.com/ylacombe/finetune-hf-vits, MIT),
step for step: the same losses and weights, the same order of discriminator
and generator updates, AdamW with PyTorch's defaults, and the learning rate
decayed once an epoch. It starts from a checkpoint that carries the
discriminator (`discriminator.*` weights), as that repo's
`convert_original_discriminator_checkpoint.py` makes.

    python -m mlx_audio.tts.models.vits.train \\
        --model <checkpoint with discriminator> --data <folder> --output <folder>

The data folder holds the clips and a `metadata.jsonl` of
`{"file_name": ..., "text": ...}`, as a Hugging Face audiofolder does.
"""

import argparse
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.utils import load_audio

from .modules import Conv1d
from .vits import Model, ModelConfig


@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.01  # torch.optim.AdamW's default
    lr_decay: float = 0.999875
    max_grad_norm: float = 1.0
    weight_disc: float = 3.0
    weight_fmaps: float = 1.0
    weight_gen: float = 1.0
    weight_kl: float = 1.5
    weight_duration: float = 1.0
    weight_mel: float = 35.0
    batch_size: int = 16
    epochs: int = 200
    segment_size: int = 8192  # samples the decoder makes per clip per step
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    min_duration: float = 1.0
    max_duration: float = 20.0
    max_tokens_length: int = 500
    seed: int = 456


# Spectrograms


class Spectrogram:
    """The linear and log-mel spectrograms finetune-hf-vits trains on:
    reflect-padded, Hann window, no centring, magnitudes with 1e-6 inside the
    root, Slaney mel filters. Differentiable, for the mel loss."""

    def __init__(self, sample_rate: int, n_fft=1024, hop_length=256, n_mels=80):
        from mlx_audio.dsp import mel_filters

        self.n_fft = n_fft
        self.hop_length = hop_length
        n = mx.arange(n_fft)
        self.window = 0.5 - 0.5 * mx.cos(2 * math.pi * n / n_fft)  # periodic
        self.mel_filters = mel_filters(
            sample_rate,
            n_fft,
            n_mels,
            f_min=0.0,
            f_max=sample_rate // 2,
            norm="slaney",
            mel_scale="slaney",
            precise=True,
        ).T  # (bins, mels)

    def __call__(self, waveform):
        """(batch, samples) to magnitudes (batch, frames, bins) and log-mel
        (batch, frames, mels)."""
        pad = (self.n_fft - self.hop_length) // 2
        left = waveform[:, 1 : pad + 1][:, ::-1]
        right = waveform[:, -pad - 1 : -1][:, ::-1]
        x = mx.concatenate([left, waveform, right], axis=1)
        num_frames = (x.shape[1] - self.n_fft) // self.hop_length + 1
        index = (
            mx.arange(num_frames)[:, None] * self.hop_length
            + mx.arange(self.n_fft)[None, :]
        )
        frames = x[:, index] * self.window
        spectrum = mx.fft.rfft(frames, axis=-1)
        magnitudes = mx.sqrt(mx.real(spectrum) ** 2 + mx.imag(spectrum) ** 2 + 1e-6)
        mel = magnitudes @ self.mel_filters
        return magnitudes, mx.log(mx.maximum(mel, 1e-5))


# The discriminator


class Conv2d(nn.Module):
    """A 2-D convolution, plain or weight-normed, channels-last."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        scale = math.sqrt(1.0 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            -scale, scale, (out_channels, kernel_size[0], kernel_size[1], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def _norm(self, w):
        return mx.sqrt(mx.sum(w * w, axis=(1, 2, 3), keepdims=True))

    def apply_weight_norm(self):
        if "weight" in self:
            self.weight_v = self.weight
            self.weight_g = self._norm(self.weight)
            del self["weight"]

    def __call__(self, x):
        w = (
            self.weight_g * self.weight_v / self._norm(self.weight_v)
            if "weight_v" in self
            else self.weight
        )
        return mx.conv2d(x, w, stride=self.stride, padding=self.padding) + self.bias


class ScaleDiscriminator(nn.Module):
    def __init__(self, channels: List[int], leaky_relu_slope=0.1):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs = [Conv1d(channels[0], channels[1], 15, padding=7)]
        groups = 4
        for c_in, c_out in zip(channels[1:-1], channels[2:]):
            self.convs.append(
                Conv1d(c_in, c_out, 41, stride=4, groups=groups, padding=20)
            )
            groups *= 4
        last = channels[-1]
        self.convs.append(Conv1d(last, last, 41, stride=4, groups=groups, padding=20))
        self.convs.append(Conv1d(last, last, 5, padding=2))
        self.final_conv = Conv1d(last, 1, 3, padding=1)

    def layers(self):
        return self.convs + [self.final_conv]

    def __call__(self, x):
        """x: (batch, samples, 1)."""
        fmap = []
        for conv in self.convs:
            x = nn.leaky_relu(conv(x), self.leaky_relu_slope)
            fmap.append(x)
        x = self.final_conv(x)
        fmap.append(x)
        return x.reshape(x.shape[0], -1), fmap


class PeriodDiscriminator(nn.Module):
    def __init__(
        self, channels: List[int], period, kernel_size=5, stride=3, leaky_relu_slope=0.1
    ):
        super().__init__()
        self.period = period
        self.leaky_relu_slope = leaky_relu_slope
        pad = (kernel_size - 1) // 2
        self.convs = [
            Conv2d(c_in, c_out, (kernel_size, 1), (stride, 1), (pad, 0))
            for c_in, c_out in zip(channels[:-1], channels[1:])
        ]
        self.convs.append(
            Conv2d(channels[-1], channels[-1], (kernel_size, 1), (1, 1), (pad, 0))
        )
        self.final_conv = Conv2d(channels[-1], 1, (3, 1), (1, 1), (1, 0))

    def layers(self):
        return self.convs + [self.final_conv]

    def __call__(self, x):
        """x: (batch, samples, 1)."""
        fmap = []
        batch, length, channels = x.shape
        if length % self.period:
            n_pad = self.period - length % self.period
            x = mx.concatenate([x, x[:, -n_pad - 1 : -1][:, ::-1]], axis=1)  # reflect
            length += n_pad
        x = x.reshape(batch, length // self.period, self.period, channels)
        for conv in self.convs:
            x = nn.leaky_relu(conv(x), self.leaky_relu_slope)
            fmap.append(x)
        x = self.final_conv(x)
        fmap.append(x)
        return x.reshape(batch, -1), fmap


class Discriminator(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        slope = config.get("leaky_relu_slope", 0.1)
        self.discriminators = []
        if config.get("discriminator_scale_channels") is not None:
            self.discriminators.append(
                ScaleDiscriminator(config["discriminator_scale_channels"], slope)
            )
        for period in config.get("discriminator_periods", [2, 3, 5, 7, 11]):
            self.discriminators.append(
                PeriodDiscriminator(
                    config.get(
                        "discriminator_period_channels", [1, 32, 128, 512, 1024]
                    ),
                    period,
                    config.get("discriminator_kernel_size", 5),
                    config.get("discriminator_stride", 3),
                    slope,
                )
            )

    def apply_weight_norm(self):
        for d in self.discriminators:
            for layer in d.layers():
                layer.apply_weight_norm()

    def __call__(self, waveform):
        outputs, fmaps = [], []
        for d in self.discriminators:
            out, fmap = d(waveform)
            outputs.append(out)
            fmaps.append(fmap)
        return outputs, fmaps

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """The `discriminator.*` weights of a training checkpoint, PyTorch's
        layout to MLX's."""
        out = {}
        for key, value in weights.items():
            if not key.startswith("discriminator."):
                continue
            key = key[len("discriminator.") :]
            if value.ndim == 3:
                value = value.transpose(0, 2, 1)
            elif value.ndim == 4:
                value = value.transpose(0, 2, 3, 1)
            out[key] = value
        return out


# Monotonic alignment search


def maximum_path(neg_cent: np.ndarray, frame_lengths, text_lengths) -> np.ndarray:
    """The most likely monotonic alignment of frames to tokens, as
    finetune-hf-vits's Cython `maximum_path` finds it. neg_cent: (batch,
    frames, tokens). A row at a time, each row a vector operation."""
    max_neg_val = np.float32(-1e9)
    paths = np.zeros(neg_cent.shape, dtype=np.float32)
    for b in range(neg_cent.shape[0]):
        t_y, t_x = int(frame_lengths[b]), int(text_lengths[b])
        value = neg_cent[b, :t_y, :t_x].astype(np.float32).copy()
        xs = np.arange(t_x)
        for y in range(t_y):
            lo, hi = max(0, t_x + y - t_y), min(t_x, y + 1)
            if lo >= hi:
                continue
            x = xs[lo:hi]
            if y == 0:
                v_cur = np.full(len(x), max_neg_val, dtype=np.float32)
                v_prev = np.where(x == 0, np.float32(0), max_neg_val)
            else:
                v_cur = np.where(x == y, max_neg_val, value[y - 1, x])
                v_prev = np.where(
                    x == 0, max_neg_val, value[y - 1, np.maximum(x - 1, 0)]
                )
            value[y, lo:hi] += np.maximum(v_prev, v_cur)
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            paths[b, y, index] = 1
            if (
                index != 0
                and y != 0
                and (index == y or value[y - 1, index] < value[y - 1, index - 1])
            ):
                index -= 1
    return paths


# The training forward pass and losses


def slice_segments(x, starts, size):
    """x: (batch, time, channels); a window of `size` from each start."""
    index = mx.stop_gradient(starts[:, None] + mx.arange(size)[None, :])
    return mx.take_along_axis(x, index[..., None], axis=1)


def training_forward(
    model: Model,
    batch: dict,
    segment_frames: int,
    posterior_noise=None,
    duration_noise=None,
    slice_starts=None,
):
    """finetune-hf-vits's `VitsModelForPreTraining.forward` with labels. The
    noise and slice arguments, if given, replace the random draws: tests pass
    the same to both frameworks."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]  # (batch, frames, bins)
    labels_mask = batch["labels_attention_mask"]  # (batch, frames)
    input_padding_mask = attention_mask[..., None].astype(mx.float32)
    labels_padding_mask = labels_mask[..., None].astype(mx.float32)
    speaker_embeddings = model.speaker_embeddings(batch.get("speaker_id"))

    hidden_states, prior_means, prior_log_variances = model.text_encoder(
        input_ids, input_padding_mask, attention_mask
    )
    latents, posterior_means, posterior_log_variances = model.posterior_encoder(
        labels, labels_padding_mask, speaker_embeddings, noise=posterior_noise
    )
    prior_latents = model.flow(
        latents, labels_padding_mask, speaker_embeddings, reverse=False
    )

    # The alignment, found without gradients: (batch, frames, tokens).
    pl, pm, plv = (
        mx.stop_gradient(a) for a in (prior_latents, prior_means, prior_log_variances)
    )
    prior_variances = mx.exp(-2 * plv)  # (batch, tokens, channels)
    neg_cent1 = mx.sum(-0.5 * math.log(2 * math.pi) - plv, axis=-1)[:, None, :]
    neg_cent2 = (-0.5 * pl**2) @ prior_variances.transpose(0, 2, 1)
    neg_cent3 = pl @ (pm * prior_variances).transpose(0, 2, 1)
    neg_cent4 = mx.sum(-0.5 * pm**2 * prior_variances, axis=-1)[:, None, :]
    neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
    attn = mx.array(
        maximum_path(
            np.array(neg_cent),
            np.array(labels_mask.sum(axis=1)),
            np.array(attention_mask.sum(axis=1)),
        )
    )
    durations = attn.sum(axis=1)[..., None]  # (batch, tokens, 1)

    if model.config.use_stochastic_duration_prediction:
        log_duration = model.duration_predictor(
            hidden_states,
            input_padding_mask,
            speaker_embeddings,
            durations=durations,
            reverse=False,
            noise=duration_noise,
        )
        log_duration = log_duration / mx.sum(input_padding_mask)
    else:
        log_duration_padded = mx.log(durations + 1e-6) * input_padding_mask
        predicted = model.duration_predictor(
            hidden_states, input_padding_mask, speaker_embeddings
        )
        log_duration = mx.sum(
            (predicted - log_duration_padded) ** 2, axis=(1, 2)
        ) / mx.sum(input_padding_mask)

    prior_means = attn @ prior_means  # (batch, frames, channels)
    prior_log_variances = attn @ prior_log_variances

    label_lengths = labels_mask.sum(axis=1)
    if slice_starts is None:
        slice_starts = (
            mx.random.uniform(shape=label_lengths.shape)
            * (label_lengths - segment_frames + 1)
        ).astype(mx.int32)
    latents_slice = slice_segments(latents, slice_starts, segment_frames)
    waveform = model.decoder(latents_slice, speaker_embeddings)  # (batch, samples, 1)

    return dict(
        waveform=waveform,
        log_duration=log_duration,
        attn=attn,
        ids_slice=slice_starts,
        labels_padding_mask=labels_padding_mask,
        prior_latents=prior_latents,
        prior_means=prior_means,
        prior_log_variances=prior_log_variances,
        posterior_means=posterior_means,
        posterior_log_variances=posterior_log_variances,
    )


def discriminator_loss(real_outputs, generated_outputs):
    real_loss = sum(mx.mean((1 - r) ** 2) for r in real_outputs)
    generated_loss = sum(mx.mean(g**2) for g in generated_outputs)
    return real_loss + generated_loss, real_loss, generated_loss


def feature_loss(fmaps_real, fmaps_generated):
    loss = 0
    for maps_real, maps_generated in zip(fmaps_real, fmaps_generated):
        for real, generated in zip(maps_real, maps_generated):
            loss = loss + mx.mean(mx.abs(mx.stop_gradient(real) - generated))
    return loss * 2


def generator_loss(outputs):
    return sum(mx.mean((1 - o) ** 2) for o in outputs)


def kl_loss(
    prior_latents, posterior_log_variance, prior_means, prior_log_variance, labels_mask
):
    kl = prior_log_variance - posterior_log_variance - 0.5
    kl = kl + 0.5 * (prior_latents - prior_means) ** 2 * mx.exp(
        -2.0 * prior_log_variance
    )
    return mx.sum(kl * labels_mask) / mx.sum(labels_mask)


# Training


def _adamw(config: TrainConfig, learning_rate):
    return optim.AdamW(
        learning_rate=learning_rate,
        betas=[config.adam_beta1, config.adam_beta2],
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
        bias_correction=True,  # as PyTorch's
    )


def prepare_for_training(model: Model):
    """Weight norm where finetune-hf-vits trains with it: the decoder's
    upsampling and residual layers, and each flow's input and output."""
    for conv in model.decoder.weight_normed_convs():
        conv.apply_weight_norm()
    for flow in model.flow.flows:
        flow.conv_pre.apply_weight_norm()
        flow.conv_post.apply_weight_norm()
    model.train()


class Trainer:
    def __init__(self, model: Model, discriminator: Discriminator, config: TrainConfig):
        self.model = model
        self.discriminator = discriminator
        self.config = config
        self.segment_frames = config.segment_size // config.hop_length
        self.spectrogram = Spectrogram(
            model.sample_rate, config.n_fft, config.hop_length, config.n_mels
        )
        self.gen_optimizer = _adamw(config, config.learning_rate)
        self.disc_optimizer = _adamw(config, config.learning_rate)
        self.gen_grad = nn.value_and_grad(model, self._generator_loss)
        self.disc_grad = nn.value_and_grad(discriminator, self._discriminator_loss)

    def set_epoch(self, epoch: int):
        """ExponentialLR, stepped at the start of each epoch as
        finetune-hf-vits does, so epoch 0 already runs at lr × decay."""
        lr = self.config.learning_rate * self.config.lr_decay ** (epoch + 1)
        self.gen_optimizer.learning_rate = lr
        self.disc_optimizer.learning_rate = lr

    def _targets(self, batch, outputs):
        frames = self.segment_frames
        mel_target = slice_segments(batch["mel"], outputs["ids_slice"], frames)
        wave_target = slice_segments(
            batch["waveform"][..., None],
            outputs["ids_slice"] * self.config.hop_length,
            self.config.segment_size,
        )
        return mel_target, wave_target

    def _discriminator_loss(self, fake, real):
        real_out, _ = self.discriminator(real)
        fake_out, _ = self.discriminator(fake)
        loss, real_loss, fake_loss = discriminator_loss(real_out, fake_out)
        return loss * self.config.weight_disc, (loss, real_loss, fake_loss)

    def _generator_loss(self, batch, noise):
        c = self.config
        outputs = training_forward(self.model, batch, self.segment_frames, **noise)
        mel_target, wave_target = self._targets(batch, outputs)
        _, mel_generated = self.spectrogram(outputs["waveform"][..., 0])
        _, fmaps_target = self.discriminator(wave_target)
        disc_generated, fmaps_generated = self.discriminator(outputs["waveform"])

        loss_duration = mx.sum(outputs["log_duration"])
        loss_mel = mx.mean(mx.abs(mel_target - mel_generated))
        loss_kl = kl_loss(
            outputs["prior_latents"],
            outputs["posterior_log_variances"],
            outputs["prior_means"],
            outputs["prior_log_variances"],
            outputs["labels_padding_mask"],
        )
        loss_fmaps = feature_loss(fmaps_target, fmaps_generated)
        loss_gen = generator_loss(disc_generated)
        total = (
            loss_duration * c.weight_duration
            + loss_mel * c.weight_mel
            + loss_kl * c.weight_kl
            + loss_fmaps * c.weight_fmaps
            + loss_gen * c.weight_gen
        )
        return total, dict(
            duration=loss_duration,
            mel=loss_mel,
            kl=loss_kl,
            fmaps=loss_fmaps,
            gen=loss_gen,
        )

    def _draw_noise(self, batch):
        """The random draws for one step, made once and used by both passes
        through the generator."""
        b, frames = batch["labels"].shape[:2]
        tokens = batch["input_ids"].shape[1]
        lengths = batch["labels_attention_mask"].sum(axis=1)
        return dict(
            posterior_noise=mx.random.normal((b, frames, self.model.config.flow_size)),
            duration_noise=mx.random.normal((b, tokens, 2)),
            slice_starts=(
                mx.random.uniform(shape=(b,)) * (lengths - self.segment_frames + 1)
            ).astype(mx.int32),
        )

    def step(self, batch, noise: Optional[dict] = None) -> Dict[str, float]:
        """One update of each network. `noise`, if given, replaces the
        step's random draws: tests pass the same to both frameworks."""
        c = self.config
        noise = noise or self._draw_noise(batch)
        # Dropout and layer drop draw too; the same seed for both passes
        # makes the second see what the first did.
        seed = random.getrandbits(31)

        # 1. The discriminator, on this step's generated audio.
        mx.random.seed(seed)
        random.seed(seed)
        outputs = training_forward(self.model, batch, self.segment_frames, **noise)
        fake = mx.stop_gradient(outputs["waveform"])
        _, wave_target = self._targets(batch, outputs)
        (_, (loss_disc, real_disc, fake_disc)), grads = self.disc_grad(
            fake, wave_target
        )
        grads, _ = optim.clip_grad_norm(grads, c.max_grad_norm)
        self.disc_optimizer.update(self.discriminator, grads)
        mx.eval(self.discriminator.parameters(), self.disc_optimizer.state)

        # 2. The generator, against the updated discriminator.
        mx.random.seed(seed)
        random.seed(seed)
        (total, parts), grads = self.gen_grad(batch, noise)
        grads, _ = optim.clip_grad_norm(grads, c.max_grad_norm)
        self.gen_optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.gen_optimizer.state)
        # Every batch has its own shape, so freed buffers rarely fit the next
        # step; kept, the cache grows by gigabytes a step until the Mac swaps.
        mx.clear_cache()

        losses = {k: v.item() for k, v in parts.items()}
        losses.update(
            total=sum(losses.values()),
            disc=loss_disc.item(),
            real_disc=real_disc.item(),
            fake_disc=fake_disc.item(),
        )
        return losses


# Data


def load_clips(
    folder: Path,
    model: Model,
    spectrogram: Spectrogram,
    config: TrainConfig,
    text_map: Optional[Dict[str, str]] = None,
):
    """Clips from `metadata.jsonl`, with their tokens and spectrograms, those
    too short, too long or with too many tokens left out."""
    clips = []
    for line in (folder / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row["text"]
        for old, new in (text_map or {}).items():
            text = text.replace(old, new)
        audio = np.array(
            load_audio(str(folder / row["file_name"]), sample_rate=model.sample_rate)
        )
        duration = len(audio) / model.sample_rate
        if not config.min_duration <= duration <= config.max_duration:
            continue
        ids = model.tokenizer.encode(text)[: config.max_tokens_length + 1]
        magnitudes, mel = spectrogram(mx.array(audio)[None])
        clips.append(
            dict(
                file=row["file_name"],
                ids=np.array(ids, dtype=np.int32),
                waveform=audio.astype(np.float32),
                labels=np.array(magnitudes[0]),
                mel=np.array(mel[0]),
            )
        )
    return clips


def collate(clips) -> dict:
    """Pads a batch: tokens with 0, audio and spectrograms with 0, and masks
    saying what is real."""

    def pad(arrays):
        n = max(len(a) for a in arrays)
        out = np.zeros((len(arrays), n) + arrays[0].shape[1:], dtype=arrays[0].dtype)
        mask = np.zeros((len(arrays), n), dtype=np.float32)
        for i, a in enumerate(arrays):
            out[i, : len(a)] = a
            mask[i, : len(a)] = 1
        return mx.array(out), mx.array(mask)

    input_ids, attention_mask = pad([c["ids"] for c in clips])
    labels, labels_mask = pad([c["labels"] for c in clips])
    mel, _ = pad([c["mel"] for c in clips])
    waveform, _ = pad([c["waveform"] for c in clips])
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        labels_attention_mask=labels_mask,
        mel=mel,
        waveform=waveform,
    )


# Loading and saving


def load_for_training(path: str):
    """The generator and discriminator from a training checkpoint, ready to
    train."""
    from mlx_audio.utils import get_model_path, load_config, load_weights

    path = get_model_path(path) if not Path(path).exists() else Path(path)
    config = load_config(path)
    weights = load_weights(path)
    model = Model(ModelConfig.from_dict(config))
    model.load_weights(list(model.sanitize(weights).items()))
    model.tokenizer = Model.post_load_hook(model, path).tokenizer
    discriminator = Discriminator(config)
    disc_weights = Discriminator.sanitize(weights)
    if not disc_weights:
        raise ValueError(
            f"{path} has no discriminator. Make a training checkpoint with "
            "finetune-hf-vits's convert_original_discriminator_checkpoint.py."
        )
    discriminator.load_weights(list(disc_weights.items()))
    discriminator.apply_weight_norm()
    prepare_for_training(model)
    mx.eval(model.parameters(), discriminator.parameters())
    return model, discriminator, path, config


def save(model: Model, source: Path, output: Path):
    """The generator in transformers' format, weight norm folded where an
    inference checkpoint has none, with the source's config and tokenizer."""
    output.mkdir(parents=True, exist_ok=True)
    weights = model.to_transformers()
    folded = {}
    for key, value in weights.items():
        foldable = key.startswith("decoder.") or (
            key.startswith("flow.flows.")
            and (".conv_pre." in key or ".conv_post." in key)
        )
        if foldable and key.endswith(".weight_v"):
            base = key[: -len(".weight_v")]
            g = weights[base + ".weight_g"]
            norm = mx.sqrt(mx.sum(value * value, axis=(1, 2), keepdims=True))
            folded[base + ".weight"] = g * value / norm
        elif foldable and key.endswith(".weight_g"):
            continue
        else:
            folded[key] = value
    mx.save_safetensors(
        str(output / "model.safetensors"), folded, metadata={"format": "pt"}
    )
    config = json.loads((source / "config.json").read_text())
    config = {k: v for k, v in config.items() if not k.startswith("discriminator_")}
    config["architectures"] = ["VitsModel"]
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for name in (
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        if (source / name).exists():
            shutil.copy(source / name, output / name)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a VITS voice on MLX.")
    parser.add_argument(
        "--model", required=True, help="Training checkpoint, with discriminator"
    )
    parser.add_argument(
        "--data", required=True, help="Folder with metadata.jsonl and clips"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--learning-rate", type=float, default=TrainConfig.learning_rate
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Epochs between saves; 0 for the end only",
    )
    parser.add_argument(
        "--text-map",
        default=None,
        help="JSON of replacements applied to each transcript",
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--layerdrop",
        type=float,
        default=None,
        help="Chance of skipping each text-encoder layer in a step; the checkpoint's (0.1 for MMS) "
        "if not given. 0 steadies the KL loss, which spikes when a layer is skipped.",
    )
    args = parser.parse_args()

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    random.seed(config.seed)
    mx.random.seed(config.seed)
    model, discriminator, source, _ = load_for_training(args.model)
    if args.layerdrop is not None:
        model.text_encoder.encoder.layerdrop = args.layerdrop
    trainer = Trainer(model, discriminator, config)
    text_map = json.loads(Path(args.text_map).read_text()) if args.text_map else None
    clips = load_clips(Path(args.data), model, trainer.spectrogram, config, text_map)
    if not clips:
        raise SystemExit("No clips between the minimum and maximum duration.")
    print(
        f"{len(clips)} clips, {sum(len(c['waveform']) for c in clips) / model.sample_rate / 60:.1f} minutes"
    )

    output = Path(args.output)
    step = 0
    start = time.time()
    for epoch in range(config.epochs):
        trainer.set_epoch(epoch)
        order = list(range(len(clips)))
        random.shuffle(order)
        for i in range(0, len(order), config.batch_size):
            batch = collate([clips[j] for j in order[i : i + config.batch_size]])
            losses = trainer.step(batch)
            step += 1
            print(
                f"epoch {epoch} step {step} "
                + " ".join(f"{k} {v:.3f}" for k, v in losses.items())
                + f" ({time.time() - start:.0f}s)",
                flush=True,
            )
            if args.max_steps and step >= args.max_steps:
                break
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save(model, source, output / f"epoch-{epoch + 1}")
        if args.max_steps and step >= args.max_steps:
            break
    save(model, source, output)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
