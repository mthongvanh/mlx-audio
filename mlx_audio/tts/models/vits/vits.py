"""VITS for MLX: Meta's MMS-TTS voices (about 1,100 languages) and other
VITS checkpoints in transformers' format."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..base import BaseModelArgs, GenerationResult
from .modules import (
    ConvTranspose1d,
    DurationPredictor,
    HifiGan,
    PosteriorEncoder,
    ResidualCouplingBlock,
    StochasticDurationPredictor,
    TextEncoder,
)
from .tokenizer import VitsTokenizer


@dataclass
class ModelConfig(BaseModelArgs):
    model_type: str = "vits"
    vocab_size: int = 38
    hidden_size: int = 192
    num_hidden_layers: int = 6
    num_attention_heads: int = 2
    window_size: int = 4
    use_bias: bool = True
    ffn_dim: int = 768
    layerdrop: float = 0.1
    ffn_kernel_size: int = 3
    flow_size: int = 192
    spectrogram_bins: int = 513
    hidden_act: str = "relu"
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_stochastic_duration_prediction: bool = True
    num_speakers: int = 1
    speaker_embedding_size: int = 0
    upsample_initial_channel: int = 512
    upsample_rates: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(
        default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    )
    leaky_relu_slope: float = 0.1
    depth_separable_channels: int = 2
    depth_separable_num_layers: int = 3
    duration_predictor_flow_bins: int = 10
    duration_predictor_tail_bound: float = 5.0
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout: float = 0.5
    duration_predictor_num_flows: int = 4
    duration_predictor_filter_channels: int = 256
    prior_encoder_num_flows: int = 4
    prior_encoder_num_wavenet_layers: int = 4
    posterior_encoder_num_wavenet_layers: int = 16
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_dropout: float = 0.0
    speaking_rate: float = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    sampling_rate: int = 16000
    model_path: Optional[str] = None

    @property
    def sample_rate(self):
        return self.sampling_rate


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.text_encoder = TextEncoder(config)
        self.flow = ResidualCouplingBlock(config)
        self.decoder = HifiGan(config)
        if config.use_stochastic_duration_prediction:
            self.duration_predictor = StochasticDurationPredictor(config)
        else:
            self.duration_predictor = DurationPredictor(config)
        if config.num_speakers > 1:
            self.embed_speaker = nn.Embedding(
                config.num_speakers, config.speaker_embedding_size
            )
        self.posterior_encoder = PosteriorEncoder(config)
        self.tokenizer: Optional[VitsTokenizer] = None

    @property
    def sample_rate(self):
        return self.config.sampling_rate

    @property
    def hop_length(self):
        return int(np.prod(self.config.upsample_rates))

    def speaker_embeddings(self, speaker_id):
        if self.config.num_speakers <= 1 or speaker_id is None:
            return None
        if isinstance(speaker_id, int):
            speaker_id = mx.array([speaker_id])
        return self.embed_speaker(speaker_id)[:, None, :]

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        speaker_id=None,
        noise_scale: Optional[float] = None,
        noise_scale_duration: Optional[float] = None,
        speaking_rate: Optional[float] = None,
        duration_noise: Optional[mx.array] = None,
        prior_noise: Optional[mx.array] = None,
    ):
        """Waveforms (batch, samples) for token ids (batch, length), and each
        one's length in samples. The noise arguments, if given, replace the
        random draws: tests pass the same to both frameworks."""
        config = self.config
        noise_scale = config.noise_scale if noise_scale is None else noise_scale
        noise_scale_duration = (
            config.noise_scale_duration
            if noise_scale_duration is None
            else noise_scale_duration
        )
        speaking_rate = config.speaking_rate if speaking_rate is None else speaking_rate

        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape)
        input_padding_mask = attention_mask[..., None].astype(mx.float32)
        speaker_embeddings = self.speaker_embeddings(speaker_id)

        hidden_states, prior_means, prior_log_variances = self.text_encoder(
            input_ids, input_padding_mask, attention_mask
        )

        if config.use_stochastic_duration_prediction:
            log_duration = self.duration_predictor(
                hidden_states,
                input_padding_mask,
                speaker_embeddings,
                reverse=True,
                noise_scale=noise_scale_duration,
                noise=duration_noise,
            )
        else:
            log_duration = self.duration_predictor(
                hidden_states, input_padding_mask, speaker_embeddings
            )

        duration = mx.ceil(mx.exp(log_duration) * input_padding_mask / speaking_rate)
        predicted_lengths = mx.maximum(mx.sum(duration, axis=(1, 2)), 1).astype(
            mx.int32
        )
        output_length = int(predicted_lengths.max().item())
        output_padding_mask = (
            mx.arange(output_length)[None, :] < predicted_lengths[:, None]
        )[..., None].astype(mx.float32)

        # Each output frame to the input token it expands.
        cum_duration = mx.cumsum(duration[..., 0], axis=1)
        valid = (
            mx.arange(output_length)[None, None, :] < cum_duration[..., None]
        ).astype(mx.float32)
        path = valid - mx.pad(valid, [(0, 0), (1, 0), (0, 0)])[:, :-1]
        attn = path.transpose(0, 2, 1) * (
            output_padding_mask * input_padding_mask.transpose(0, 2, 1)
        )

        prior_means = attn @ prior_means
        prior_log_variances = attn @ prior_log_variances
        noise = (
            prior_noise
            if prior_noise is not None
            else mx.random.normal(prior_means.shape)
        )
        prior_latents = prior_means + noise * mx.exp(prior_log_variances) * noise_scale

        latents = self.flow(
            prior_latents, output_padding_mask, speaker_embeddings, reverse=True
        )
        spectrogram = latents * output_padding_mask
        waveform = self.decoder(spectrogram, speaker_embeddings)[..., 0]
        return waveform, predicted_lengths * self.hop_length

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        speaker_id: Optional[int] = None,
        noise_scale: Optional[float] = None,
        noise_scale_duration: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        """Speech for `text`, as one result. `speed` scales the speaking
        rate; `seed` makes the result repeatable."""
        if self.tokenizer is None:
            raise ValueError("VITS: no tokenizer; load the model from its folder.")
        start = time.time()
        if seed is not None:
            mx.random.seed(seed)
        ids = self.tokenizer.encode(text)
        if len(ids) <= 1:
            raise ValueError("VITS: nothing in the text this voice can read.")
        if speaker_id is None and voice is not None and str(voice).isdigit():
            speaker_id = int(voice)

        waveform, lengths = self(
            mx.array([ids]),
            speaker_id=speaker_id,
            noise_scale=noise_scale,
            noise_scale_duration=noise_scale_duration,
            speaking_rate=self.config.speaking_rate * speed,
        )
        audio = waveform[0, : int(lengths[0].item())]
        mx.eval(audio)
        elapsed = time.time() - start
        samples = audio.shape[0]
        duration = samples / self.sample_rate
        yield GenerationResult(
            audio=audio,
            samples=samples,
            sample_rate=self.sample_rate,
            segment_idx=0,
            token_count=len(ids),
            audio_duration=f"{int(duration // 3600):02d}:{int(duration % 3600 // 60):02d}:{int(duration % 60):02d}.{int(duration % 1 * 1000):03d}",
            real_time_factor=round(elapsed / duration, 2) if duration > 0 else 0,
            prompt={
                "tokens": len(ids),
                "tokens-per-sec": round(len(ids) / elapsed, 2) if elapsed > 0 else 0,
            },
            audio_samples={
                "samples": samples,
                "samples-per-sec": round(samples / elapsed, 2) if elapsed > 0 else 0,
            },
            processing_time_seconds=elapsed,
            peak_memory_usage=mx.get_peak_memory() / 1e9,
        )

    def sanitize(self, weights):
        """transformers' weights, in PyTorch's layout, to this model's. Weights
        already in this model's layout, as mlx-audio saves them, pass as they
        are."""
        from mlx.utils import tree_flatten

        expected = {k: v.shape for k, v in tree_flatten(self.parameters())}
        sanitized = {}
        for key, value in weights.items():
            if key.startswith("discriminator."):
                continue  # training checkpoints carry one; see train.py
            # PyTorch's newer name for weight norm's two halves.
            key = key.replace(".parametrizations.weight.original0", ".weight_g")
            key = key.replace(".parametrizations.weight.original1", ".weight_v")
            if key.endswith((".translate", ".log_scale")):
                value = value.reshape(-1)
            elif value.ndim == 3 and not key.endswith(
                ("emb_rel_k", "emb_rel_v", "weight_g")
            ):
                if key.startswith("decoder.upsampler."):
                    order = (1, 2, 0)  # (in, out, k) to (out, k, in)
                else:
                    order = (0, 2, 1)  # (out, in, k) to (out, k, in)
                transposed = tuple(value.shape[i] for i in order)
                # Already this model's layout: the shape fits as it is and
                # wouldn't once transposed. When both fit (in == k), PyTorch's
                # layout is assumed.
                mlx_layout = (
                    expected.get(key) == value.shape and transposed != value.shape
                )
                if not mlx_layout:
                    value = value.transpose(*order)
            elif key.endswith("weight_g") and key.startswith("decoder.upsampler."):
                value = value.reshape(1, 1, -1)  # normed over the input channel
            sanitized[key] = value
        return sanitized

    def to_transformers(self) -> dict:
        """This model's weights in transformers' names and layout, the inverse
        of `sanitize`, so a model trained here loads in transformers."""
        from mlx.utils import tree_flatten

        out = {}
        for key, value in tree_flatten(self.parameters()):
            if key.endswith((".translate", ".log_scale")):
                value = value.reshape(-1, 1)
            elif value.ndim == 3 and not key.endswith(
                ("emb_rel_k", "emb_rel_v", "weight_g")
            ):
                if key.startswith("decoder.upsampler."):
                    value = value.transpose(2, 0, 1)
                else:
                    value = value.transpose(0, 2, 1)
            elif key.endswith("weight_g") and key.startswith("decoder.upsampler."):
                value = value.reshape(-1, 1, 1)
            out[key] = value
        return out

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        model.tokenizer = VitsTokenizer.from_pretrained(model_path)
        return model
