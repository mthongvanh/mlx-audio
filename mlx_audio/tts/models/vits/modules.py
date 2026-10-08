"""VITS's layers, ported from transformers' `modeling_vits.py`.

Names follow transformers', so its checkpoints (MMS-TTS among them) load
without renaming. Tensors are channels-last, (batch, time, channels), as
MLX's convolutions want; transformers' convolutional parts are (batch,
channels, time). Masks are (batch, time, 1).
"""

import math
import random

import mlx.core as mx
import mlx.nn as nn

from .spline import unconstrained_rational_quadratic_spline


class Conv1d(nn.Module):
    """A 1-D convolution that holds its weight plainly or weight-normed.

    Weight-normed, it holds `weight_g` and `weight_v` as transformers'
    checkpoints do, and the weight is `weight_g * weight_v / |weight_v|`,
    the norm taken over each output channel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        weight_norm: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        scale = math.sqrt(1.0 / (in_channels // groups * kernel_size))
        weight = mx.random.uniform(
            -scale, scale, (out_channels, kernel_size, in_channels // groups)
        )
        if weight_norm:
            self.weight_v = weight
            self.weight_g = _norm(weight, axes=(1, 2))
        else:
            self.weight = weight
        if bias:
            self.bias = mx.zeros((out_channels,))

    @property
    def weight_normed(self) -> bool:
        return "weight_v" in self

    def effective_weight(self):
        if self.weight_normed:
            return self.weight_g * self.weight_v / _norm(self.weight_v, axes=(1, 2))
        return self.weight

    def apply_weight_norm(self):
        if not self.weight_normed:
            self.weight_v = self.weight
            self.weight_g = _norm(self.weight, axes=(1, 2))
            del self["weight"]

    def remove_weight_norm(self):
        if self.weight_normed:
            self.weight = self.effective_weight()
            del self["weight_v"]
            del self["weight_g"]

    def __call__(self, x):
        y = mx.conv1d(
            x,
            self.effective_weight(),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        if "bias" in self:
            y = y + self.bias
        return y


class ConvTranspose1d(nn.Module):
    """A transposed 1-D convolution, plain or weight-normed.

    PyTorch normalises a transposed convolution's weight over its *input*
    channel, its first axis. In MLX's layout, (out, kernel, in), that is the
    last axis, so the norm is over the other two.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        scale = math.sqrt(1.0 / (in_channels * kernel_size))
        self.weight = mx.random.uniform(
            -scale, scale, (out_channels, kernel_size, in_channels)
        )
        if bias:
            self.bias = mx.zeros((out_channels,))

    @property
    def weight_normed(self) -> bool:
        return "weight_v" in self

    def effective_weight(self):
        if self.weight_normed:
            return self.weight_g * self.weight_v / _norm(self.weight_v, axes=(0, 1))
        return self.weight

    def apply_weight_norm(self):
        if not self.weight_normed:
            self.weight_v = self.weight
            self.weight_g = _norm(self.weight, axes=(0, 1))
            del self["weight"]

    def remove_weight_norm(self):
        if self.weight_normed:
            self.weight = self.effective_weight()
            del self["weight_v"]
            del self["weight_g"]

    def __call__(self, x):
        y = mx.conv_transpose1d(
            x, self.effective_weight(), stride=self.stride, padding=self.padding
        )
        if "bias" in self:
            y = y + self.bias
        return y


def _norm(w, axes):
    return mx.sqrt(mx.sum(w * w, axis=axes, keepdims=True))


def fused_add_tanh_sigmoid_multiply(input_a, input_b, num_channels):
    in_act = input_a + input_b
    return mx.tanh(in_act[..., :num_channels]) * mx.sigmoid(in_act[..., num_channels:])


class WaveNet(nn.Module):
    def __init__(self, config, num_layers: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(config.wavenet_dropout)
        if config.speaker_embedding_size != 0:
            self.cond_layer = Conv1d(
                config.speaker_embedding_size,
                2 * config.hidden_size * num_layers,
                1,
                weight_norm=True,
            )
        self.in_layers = []
        self.res_skip_layers = []
        for i in range(num_layers):
            dilation = config.wavenet_dilation_rate**i
            padding = (config.wavenet_kernel_size * dilation - dilation) // 2
            self.in_layers.append(
                Conv1d(
                    config.hidden_size,
                    2 * config.hidden_size,
                    config.wavenet_kernel_size,
                    dilation=dilation,
                    padding=padding,
                    weight_norm=True,
                )
            )
            res_skip_channels = (
                2 * config.hidden_size if i < num_layers - 1 else config.hidden_size
            )
            self.res_skip_layers.append(
                Conv1d(config.hidden_size, res_skip_channels, 1, weight_norm=True)
            )

    def __call__(self, inputs, padding_mask, global_conditioning=None):
        outputs = mx.zeros_like(inputs)
        if global_conditioning is not None:
            global_conditioning = self.cond_layer(global_conditioning)
        for i in range(self.num_layers):
            hidden_states = self.in_layers[i](inputs)
            if global_conditioning is not None:
                offset = i * 2 * self.hidden_size
                global_states = global_conditioning[
                    ..., offset : offset + 2 * self.hidden_size
                ]
            else:
                global_states = mx.zeros_like(hidden_states)
            acts = fused_add_tanh_sigmoid_multiply(
                hidden_states, global_states, self.hidden_size
            )
            acts = self.dropout(acts)
            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.num_layers - 1:
                inputs = (
                    inputs + res_skip_acts[..., : self.hidden_size]
                ) * padding_mask
                outputs = outputs + res_skip_acts[..., self.hidden_size :]
            else:
                outputs = outputs + res_skip_acts
        return outputs * padding_mask


class PosteriorEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_channels = config.flow_size
        self.conv_pre = Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = WaveNet(config, config.posterior_encoder_num_wavenet_layers)
        self.conv_proj = Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def __call__(self, inputs, padding_mask, global_conditioning=None, noise=None):
        inputs = self.conv_pre(inputs) * padding_mask
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        stats = self.conv_proj(inputs) * padding_mask
        mean, log_stddev = mx.split(stats, 2, axis=-1)
        if noise is None:
            noise = mx.random.normal(mean.shape)
        sampled = (mean + noise * mx.exp(log_stddev)) * padding_mask
        return sampled, mean, log_stddev


class HifiGanResidualBlock(nn.Module):
    def __init__(
        self, channels, kernel_size=3, dilation=(1, 3, 5), leaky_relu_slope=0.1
    ):
        super().__init__()
        self.leaky_relu_slope = leaky_relu_slope
        self.convs1 = [
            Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=d,
                padding=(kernel_size * d - d) // 2,
            )
            for d in dilation
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
            for _ in dilation
        ]

    def __call__(self, hidden_states):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = hidden_states
            hidden_states = nn.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv1(hidden_states)
            hidden_states = nn.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = conv2(hidden_states)
            hidden_states = hidden_states + residual
        return hidden_states


class HifiGan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.leaky_relu_slope = config.leaky_relu_slope
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = Conv1d(
            config.flow_size, config.upsample_initial_channel, 7, padding=3
        )
        self.upsampler = [
            ConvTranspose1d(
                config.upsample_initial_channel // (2**i),
                config.upsample_initial_channel // (2 ** (i + 1)),
                kernel_size=k,
                stride=u,
                padding=(k - u) // 2,
            )
            for i, (u, k) in enumerate(
                zip(config.upsample_rates, config.upsample_kernel_sizes)
            )
        ]
        self.resblocks = []
        for i in range(self.num_upsamples):
            channels = config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(
                    HifiGanResidualBlock(channels, k, d, config.leaky_relu_slope)
                )
        self.conv_post = Conv1d(channels, 1, 7, padding=3, bias=False)
        if config.speaker_embedding_size != 0:
            self.cond = Conv1d(
                config.speaker_embedding_size, config.upsample_initial_channel, 1
            )

    def weight_normed_convs(self):
        """The layers training holds weight-normed, as finetune-hf-vits does."""
        convs = list(self.upsampler)
        for block in self.resblocks:
            convs += block.convs1 + block.convs2
        return convs

    def __call__(self, spectrogram, global_conditioning=None):
        hidden_states = self.conv_pre(spectrogram)
        if global_conditioning is not None:
            hidden_states = hidden_states + self.cond(global_conditioning)
        for i in range(self.num_upsamples):
            hidden_states = nn.leaky_relu(hidden_states, self.leaky_relu_slope)
            hidden_states = self.upsampler[i](hidden_states)
            res_state = self.resblocks[i * self.num_kernels](hidden_states)
            for j in range(1, self.num_kernels):
                res_state = res_state + self.resblocks[i * self.num_kernels + j](
                    hidden_states
                )
            hidden_states = res_state / self.num_kernels
        # The last activation uses PyTorch's default slope, not the config's.
        hidden_states = nn.leaky_relu(hidden_states, 0.01)
        hidden_states = self.conv_post(hidden_states)
        return mx.tanh(hidden_states)


class ResidualCouplingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.half_channels = config.flow_size // 2
        self.conv_pre = Conv1d(self.half_channels, config.hidden_size, 1)
        self.wavenet = WaveNet(config, config.prior_encoder_num_wavenet_layers)
        self.conv_post = Conv1d(config.hidden_size, self.half_channels, 1)

    def __call__(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = mx.split(inputs, 2, axis=-1)
        hidden_states = self.conv_pre(first_half) * padding_mask
        hidden_states = self.wavenet(hidden_states, padding_mask, global_conditioning)
        mean = self.conv_post(hidden_states) * padding_mask
        # Mean only: the scale is 1, so the log-determinant is 0.
        if not reverse:
            second_half = mean + second_half * padding_mask
        else:
            second_half = (second_half - mean) * padding_mask
        return mx.concatenate([first_half, second_half], axis=-1)


class ResidualCouplingBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.flows = [
            ResidualCouplingLayer(config) for _ in range(config.prior_encoder_num_flows)
        ]

    def __call__(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                inputs = flow(inputs, padding_mask, global_conditioning)
                inputs = inputs[..., ::-1]
        else:
            for flow in reversed(self.flows):
                inputs = inputs[..., ::-1]
                inputs = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs


class DilatedDepthSeparableConv(nn.Module):
    def __init__(self, config, dropout_rate=0.0):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        channels = config.hidden_size
        self.num_layers = config.depth_separable_num_layers
        self.dropout = nn.Dropout(dropout_rate)
        self.convs_dilated = []
        self.convs_pointwise = []
        self.norms_1 = []
        self.norms_2 = []
        for i in range(self.num_layers):
            dilation = kernel_size**i
            self.convs_dilated.append(
                Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    groups=channels,
                    dilation=dilation,
                    padding=(kernel_size * dilation - dilation) // 2,
                )
            )
            self.convs_pointwise.append(Conv1d(channels, channels, 1))
            self.norms_1.append(nn.LayerNorm(channels))
            self.norms_2.append(nn.LayerNorm(channels))

    def __call__(self, inputs, padding_mask, global_conditioning=None):
        if global_conditioning is not None:
            inputs = inputs + global_conditioning
        for i in range(self.num_layers):
            hidden_states = self.convs_dilated[i](inputs * padding_mask)
            hidden_states = nn.gelu(self.norms_1[i](hidden_states))
            hidden_states = self.convs_pointwise[i](hidden_states)
            hidden_states = nn.gelu(self.norms_2[i](hidden_states))
            hidden_states = self.dropout(hidden_states)
            inputs = inputs + hidden_states
        return inputs * padding_mask


class ConvFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.filter_channels = config.hidden_size
        self.half_channels = config.depth_separable_channels // 2
        self.num_bins = config.duration_predictor_flow_bins
        self.tail_bound = config.duration_predictor_tail_bound
        self.conv_pre = Conv1d(self.half_channels, self.filter_channels, 1)
        self.conv_dds = DilatedDepthSeparableConv(config)
        self.conv_proj = Conv1d(
            self.filter_channels, self.half_channels * (self.num_bins * 3 - 1), 1
        )

    def __call__(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        first_half, second_half = mx.split(inputs, 2, axis=-1)
        hidden_states = self.conv_pre(first_half)
        hidden_states = self.conv_dds(hidden_states, padding_mask, global_conditioning)
        hidden_states = self.conv_proj(hidden_states) * padding_mask

        batch_size, length, channels = first_half.shape
        # (batch, time, channels, bins), channel-major as transformers splits it.
        hidden_states = hidden_states.reshape(batch_size, length, channels, -1)
        scale = math.sqrt(self.filter_channels)
        unnormalized_widths = hidden_states[..., : self.num_bins] / scale
        unnormalized_heights = (
            hidden_states[..., self.num_bins : 2 * self.num_bins] / scale
        )
        unnormalized_derivatives = hidden_states[..., 2 * self.num_bins :]

        second_half, log_abs_det = unconstrained_rational_quadratic_spline(
            second_half,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            reverse=reverse,
            tail_bound=self.tail_bound,
        )
        outputs = mx.concatenate([first_half, second_half], axis=-1) * padding_mask
        if not reverse:
            return outputs, mx.sum(log_abs_det * padding_mask, axis=(1, 2))
        return outputs, None


class ElementwiseAffine(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Per channel, held as (channels,); transformers holds (channels, 1).
        self.translate = mx.zeros((config.depth_separable_channels,))
        self.log_scale = mx.zeros((config.depth_separable_channels,))

    def __call__(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            outputs = (self.translate + mx.exp(self.log_scale) * inputs) * padding_mask
            return outputs, mx.sum(self.log_scale * padding_mask, axis=(1, 2))
        outputs = (inputs - self.translate) * mx.exp(-self.log_scale) * padding_mask
        return outputs, None


class StochasticDurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.speaker_embedding_size
        filter_channels = config.hidden_size
        self.conv_pre = Conv1d(filter_channels, filter_channels, 1)
        self.conv_proj = Conv1d(filter_channels, filter_channels, 1)
        self.conv_dds = DilatedDepthSeparableConv(
            config, dropout_rate=config.duration_predictor_dropout
        )
        if embed_dim != 0:
            self.cond = Conv1d(embed_dim, filter_channels, 1)
        self.flows = [ElementwiseAffine(config)] + [
            ConvFlow(config) for _ in range(config.duration_predictor_num_flows)
        ]
        self.post_conv_pre = Conv1d(1, filter_channels, 1)
        self.post_conv_proj = Conv1d(filter_channels, filter_channels, 1)
        self.post_conv_dds = DilatedDepthSeparableConv(
            config, dropout_rate=config.duration_predictor_dropout
        )
        self.post_flows = [ElementwiseAffine(config)] + [
            ConvFlow(config) for _ in range(config.duration_predictor_num_flows)
        ]

    def __call__(
        self,
        inputs,
        padding_mask,
        global_conditioning=None,
        durations=None,
        reverse=False,
        noise_scale=1.0,
        noise=None,
    ):
        """The negative log-likelihood of `durations`, or with `reverse`,
        sampled log-durations. `noise`, if given, replaces the random draw:
        tests pass the same to both frameworks."""
        inputs = mx.stop_gradient(inputs)
        inputs = self.conv_pre(inputs)
        if global_conditioning is not None:
            global_conditioning = mx.stop_gradient(global_conditioning)
            inputs = inputs + self.cond(global_conditioning)
        inputs = self.conv_dds(inputs, padding_mask)
        inputs = self.conv_proj(inputs) * padding_mask

        log2pi = math.log(2 * math.pi)
        if not reverse:
            hidden_states = self.post_conv_pre(durations)
            hidden_states = self.post_conv_dds(hidden_states, padding_mask)
            hidden_states = self.post_conv_proj(hidden_states) * padding_mask
            shape = (durations.shape[0], durations.shape[1], 2)
            random_posterior = (
                noise if noise is not None else mx.random.normal(shape)
            ) * padding_mask

            log_determinant_posterior_sum = 0
            latents_posterior = random_posterior
            # No flip after the first flow, the affine one, as in the original
            # VITS and finetune-hf-vits; transformers' copy flips there too,
            # which its inference never reaches.
            for i, flow in enumerate(self.post_flows):
                latents_posterior, log_determinant = flow(
                    latents_posterior,
                    padding_mask,
                    global_conditioning=inputs + hidden_states,
                )
                if i > 0:
                    latents_posterior = latents_posterior[..., ::-1]
                log_determinant_posterior_sum += log_determinant
            first_half, second_half = mx.split(latents_posterior, 2, axis=-1)
            log_determinant_posterior_sum += mx.sum(
                (nn.log_sigmoid(first_half) + nn.log_sigmoid(-first_half))
                * padding_mask,
                axis=(1, 2),
            )
            logq = (
                mx.sum(
                    -0.5 * (log2pi + random_posterior**2) * padding_mask, axis=(1, 2)
                )
                - log_determinant_posterior_sum
            )

            first_half = (durations - mx.sigmoid(first_half)) * padding_mask
            first_half = mx.log(mx.maximum(first_half, 1e-5)) * padding_mask
            log_determinant_sum = mx.sum(-first_half, axis=(1, 2))

            latents = mx.concatenate([first_half, second_half], axis=-1)
            for i, flow in enumerate(self.flows):
                latents, log_determinant = flow(
                    latents, padding_mask, global_conditioning=inputs
                )
                if i > 0:
                    latents = latents[..., ::-1]
                log_determinant_sum += log_determinant
            nll = (
                mx.sum(0.5 * (log2pi + latents**2) * padding_mask, axis=(1, 2))
                - log_determinant_sum
            )
            return nll + logq

        flows = list(reversed(self.flows))
        flows = flows[:-2] + [flows[-1]]  # transformers drops one flow here
        shape = (inputs.shape[0], inputs.shape[1], 2)
        latents = (
            noise if noise is not None else mx.random.normal(shape)
        ) * noise_scale
        for flow in flows:
            latents = latents[..., ::-1]
            latents, _ = flow(
                latents, padding_mask, global_conditioning=inputs, reverse=True
            )
        return latents[..., :1]


class DurationPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = config.duration_predictor_kernel_size
        filter_channels = config.duration_predictor_filter_channels
        self.dropout = nn.Dropout(config.duration_predictor_dropout)
        self.conv_1 = Conv1d(
            config.hidden_size, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.conv_2 = Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = nn.LayerNorm(filter_channels, eps=config.layer_norm_eps)
        self.proj = Conv1d(filter_channels, 1, 1)
        if config.speaker_embedding_size != 0:
            self.cond = Conv1d(config.speaker_embedding_size, config.hidden_size, 1)

    def __call__(self, inputs, padding_mask, global_conditioning=None):
        inputs = mx.stop_gradient(inputs)
        if global_conditioning is not None:
            inputs = inputs + self.cond(mx.stop_gradient(global_conditioning))
        inputs = self.dropout(self.norm_1(nn.relu(self.conv_1(inputs * padding_mask))))
        inputs = self.dropout(self.norm_2(nn.relu(self.conv_2(inputs * padding_mask))))
        return self.proj(inputs * padding_mask) * padding_mask


class Attention(nn.Module):
    """Multi-head attention with relative position embeddings in a window."""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.window_size = config.window_size
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.dropout = nn.Dropout(config.attention_dropout)
        bias = config.use_bias
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        if self.window_size:
            shape = (1, self.window_size * 2 + 1, self.head_dim)
            self.emb_rel_k = mx.random.normal(shape) * self.scaling
            self.emb_rel_v = mx.random.normal(shape) * self.scaling

    def _heads(self, x, bsz, length):
        x = x.reshape(bsz, length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        return x.reshape(bsz * self.num_heads, length, self.head_dim)

    def __call__(self, hidden_states, attention_mask=None):
        bsz, length, _ = hidden_states.shape
        query = self._heads(self.q_proj(hidden_states) * self.scaling, bsz, length)
        key = self._heads(self.k_proj(hidden_states), bsz, length)
        value = self._heads(self.v_proj(hidden_states), bsz, length)

        attn_weights = query @ key.transpose(0, 2, 1)
        if self.window_size:
            key_rel = self._relative_embeddings(self.emb_rel_k, length)
            relative_logits = query @ key_rel.transpose(0, 2, 1)
            attn_weights = attn_weights + _relative_to_absolute(relative_logits)
        if attention_mask is not None:
            attn_weights = (
                attn_weights.reshape(bsz, self.num_heads, length, length)
                + attention_mask
            ).reshape(bsz * self.num_heads, length, length)
        attn_weights = mx.softmax(attn_weights, axis=-1, precise=True)
        attn_probs = self.dropout(attn_weights)

        attn_output = attn_probs @ value
        if self.window_size:
            value_rel = self._relative_embeddings(self.emb_rel_v, length)
            attn_output = attn_output + _absolute_to_relative(attn_probs) @ value_rel

        attn_output = attn_output.reshape(bsz, self.num_heads, length, self.head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            bsz, length, self.embed_dim
        )
        return self.out_proj(attn_output)

    def _relative_embeddings(self, relative_embeddings, length):
        pad_length = max(length - (self.window_size + 1), 0)
        if pad_length > 0:
            relative_embeddings = mx.pad(
                relative_embeddings, [(0, 0), (pad_length, pad_length), (0, 0)]
            )
        start = max((self.window_size + 1) - length, 0)
        return relative_embeddings[:, start : start + 2 * length - 1]


def _relative_to_absolute(x):
    batch_heads, length, _ = x.shape
    x = mx.pad(x, [(0, 0), (0, 0), (0, 1)])
    x_flat = x.reshape(batch_heads, length * 2 * length)
    x_flat = mx.pad(x_flat, [(0, 0), (0, length - 1)])
    x_final = x_flat.reshape(batch_heads, length + 1, 2 * length - 1)
    return x_final[:, :length, length - 1 :]


def _absolute_to_relative(x):
    batch_heads, length, _ = x.shape
    x = mx.pad(x, [(0, 0), (0, 0), (0, length - 1)])
    x_flat = x.reshape(batch_heads, length * (2 * length - 1))
    x_flat = mx.pad(x_flat, [(0, 0), (length, 0)])
    return x_flat.reshape(batch_heads, length, 2 * length)[:, :, 1:]


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        k = config.ffn_kernel_size
        self.conv_1 = Conv1d(config.hidden_size, config.ffn_dim, k)
        self.conv_2 = Conv1d(config.ffn_dim, config.hidden_size, k)
        self.dropout = nn.Dropout(config.activation_dropout)
        if config.hidden_act not in ("relu", "gelu"):
            raise ValueError(f"VITS: unsupported hidden_act {config.hidden_act!r}")
        self.act_fn = nn.relu if config.hidden_act == "relu" else nn.gelu
        self.padding = [(0, 0), ((k - 1) // 2, k // 2), (0, 0)] if k > 1 else None

    def _pad(self, x):
        return mx.pad(x, self.padding) if self.padding else x

    def __call__(self, hidden_states, padding_mask):
        hidden_states = self.conv_1(self._pad(hidden_states * padding_mask))
        hidden_states = self.dropout(self.act_fn(hidden_states))
        hidden_states = self.conv_2(self._pad(hidden_states * padding_mask))
        return hidden_states * padding_mask


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, hidden_states, padding_mask, attention_mask=None):
        residual = hidden_states
        hidden_states = self.dropout(self.attention(hidden_states, attention_mask))
        hidden_states = self.layer_norm(residual + hidden_states)
        residual = hidden_states
        hidden_states = self.dropout(self.feed_forward(hidden_states, padding_mask))
        return self.final_layer_norm(residual + hidden_states)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.layerdrop = config.layerdrop

    def __call__(self, hidden_states, padding_mask, attention_mask=None):
        if attention_mask is not None:
            # (batch, time) of 1 and 0 to an additive (batch, 1, 1, time).
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -1e9
        hidden_states = hidden_states * padding_mask
        for layer in self.layers:
            if self.training and random.random() < self.layerdrop:
                continue
            hidden_states = layer(hidden_states, padding_mask, attention_mask)
        return hidden_states * padding_mask


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.flow_size = config.flow_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = Encoder(config)
        self.project = Conv1d(config.hidden_size, config.flow_size * 2, 1)

    def __call__(self, input_ids, padding_mask, attention_mask=None):
        """The encoded text, and the prior's means and log-variances."""
        hidden_states = self.embed_tokens(input_ids) * math.sqrt(self.hidden_size)
        hidden_states = self.encoder(hidden_states, padding_mask, attention_mask)
        stats = self.project(hidden_states) * padding_mask
        prior_means, prior_log_variances = mx.split(stats, 2, axis=-1)
        return hidden_states, prior_means, prior_log_variances
