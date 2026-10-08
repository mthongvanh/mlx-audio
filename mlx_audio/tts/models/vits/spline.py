"""Rational-quadratic spline flows for VITS's stochastic duration predictor.

A port of transformers' `_unconstrained_rational_quadratic_spline`. Where
transformers indexes only the elements inside the interval, this computes
every element and picks with `mx.where`. Inputs are clipped into the
interval first, so the unpicked branch stays finite and its gradient can't
turn into NaN.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def _take(params, index):
    return mx.take_along_axis(params, index[..., None], axis=-1)[..., 0]


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse=False,
    tail_bound=5.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # The derivative at each end is fixed so the spline meets the identity
    # outside the interval with slope 1.
    constant = math.log(math.exp(1 - min_derivative) - 1)
    edge = mx.full(unnormalized_derivatives.shape[:-1] + (1,), constant)
    unnormalized_derivatives = mx.concatenate(
        [edge, unnormalized_derivatives, edge], axis=-1
    )

    outputs, log_abs_det = rational_quadratic_spline(
        mx.clip(inputs, -tail_bound, tail_bound),
        unnormalized_widths,
        unnormalized_heights,
        unnormalized_derivatives,
        reverse=reverse,
        tail_bound=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs = mx.where(inside, outputs, inputs)
    log_abs_det = mx.where(inside, log_abs_det, 0.0)
    return outputs, log_abs_det


def _knots(unnormalized, lower, upper, min_size):
    num_bins = unnormalized.shape[-1]
    sizes = mx.softmax(unnormalized, axis=-1)
    sizes = min_size + (1 - min_size * num_bins) * sizes
    cumulative = mx.cumsum(sizes, axis=-1)
    cumulative = (upper - lower) * cumulative + lower
    # Pin both ends exactly, as transformers does.
    first = mx.full(cumulative.shape[:-1] + (1,), lower)
    last = mx.full(cumulative.shape[:-1] + (1,), upper)
    cumulative = mx.concatenate([first, cumulative[..., :-1], last], axis=-1)
    return cumulative, cumulative[..., 1:] - cumulative[..., :-1]


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    reverse,
    tail_bound,
    min_bin_width,
    min_bin_height,
    min_derivative,
):
    lower, upper = -tail_bound, tail_bound

    cumwidths, widths = _knots(unnormalized_widths, lower, upper, min_bin_width)
    cumheights, heights = _knots(unnormalized_heights, lower, upper, min_bin_height)
    derivatives = min_derivative + nn.softplus(unnormalized_derivatives)

    bin_locations = cumheights if reverse else cumwidths
    # Nudge the last knot so an input exactly at the upper bound falls in
    # the last bin.
    nudge = mx.concatenate(
        [
            mx.zeros(bin_locations.shape[:-1] + (bin_locations.shape[-1] - 1,)),
            mx.full(bin_locations.shape[:-1] + (1,), 1e-6),
        ],
        axis=-1,
    )
    bin_locations = mx.stop_gradient(bin_locations + nudge)
    bin_idx = mx.sum(inputs[..., None] >= bin_locations, axis=-1) - 1
    bin_idx = mx.stop_gradient(mx.clip(bin_idx, 0, widths.shape[-1] - 1))

    input_cumwidths = _take(cumwidths, bin_idx)
    input_bin_widths = _take(widths, bin_idx)
    input_cumheights = _take(cumheights, bin_idx)
    delta = heights / widths
    input_delta = _take(delta, bin_idx)
    input_derivatives = _take(derivatives, bin_idx)
    input_derivatives_plus_one = _take(derivatives[..., 1:], bin_idx)
    input_heights = _take(heights, bin_idx)

    intermediate1 = input_derivatives + input_derivatives_plus_one - 2 * input_delta
    if not reverse:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)
        numerator = input_heights * (
            input_delta * theta**2 + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + intermediate1 * theta_one_minus_theta
        outputs = input_cumheights + numerator / denominator
        derivative_numerator = input_delta**2 * (
            input_derivatives_plus_one * theta**2
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta) ** 2
        )
        log_abs_det = mx.log(derivative_numerator) - 2 * mx.log(denominator)
        return outputs, log_abs_det

    intermediate2 = inputs - input_cumheights
    intermediate3 = intermediate2 * intermediate1
    a = input_heights * (input_delta - input_derivatives) + intermediate3
    b = input_heights * input_derivatives - intermediate3
    c = -input_delta * intermediate2
    discriminant = mx.maximum(b**2 - 4 * a * c, 0.0)
    root = (2 * c) / (-b - mx.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths
    theta_one_minus_theta = root * (1 - root)
    denominator = input_delta + intermediate1 * theta_one_minus_theta
    derivative_numerator = input_delta**2 * (
        input_derivatives_plus_one * root**2
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - root) ** 2
    )
    log_abs_det = mx.log(derivative_numerator) - 2 * mx.log(denominator)
    return outputs, -log_abs_det
