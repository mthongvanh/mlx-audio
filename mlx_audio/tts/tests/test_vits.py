"""VITS (MMS-TTS) on a tiny random model: no downloads.

Parity with transformers and finetune-hf-vits on a real checkpoint was
checked separately (inference within 6e-5 on the waveform; one training
step's losses within 2.5e-6 and gradients' global norm within 0.05%).
"""

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_audio.tts.models.vits import train as T
from mlx_audio.tts.models.vits.spline import unconstrained_rational_quadratic_spline
from mlx_audio.tts.models.vits.tokenizer import VitsTokenizer
from mlx_audio.tts.models.vits.vits import Model, ModelConfig

VOCAB = {c: i for i, c in enumerate(["_", "a", "b", "c", " ", "'"])}

TINY = dict(
    vocab_size=len(VOCAB),
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    window_size=2,
    ffn_dim=32,
    flow_size=16,
    spectrogram_bins=33,
    upsample_initial_channel=32,
    upsample_rates=[4, 4],
    upsample_kernel_sizes=[8, 8],
    resblock_kernel_sizes=[3],
    resblock_dilation_sizes=[[1, 3]],
    duration_predictor_filter_channels=16,
    duration_predictor_num_flows=2,
    prior_encoder_num_flows=2,
    prior_encoder_num_wavenet_layers=2,
    posterior_encoder_num_wavenet_layers=2,
    sampling_rate=1024,
)
DISCRIMINATOR = dict(
    discriminator_scale_channels=[1, 4, 16, 64],
    discriminator_periods=[2, 3],
    discriminator_period_channels=[1, 4, 8],
)


def tiny_model():
    mx.random.seed(0)
    model = Model(ModelConfig.from_dict(TINY))
    model.tokenizer = VitsTokenizer(VOCAB)
    model.eval()  # as the loader leaves it: no dropout
    return model


class TestTokenizer(unittest.TestCase):
    def test_blank_between_every_character(self):
        self.assertEqual(VitsTokenizer(VOCAB).encode("ab"), [0, 1, 0, 2, 0])

    def test_unknown_characters_are_dropped_and_named(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ids = VitsTokenizer(VOCAB).encode("aꞌb")
        self.assertEqual(ids, [0, 1, 0, 2, 0])
        self.assertIn("U+A78C", str(caught[0].message))

    def test_upper_case_is_lowered(self):
        self.assertEqual(
            VitsTokenizer(VOCAB).encode("AB"), VitsTokenizer(VOCAB).encode("ab")
        )


class TestSpline(unittest.TestCase):
    def test_reverse_undoes_forward(self):
        mx.random.seed(1)
        x = mx.random.uniform(-6, 6, (2, 7, 1))
        w, h = mx.random.normal((2, 7, 1, 10)), mx.random.normal((2, 7, 1, 10))
        d = mx.random.normal((2, 7, 1, 9))
        y, ld = unconstrained_rational_quadratic_spline(x, w, h, d)
        x2, ld2 = unconstrained_rational_quadratic_spline(y, w, h, d, reverse=True)
        np.testing.assert_allclose(np.array(x2), np.array(x), atol=1e-4)
        np.testing.assert_allclose(np.array(ld2), -np.array(ld), atol=1e-3)

    def test_identity_outside_the_interval(self):
        x = mx.array([[[-7.0], [9.0]]])
        y, ld = unconstrained_rational_quadratic_spline(
            x, mx.zeros((1, 2, 1, 10)), mx.zeros((1, 2, 1, 10)), mx.zeros((1, 2, 1, 9))
        )
        np.testing.assert_array_equal(np.array(y), np.array(x))
        np.testing.assert_array_equal(np.array(ld), 0)

    def test_gradient_is_finite_at_the_edges(self):
        w, h, d = (
            mx.zeros((1, 3, 1, 10)),
            mx.zeros((1, 3, 1, 10)),
            mx.zeros((1, 3, 1, 9)),
        )
        grad = mx.grad(
            lambda x: unconstrained_rational_quadratic_spline(x, w, h, d)[0].sum()
        )(mx.array([[[-5.0], [5.0], [8.0]]]))
        self.assertTrue(np.isfinite(np.array(grad)).all())


class TestAlignment(unittest.TestCase):
    def test_path_is_monotonic_and_covers_every_token(self):
        rng = np.random.default_rng(0)
        neg_cent = rng.normal(size=(1, 30, 8)).astype(np.float32)
        path = T.maximum_path(neg_cent, [30], [8])[0]
        np.testing.assert_array_equal(path.sum(axis=1), 1)  # one token a frame
        tokens = path.argmax(axis=1)
        self.assertEqual(tokens[0], 0)
        self.assertEqual(tokens[-1], 7)
        self.assertTrue((np.diff(tokens) >= 0).all() and (np.diff(tokens) <= 1).all())

    def test_padding_is_left_empty(self):
        neg_cent = np.zeros((1, 12, 6), np.float32)
        path = T.maximum_path(neg_cent, [9], [4])[0]
        self.assertEqual(path[9:].sum(), 0)
        self.assertEqual(path[:, 4:].sum(), 0)


class TestModel(unittest.TestCase):
    def test_lengths_match_the_waveform(self):
        model = tiny_model()
        result = next(model.generate("abc ab", seed=0))
        self.assertGreater(result.samples, 0)
        self.assertEqual(result.sample_rate, 1024)

    def test_padded_batch_matches_a_single_item(self):
        model = tiny_model()
        a, b = model.tokenizer.encode("abcab"), model.tokenizer.encode("ab")
        single, n = model(mx.array([a]), noise_scale=0.0, noise_scale_duration=0.0)
        ids = mx.array([a, b + [0] * (len(a) - len(b))])
        mask = mx.array([[1] * len(a), [1] * len(b) + [0] * (len(a) - len(b))])
        batch, _ = model(ids, mask, noise_scale=0.0, noise_scale_duration=0.0)
        k = int(n[0].item())
        np.testing.assert_allclose(
            np.array(batch[0, :k]), np.array(single[0, :k]), atol=1e-4
        )

    def test_transformers_layout_round_trips(self):
        model = tiny_model()
        T.prepare_for_training(model)  # weight-normed layers too
        weights = model.to_transformers()
        again = dict(tree_flatten(model.parameters()))
        for k, v in model.sanitize(weights).items():
            np.testing.assert_array_equal(np.array(v), np.array(again[k]), err_msg=k)

    def test_weights_in_mlx_layout_load_unchanged(self):
        model = tiny_model()
        ids = mx.array([model.tokenizer.encode("abc")])
        want, _ = model(ids, noise_scale=0.0, noise_scale_duration=0.0)
        again = Model(ModelConfig.from_dict(TINY))
        again.eval()
        saved = dict(tree_flatten(model.parameters()))  # as mlx-audio saves
        again.load_weights(list(again.sanitize(saved).items()))
        got, _ = again(ids, noise_scale=0.0, noise_scale_duration=0.0)
        np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-5)

    def test_weight_norm_keeps_the_output(self):
        model = tiny_model()
        ids = mx.array([model.tokenizer.encode("abc")])
        before, _ = model(ids, noise_scale=0.0, noise_scale_duration=0.0)
        for conv in model.decoder.weight_normed_convs():
            conv.apply_weight_norm()
        after, _ = model(ids, noise_scale=0.0, noise_scale_duration=0.0)
        np.testing.assert_allclose(np.array(after), np.array(before), atol=1e-5)


class TestTraining(unittest.TestCase):
    def test_a_step_trains_and_the_result_loads(self):
        model = tiny_model()
        mx.random.seed(0)
        disc = T.Discriminator(DISCRIMINATOR)
        disc.apply_weight_norm()
        T.prepare_for_training(model)
        config = T.TrainConfig(
            n_fft=64, hop_length=16, n_mels=8, segment_size=128, batch_size=2
        )
        trainer = T.Trainer(model, disc, config)

        rng = np.random.default_rng(0)
        clips = []
        for text, seconds in (("abcab", 1.0), ("ab", 0.7)):
            wave = (0.1 * rng.normal(size=int(1024 * seconds))).astype(np.float32)
            mag, mel = trainer.spectrogram(mx.array(wave)[None])
            clips.append(
                dict(
                    ids=np.array(model.tokenizer.encode(text), np.int32),
                    waveform=wave,
                    labels=np.array(mag[0]),
                    mel=np.array(mel[0]),
                )
            )
        batch = T.collate(clips)
        before = {k: np.array(v) for k, v in tree_flatten(model.parameters())}
        trainer.set_epoch(0)
        losses = trainer.step(batch)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()), losses)
        changed = sum(
            not np.array_equal(before[k], np.array(v))
            for k, v in tree_flatten(model.parameters())
        )
        self.assertGreater(changed, len(before) // 2)

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "out"
            source.mkdir()
            (source / "config.json").write_text(
                json.dumps(dict(TINY, model_type="vits"))
            )
            (source / "vocab.json").write_text(json.dumps(VOCAB))
            T.save(model, source, output)
            loaded = Model(
                ModelConfig.from_dict(json.loads((output / "config.json").read_text()))
            )
            loaded.load_weights(
                list(
                    loaded.sanitize(
                        dict(mx.load(str(output / "model.safetensors")))
                    ).items()
                )
            )
            model.eval()
            loaded.eval()
            ids = mx.array([model.tokenizer.encode("abc")])
            want, _ = model(ids, noise_scale=0.0, noise_scale_duration=0.0)
            got, _ = loaded(ids, noise_scale=0.0, noise_scale_duration=0.0)
            np.testing.assert_allclose(np.array(got), np.array(want), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
