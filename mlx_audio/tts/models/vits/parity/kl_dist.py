"""The KL loss in training mode, 30 passes each in PyTorch and MLX, with
dropout and layer drop on: their spread, since single draws can't be
matched. Also with layer drop off, which is what makes the loss spike.

Runs in the reference venv, like parity_train.py:

    PYTHONPATH=<mlx-audio>:<work>/fhv <work>/venv/bin/python kl_dist.py <work>

Prints the spread; it has no pass or fail. Measured on an M2 Max (median,
max, passes above 5, torch | mlx):

    eval         1.33  1.56  0 | 1.29  1.58  0
    train        2.41 32.80  5 | 1.81 40.71  3
    nolayerdrop  1.57  1.87  0 | 1.55  1.78  0

Both spike now and then with layer drop on, and neither does with it off.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
S = Path(sys.argv[1])
sys.path.insert(0, str(S / "fhv"))
import run_vits_finetuning as R
from monotonic_align import maximum_path as t_mas
from utils import VitsModelForPreTraining

from mlx_audio.tts.models.vits import train as T

ck = S / "mms-tts-blt-train"
model, disc, _, _ = T.load_for_training(str(ck))
model.train()
tr = T.Trainer(model, disc, T.TrainConfig(batch_size=8))
clips = T.load_clips(S / "smoke_data/train", model, tr.spectrogram, tr.config)[:8]
mb = T.collate(clips)
tm = VitsModelForPreTraining.from_pretrained(ck)
del tm.discriminator
tm.train()


def tb(a):
    return torch.tensor(np.array(a))


t_ids = tb(mb["input_ids"]).long()
t_am = tb(mb["attention_mask"])
t_lab = tb(mb["labels"]).transpose(1, 2)
t_lm = tb(mb["labels_attention_mask"])


def kl_m(mode):
    if mode == "eval":
        model.eval()
    else:
        model.train()
    if mode == "nolayerdrop":
        model.text_encoder.encoder.layerdrop = 0.0
    o = T.training_forward(model, mb, 32)
    k = T.kl_loss(
        o["prior_latents"],
        o["posterior_log_variances"],
        o["prior_means"],
        o["prior_log_variances"],
        o["labels_padding_mask"],
    ).item()
    model.text_encoder.encoder.layerdrop = 0.1
    return k


def kl_t(mode):
    tm.train() if mode != "eval" else tm.eval()
    tm.text_encoder.encoder.layerdrop = 0.0 if mode == "nolayerdrop" else 0.1
    with torch.no_grad():
        o = tm(
            input_ids=t_ids,
            attention_mask=t_am,
            labels=t_lab,
            labels_attention_mask=t_lm,
            return_dict=True,
            monotonic_alignment_function=t_mas,
        )
    tm.text_encoder.encoder.layerdrop = 0.1
    return R.kl_loss(
        o.prior_latents,
        o.posterior_log_variances,
        o.prior_means,
        o.prior_log_variances,
        o.labels_padding_mask,
    ).item()


def spread(v):
    return (
        f"median {np.median(v):6.2f} max {np.max(v):7.2f} "
        f">5: {sum(x > 5 for x in v):2d}/30"
    )


for mode in ("eval", "train", "nolayerdrop"):
    a = [kl_t(mode) for _ in range(30)]
    b = [kl_m(mode) for _ in range(30)]
    print(f"{mode:12s} torch {spread(a)} | mlx {spread(b)}", flush=True)
