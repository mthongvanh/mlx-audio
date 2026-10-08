"""One training step of finetune-hf-vits (PyTorch) and the MLX port, side by
side: spectrograms, alignment, forward outputs, every loss, every gradient,
and the parameters after one update of each network. Dropout off in both;
the same noise fed to both.

Runs in the reference venv that setup.sh builds, with this repo and the
recipe on the path:

    PYTHONPATH=<mlx-audio>:<work>/fhv <work>/venv/bin/python parity_train.py <work>

Exits 1 if any figure is outside its tolerance.
"""

import json
import math
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
import torch
from mlx.utils import tree_flatten

warnings.filterwarnings("ignore")

S = Path(sys.argv[1])
ckpt = S / "mms-tts-blt-train"
sys.path.insert(0, str(S / "fhv"))
from monotonic_align import maximum_path as t_mas
from transformers import VitsTokenizer
from utils import VitsFeatureExtractor, VitsModelForPreTraining
from utils import slice_segments as t_slice

from mlx_audio.tts.models.vits import train as T

# Relative error allowed for each group of figures. Measured on an M2 Max:
# spectrograms 9e-7, losses 2.5e-6, gradient norms 0.05%, all updates 3.4e-2
# (Adam's first step turns rounding in tiny gradients into whole steps).
TOLERANCE = {
    "spectrogram": 1e-5,
    "forward": 1e-3,
    "loss": 1e-4,
    "grad norm": 5e-3,
    "updates": 1e-1,
}
failures = []


def check(name, value, group):
    if value is None or not value <= TOLERANCE[group]:
        failures.append(name)
        return "  FAIL"
    return ""


torch.manual_seed(0)
np.random.seed(0)


def cmp(name, a, b, group):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {name}: SHAPE {a.shape} vs {b.shape}")
        failures.append(name)
        return
    rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)
    flag = check(name, rel, group)
    print(f"  {name:34s} rel {rel:.2e}  max|diff| {np.abs(a-b).max():.2e}{flag}")


# --- Data: two clips of different lengths, as the collator pads them.
rows = [
    json.loads(l)
    for l in (S / "smoke_data/train/metadata.jsonl").read_text().splitlines()
][:2]
tok = VitsTokenizer.from_pretrained(ckpt)
fe = VitsFeatureExtractor.from_pretrained(ckpt)
waves = [
    sf.read(S / "smoke_data/train" / r["file_name"])[0].astype(np.float32) for r in rows
]
ids = [tok(r["text"]).input_ids for r in rows]

# --- PyTorch: the reference, set up as run_vits_finetuning.py sets it up.
tm = VitsModelForPreTraining.from_pretrained(ckpt)
tm.decoder.apply_weight_norm()
for flow in tm.flow.flows:
    torch.nn.utils.weight_norm(flow.conv_pre)
    torch.nn.utils.weight_norm(flow.conv_post)
tdisc = tm.discriminator
for d in tdisc.discriminators:
    d.apply_weight_norm()
del tm.discriminator
tm.eval()
tdisc.eval()

# --- MLX.
model, mdisc, _, _ = T.load_for_training(str(ckpt))
model.eval()
cfg = T.TrainConfig(batch_size=2)
trainer = T.Trainer(model, mdisc, cfg)

print("spectrograms")
t_feats = [fe(w, sampling_rate=16000, return_tensors="pt") for w in waves]
for i, w in enumerate(waves):
    mag, mel = trainer.spectrogram(mx.array(w)[None])
    cmp(
        f"magnitudes {i}",
        t_feats[i]["input_features"][0].T.numpy(),
        np.array(mag[0]),
        "spectrogram",
    )
    cmp(
        f"log-mel {i}",
        t_feats[i]["mel_scaled_input_features"][0].T.numpy(),
        np.array(mel[0]),
        "spectrogram",
    )

# Batches. MLX from its own features; PyTorch from the feature extractor's.
clips = []
for i, w in enumerate(waves):
    mag, mel = trainer.spectrogram(mx.array(w)[None])
    clips.append(
        dict(
            ids=np.array(ids[i], np.int32),
            waveform=w,
            labels=np.array(mag[0]),
            mel=np.array(mel[0]),
        )
    )
mb = T.collate(clips)


def tpad(arrs):
    """(time, ...) arrays to a zero-padded batch and its mask."""
    n = max(len(a) for a in arrs)
    out = np.zeros((len(arrs), n) + arrs[0].shape[1:], np.float32)
    m = np.zeros((len(arrs), n), np.float32)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
        m[i, : len(a)] = 1
    return torch.tensor(out), torch.tensor(m)


t_ids, t_am = tpad([np.array(x, np.float32) for x in ids])
t_ids = t_ids.long()
t_labels, t_lm = tpad([f["input_features"][0].T.numpy() for f in t_feats])
t_labels = t_labels.transpose(1, 2)
t_mel, _ = tpad([f["mel_scaled_input_features"][0].T.numpy() for f in t_feats])
t_mel = t_mel.transpose(1, 2)
t_wave, _ = tpad([w[:, None] for w in waves])
print("frames:", [int(x) for x in t_lm.sum(1)], "tokens:", [len(x) for x in ids])

# --- The same noise for both.
B, Ts, Tt = t_labels.shape[0], t_labels.shape[2], t_ids.shape[1]
post_noise = torch.randn(B, 192, Ts)
dur_noise = torch.randn(B, 2, Tt)
seg = cfg.segment_size // cfg.hop_length
lengths = t_lm.sum(1)
starts = (torch.rand(B) * (lengths - seg + 1)).long()
noise = dict(
    posterior_noise=mx.array(post_noise.transpose(1, 2).numpy()),
    duration_noise=mx.array(dur_noise.transpose(1, 2).numpy()),
    slice_starts=mx.array(starts.numpy().astype(np.int32)),
)


class Fixed:
    """torch.randn_like, torch.randn and torch.rand, returning our draws."""

    def __enter__(self):
        self.saved = torch.randn_like, torch.randn, torch.rand
        torch.randn_like = lambda x, **k: post_noise.clone()
        torch.randn = lambda *a, **k: dur_noise.clone()
        torch.rand = lambda *a, **k: (starts.float() + 0.5) / (lengths - seg + 1)
        return self

    def __exit__(self, *a):
        torch.randn_like, torch.randn, torch.rand = self.saved


print("alignment search")
rng = np.random.default_rng(0)
nc = rng.normal(size=(2, 90, 40)).astype(np.float32) * 5
mask = np.zeros((2, 90, 40), np.float32)
mask[0, :90, :40] = 1
mask[1, :70, :25] = 1
ref = t_mas(torch.tensor(nc), torch.tensor(mask)).numpy()
mine = T.maximum_path(nc, [90, 70], [40, 25])
same_paths = np.array_equal(ref, mine)
print("  identical paths:", same_paths)
if not same_paths:
    failures.append("alignment search")

print("training forward")
with Fixed():
    out_t = tm(
        input_ids=t_ids,
        attention_mask=t_am,
        labels=t_labels,
        labels_attention_mask=t_lm,
        return_dict=True,
        monotonic_alignment_function=t_mas,
    )
out_m = T.training_forward(model, mb, seg, **noise)
cmp("log_duration", out_t.log_duration.detach(), out_m["log_duration"], "forward")
cmp("alignment", out_t.attn[:, 0].detach(), out_m["attn"], "forward")
cmp(
    "prior_latents",
    out_t.prior_latents.detach().transpose(1, 2),
    out_m["prior_latents"],
    "forward",
)
cmp(
    "prior_means (expanded)",
    out_t.prior_means.detach().transpose(1, 2),
    out_m["prior_means"],
    "forward",
)
cmp(
    "posterior_log_variances",
    out_t.posterior_log_variances.detach().transpose(1, 2),
    out_m["posterior_log_variances"],
    "forward",
)
cmp(
    "waveform slice",
    out_t.waveform.detach()[:, 0],
    out_m["waveform"][..., 0],
    "forward",
)

print("losses and gradients")
import run_vits_finetuning as R

mel_target = t_slice(t_mel, out_t.ids_slice, seg)
mel_gen = fe._torch_extract_fbank_features(out_t.waveform.squeeze(1))[1]
wave_target = t_slice(
    t_wave.transpose(1, 2), out_t.ids_slice * cfg.hop_length, cfg.segment_size
)
d_real, _ = tdisc(wave_target)
d_fake, _ = tdisc(out_t.waveform.detach())
t_ld, t_lr, t_lf = R.discriminator_loss(d_real, d_fake)
tdisc.zero_grad()
(t_ld * cfg.weight_disc).backward()
t_disc_grads = {n: p.grad.clone() for n, p in tdisc.named_parameters()}

fake_m = mx.stop_gradient(out_m["waveform"])
_, wave_target_m = trainer._targets(mb, out_m)
(_, (m_ld, m_lr, m_lf)), m_disc_grads = trainer.disc_grad(fake_m, wave_target_m)
cmp("loss disc", t_ld.item(), m_ld.item(), "loss")
cmp("loss real disc", t_lr.item(), m_lr.item(), "loss")

# The generator's loss, against the discriminator before its update, on
# both sides.
_, fm_t = tdisc(wave_target)
d_gen, fm_g = tdisc(out_t.waveform)
lt = dict(
    duration=torch.sum(out_t.log_duration),
    mel=torch.nn.functional.l1_loss(mel_target, mel_gen),
    kl=R.kl_loss(
        out_t.prior_latents,
        out_t.posterior_log_variances,
        out_t.prior_means,
        out_t.prior_log_variances,
        out_t.labels_padding_mask,
    ),
    fmaps=R.feature_loss(fm_t, fm_g),
    gen=R.generator_loss(d_gen)[0],
)
t_total = sum(lt[k] * getattr(cfg, "weight_" + k) for k in lt)
tm.zero_grad()
tdisc.zero_grad()
t_total.backward()
t_gen_grads = {
    n: p.grad.clone() for n, p in tm.named_parameters() if p.grad is not None
}
(m_total, m_parts), m_gen_grads = trainer.gen_grad(mb, noise)
for k in lt:
    cmp(f"loss {k}", lt[k].item(), m_parts[k].item(), "loss")
cmp("loss total (weighted)", t_total.item(), m_total.item(), "loss")


def torch_name(k):
    return [
        k,
        k.replace(".weight_g", ".parametrizations.weight.original0").replace(
            ".weight_v", ".parametrizations.weight.original1"
        ),
    ]


def to_torch_layout(k, v):
    v = np.array(v)
    if k.endswith((".translate", ".log_scale")):
        return v.reshape(-1, 1)
    if v.ndim == 4:
        return v.transpose(0, 3, 1, 2)
    if v.ndim == 3 and not k.endswith(("emb_rel_k", "emb_rel_v")):
        if k.startswith("decoder.upsampler."):
            return (
                v.reshape(-1, 1, 1) if k.endswith("weight_g") else v.transpose(2, 0, 1)
            )
        if k.endswith("weight_g"):
            return v
        return v.transpose(0, 2, 1)
    return v


def grad_report(label, mgrads, tgrads):
    rels, missing = [], []
    for k, g in tree_flatten(mgrads):
        names = [n for n in torch_name(k) if n in tgrads]
        if not names:
            missing.append(k)
            continue
        a = tgrads[names[0]].numpy().astype(np.float64)
        b = to_torch_layout(k, g).astype(np.float64)
        if a.shape != b.shape:
            missing.append(f"{k} {a.shape} vs {b.shape}")
            continue
        na = np.linalg.norm(a)
        if na < 1e-12 and np.linalg.norm(b) < 1e-12:
            continue
        rels.append((np.linalg.norm(a - b) / (na + 1e-30), na, k))
    rels.sort(reverse=True)
    r = np.array([x[0] for x in rels])
    print(
        f"  {label}: {len(rels)} tensors, rel error median {np.median(r):.1e}, "
        f"95th {np.percentile(r, 95):.1e}, worst {r.max():.1e} "
        f"({rels[0][2]}, |g| {rels[0][1]:.1e})"
    )
    if missing:
        print(f"    unmatched: {missing[:5]} ({len(missing)})")
        failures.append(f"{label} unmatched")
    tot_t = math.sqrt(sum(float((t.double() ** 2).sum()) for t in tgrads.values()))
    tot_m = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in tree_flatten(mgrads)))
    flag = check(f"{label} norm", abs(tot_t - tot_m) / tot_t, "grad norm")
    print(f"    global grad norm torch {tot_t:.4e}  mlx {tot_m:.4e}{flag}")


grad_report("discriminator grads", m_disc_grads, t_disc_grads)
grad_report("generator grads", m_gen_grads, t_gen_grads)

print("one full step (disc then gen, AdamW, clipping)")
# PyTorch, as the script does it.
ad = dict(
    lr=cfg.learning_rate * cfg.lr_decay,
    betas=(cfg.adam_beta1, cfg.adam_beta2),
    eps=cfg.adam_epsilon,
)
g_opt = torch.optim.AdamW(tm.parameters(), **ad)
d_opt = torch.optim.AdamW(tdisc.parameters(), **ad)
before = {n: p.detach().clone() for n, p in tm.named_parameters()}
with Fixed():
    o = tm(
        input_ids=t_ids,
        attention_mask=t_am,
        labels=t_labels,
        labels_attention_mask=t_lm,
        return_dict=True,
        monotonic_alignment_function=t_mas,
    )
mt = t_slice(t_mel, o.ids_slice, seg)
wt = t_slice(t_wave.transpose(1, 2), o.ids_slice * cfg.hop_length, cfg.segment_size)
mg = fe._torch_extract_fbank_features(o.waveform.squeeze(1))[1]
dr, _ = tdisc(wt)
dfk, _ = tdisc(o.waveform.detach())
ld, _, _ = R.discriminator_loss(dr, dfk)
d_opt.zero_grad()
(ld * cfg.weight_disc).backward()
torch.nn.utils.clip_grad_norm_(tdisc.parameters(), cfg.max_grad_norm)
d_opt.step()
d_opt.zero_grad()
_, f1 = tdisc(wt)
dg, f2 = tdisc(o.waveform)
tot = (
    torch.sum(o.log_duration) * cfg.weight_duration
    + torch.nn.functional.l1_loss(mt, mg) * cfg.weight_mel
    + R.kl_loss(
        o.prior_latents,
        o.posterior_log_variances,
        o.prior_means,
        o.prior_log_variances,
        o.labels_padding_mask,
    )
    * cfg.weight_kl
    + R.feature_loss(f1, f2) * cfg.weight_fmaps
    + R.generator_loss(dg)[0] * cfg.weight_gen
)
g_opt.zero_grad()
tot.backward()
torch.nn.utils.clip_grad_norm_(tm.parameters(), cfg.max_grad_norm)
g_opt.step()
# MLX.
mbefore = dict(tree_flatten(model.parameters()))
trainer.set_epoch(0)
losses = trainer.step(mb, noise)
rels = []
num = den = 0.0
for k, v in tree_flatten(model.parameters()):
    names = [n for n in torch_name(k) if n in before]
    if not names:
        continue
    n = names[0]
    dt = (
        (dict(tm.named_parameters())[n].detach() - before[n]).numpy().astype(np.float64)
    )
    dm = to_torch_layout(k, v - mbefore[k]).astype(np.float64)
    num += float(((dt - dm) ** 2).sum())
    den += float((dt**2).sum())
    if np.linalg.norm(dt) < 1e-12:
        continue
    rels.append(np.linalg.norm(dt - dm) / np.linalg.norm(dt))
r = np.array(rels)
print(
    f"  parameter updates: {len(r)} tensors, rel error median {np.median(r):.1e}, "
    f"95th {np.percentile(r, 95):.1e}, worst {r.max():.1e}"
)
all_updates = math.sqrt(num / den)
flag = check("all updates", all_updates, "updates")
print(f"  all parameter updates as one vector: rel error {all_updates:.1e}{flag}")
print("  mlx step losses: " + " ".join(f"{k} {v:.4f}" for k, v in losses.items()))
w = {k: getattr(cfg, "weight_" + k) for k in ("duration", "mel", "kl", "fmaps", "gen")}
m_weighted = sum(losses[k] * w[k] for k in w)
flag = check(
    "weighted generator loss", abs(tot.item() - m_weighted) / abs(tot.item()), "loss"
)
print(f"  weighted generator loss: torch {tot.item():.4f}  mlx {m_weighted:.4f}{flag}")

print("PASS" if not failures else f"FAIL: {', '.join(failures)}")
sys.exit(1 if failures else 0)
