# Parity checks

Scripts that compare this port with its sources, on Tai Dam (`facebook/mms-tts-blt`).
They are not part of the test suite: they download several GB, need PyTorch, and
two of them need a reference built first. Run them after changing the model or the
trainer, or to check another port (a Swift one, say) against this one.

Each check prints its figures and exits 1 when one is outside its tolerance. The
tolerances sit 10 to 40 times above what an M2 Max measured.

| Script | Compares | Against | Measured |
|---|---|---|---|
| `parity_infer.py` | tokens, text encoder, durations, waveform, a padded batch | transformers' `VitsModel`, noise off | waveform within 6.0e-5 |
| `parity_train.py` | spectrograms, alignment, forward outputs, every loss, gradients, one AdamW step | finetune-hf-vits, dropout off, same noise | losses within 2.5e-6, gradient norms 0.05% |
| `kl_dist.py` | the KL loss's spread over 30 training-mode passes | finetune-hf-vits, dropout and layer drop on | (prints only) |
| `asr_check.py` | character error of the voice, read back by MMS's Tai Dam recogniser | `facebook/mms-1b-all`, `blt` adapter | 1.8% |

## Speaking

In mlx-audio's own environment, with PyTorch added:

```sh
pip install torch jiwer soundfile
python mlx_audio/tts/models/vits/parity/parity_infer.py
```

## Training

`setup.sh` builds the reference in a folder of its own: finetune-hf-vits at
`6f3f51f` (patched for a single speaker), a venv with the libraries it was written
for (transformers 4.46.3), `blt`'s checkpoint with its discriminator, Tai Dam verses
from eBible, and 48 clips of the stock voice reading them.

```sh
W=.parity-work
mlx_audio/tts/models/vits/parity/setup.sh $W
PYTHONPATH=$PWD:$W/fhv $W/venv/bin/python mlx_audio/tts/models/vits/parity/parity_train.py $W
PYTHONPATH=$PWD:$W/fhv $W/venv/bin/python mlx_audio/tts/models/vits/parity/kl_dist.py $W
python mlx_audio/tts/models/vits/parity/asr_check.py $W
```

It is safe to run again: each step is skipped once done. If the Hub cache lives on
a drive that isn't there, set `HF_HOME` for every command.

## Licences

The MMS weights (`mms-tts-blt`, `mms-1b-all`) are CC-BY-NC 4.0. The Tai Dam
New Testament the verses come from (Wycliffe Bible Translators, 2020) is all rights
reserved: setup downloads it for private checks, and nothing made from it should be
shared.
