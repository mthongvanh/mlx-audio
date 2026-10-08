# MMS-TTS and VITS

Meta's [Massively Multilingual Speech](https://huggingface.co/facebook/mms-tts) voices:
one small VITS model (36M parameters, 16 kHz) for each of about 1,100 languages. Any
other VITS checkpoint in transformers' format loads too.

Ported from transformers' `modeling_vits.py`, with the same weight names, so the
`facebook/mms-tts-*` checkpoints load as they are. On `facebook/mms-tts-blt` the
waveform matches transformers' within 6e-5, with noise off.

**Licence.** The MMS weights are **CC-BY-NC 4.0**: no commercial use, and anything
fine-tuned from them carries the same terms. VITS itself, and this port, are MIT.

## Speech

```python
from mlx_audio.audio_io import write
from mlx_audio.tts.utils import load

model = load("facebook/mms-tts-eng")
for result in model.generate("Hello from MLX.", seed=0):
    write("hello.wav", result.audio, result.sample_rate)
```

`speed` scales the speaking rate. `noise_scale` (0.667) and `noise_scale_duration`
(0.8) set how much the voice and its timing vary. `seed` makes a result repeatable.

**Spell the text as the voice expects.** Each voice reads only the characters in its
`vocab.json`. Others are dropped, and the tokenizer warns which. transformers drops
them without a word. Check the vocabulary against real text before trusting a voice:
a language's usual spelling may use marks the voice was trained without. Some voices
read romanised text (`is_uroman` in `tokenizer_config.json`); install `uroman` for those.

## Fine-tuning

A port of [finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) to MLX:
the same losses and weights, the same order of discriminator and generator updates,
AdamW with PyTorch's defaults, the learning rate decayed once an epoch. One step,
checked against the PyTorch recipe on the same batch and noise, gives every loss
within 2.5e-6 and the gradients' global norm within 0.05%. On an M2 Max it runs at
about 1.4 s a step at batch 8, against about 2 s for PyTorch on the same Mac's GPU.

1. **A checkpoint with its discriminator.** The Hub's voices have none. Make one with
   finetune-hf-vits's converter (it needs PyTorch):

   ```sh
   python convert_original_discriminator_checkpoint.py \
     --language_code eng --pytorch_dump_folder_path mms-tts-eng-train
   ```

2. **Data:** a folder of clips (mono, any sample rate; 1 to 20 seconds each) and a
   `metadata.jsonl` with one `{"file_name": "clip1.wav", "text": "..."}` a line.
   80 to 150 clips of one speaker is a start; more is better.

3. **Train:**

   ```sh
   python -m mlx_audio.tts.models.vits.train \
     --model mms-tts-eng-train --data my-voice --output my-voice-tts \
     --epochs 200 --batch-size 16
   ```

   `--text-map` takes a JSON object of replacements made to each transcript first,
   for spellings the voice's vocabulary lacks. `--layerdrop 0` stops the text
   encoder skipping layers; finetune-hf-vits keeps the checkpoint's 0.1, which makes
   the KL loss spike now and then (PyTorch does the same).

The output is a plain transformers checkpoint: it loads here, in transformers'
`VitsModel`, and in anything that reads those.

## Notes

- **transformers' duration-predictor likelihood flips the channels after the first
  (affine) flow**; the original VITS and finetune-hf-vits don't. transformers never
  trains, so its inference is unaffected. Training here follows the original.
- **Every batch has its own shape**, so the trainer clears MLX's buffer cache each
  step. Kept, it grew by about 5 GB a step until the Mac swapped.
