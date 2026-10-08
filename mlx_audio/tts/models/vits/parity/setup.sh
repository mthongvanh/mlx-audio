#!/usr/bin/env bash
# Builds the PyTorch reference the parity checks compare against, in a
# folder of its own: finetune-hf-vits with its pinned 2024 libraries, blt's
# training checkpoint (discriminator included), Tai Dam verses, and a small
# dataset the stock voice reads itself.
#
#   mlx_audio/tts/models/vits/parity/setup.sh <work dir>
#
# Needs Python 3.12, git and curl, and about 3 GB of downloads. Run it from
# anywhere; nothing is written outside <work dir> except pip's and the Hub's
# caches.
set -euo pipefail

WORK=${1:?usage: setup.sh <work dir>}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3.12}
mkdir -p "$WORK" && WORK=$(cd "$WORK" && pwd)

# finetune-hf-vits at its last commit (February 2024).
if [ ! -d "$WORK/fhv" ]; then
  git clone -q https://github.com/ylacombe/finetune-hf-vits.git "$WORK/fhv"
  git -C "$WORK/fhv" checkout -q 6f3f51f
  # One speaker: the script reads batch["speaker_id"], which isn't there
  # without a speaker column.
  sed -i.orig 's/speaker_id=batch\["speaker_id"\]/speaker_id=batch.get("speaker_id")/' \
    "$WORK/fhv/run_vits_finetuning.py"
fi

# The reference venv. transformers 4.46 is the last the recipe runs on;
# mlx is the version the port was checked with.
if [ ! -x "$WORK/venv/bin/python" ]; then
  "$PY" -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q --upgrade pip
  "$WORK/venv/bin/pip" install -q torch torchaudio "transformers==4.46.3" \
    "datasets[audio]==2.21.0" "accelerate==0.34.2" "matplotlib<3.10" \
    Cython tensorboard soundfile librosa safetensors jiwer \
    "mlx==0.32.3" miniaudio scipy
fi

# The alignment search, as a C extension.
if ! ls "$WORK/fhv/monotonic_align/monotonic_align/"core*.so >/dev/null 2>&1; then
  (cd "$WORK/fhv/monotonic_align" && mkdir -p monotonic_align &&
    "$WORK/venv/bin/python" setup.py build_ext --inplace >/dev/null)
fi

# blt's checkpoint with its discriminator. Not with `python -I`, which
# hides the repo's utils.
if [ ! -f "$WORK/mms-tts-blt-train/model.safetensors" ]; then
  (cd "$WORK/fhv" && "$WORK/venv/bin/python" convert_original_discriminator_checkpoint.py \
    --language_code blt --pytorch_dump_folder_path "$WORK/mms-tts-blt-train" >/dev/null)
fi

# Tai Dam verses from eBible (Mark 1, John 1), for text the voice knows.
# The text is all rights reserved: for private checks only.
mkdir -p "$WORK/text"
for ch in MK1 JN1; do
  [ -f "$WORK/text/$ch.html" ] ||
    curl -sfL -o "$WORK/text/$ch.html" "https://ebible.org/study/content/texts/blt/$ch.html"
done
"$WORK/venv/bin/python" -I "$HERE/prep_text.py" "$WORK/text/verses.json" \
  "$WORK/text/MK1.html" "$WORK/text/JN1.html"

# A stand-in dataset: 40 training and 8 test clips of the stock voice.
[ -f "$WORK/smoke_data/train/metadata.jsonl" ] ||
  "$WORK/venv/bin/python" "$HERE/smoke_data.py" "$WORK"

echo "reference ready in $WORK"
