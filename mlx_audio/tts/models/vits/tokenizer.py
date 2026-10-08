"""VITS's character tokenizer, as transformers' `VitsTokenizer` reads text.

One difference: transformers drops characters outside the vocabulary
without a word, which can quietly turn a sentence into another. This one
does the same but says which it dropped.
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Optional


class VitsTokenizer:
    def __init__(
        self,
        vocab: dict,
        add_blank: bool = True,
        normalize: bool = True,
        phonemize: bool = False,
        is_uroman: bool = False,
        language: Optional[str] = None,
        added_tokens: Optional[dict] = None,
    ):
        self.encoder = vocab
        self.added_tokens = added_tokens or {}
        self.add_blank = add_blank
        self.normalize = normalize
        self.phonemize = phonemize
        self.is_uroman = is_uroman
        self.language = language
        # Matched in this order, as transformers does.
        self._vocabulary = list(self.encoder) + list(self.added_tokens)

    @classmethod
    def from_pretrained(cls, path) -> "VitsTokenizer":
        path = Path(path)
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        config = {}
        if (path / "tokenizer_config.json").exists():
            config = json.loads(
                (path / "tokenizer_config.json").read_text(encoding="utf-8")
            )
        added = {}
        if (path / "added_tokens.json").exists():
            added = json.loads((path / "added_tokens.json").read_text(encoding="utf-8"))
        return cls(
            vocab,
            add_blank=config.get("add_blank", True),
            normalize=config.get("normalize", True),
            phonemize=config.get("phonemize", False),
            is_uroman=config.get("is_uroman", False),
            language=config.get("language"),
            added_tokens=added,
        )

    def normalize_text(self, text: str) -> str:
        out = []
        i = 0
        while i < len(text):
            for word in self._vocabulary:
                if text.startswith(word, i):
                    out.append(word)
                    i += len(word)
                    break
            else:
                out.append(text[i].lower())
                i += 1
        return "".join(out)

    def prepare(self, text: str) -> str:
        """The text as the model reads it: normalised, romanised or
        phonemised if the checkpoint asks, and only its own characters."""
        if self.normalize:
            text = self.normalize_text(text)
        if self.language == "ron":
            text = text.replace("ț", "ţ")
        if self.is_uroman and re.search(r"[^\x00-\x7F]", text):
            try:
                import uroman as ur
            except ImportError as e:
                raise ImportError(
                    "This checkpoint reads romanised text. Install `uroman` "
                    "(`pip install uroman`), or romanise the text first."
                ) from e
            text = ur.Uroman().romanize_string(text)
        if self.phonemize:
            try:
                import phonemizer
            except ImportError as e:
                raise ImportError(
                    "This checkpoint reads phonemes. Install `phonemizer` and espeak."
                ) from e
            text = phonemizer.phonemize(
                text,
                language="en-us",
                backend="espeak",
                strip=True,
                preserve_punctuation=True,
                with_stress=True,
            )
            return re.sub(r"\s+", " ", text)
        if self.normalize:
            dropped = sorted({c for c in text if c not in self.encoder})
            if dropped:
                warnings.warn(
                    "VITS: characters not in this voice's vocabulary were dropped: "
                    + " ".join(f"{c!r} (U+{ord(c):04X})" for c in dropped),
                    stacklevel=3,
                )
            text = "".join(c for c in text if c in self.encoder).strip()
        return text

    def encode(self, text: str) -> List[int]:
        tokens = list(self.prepare(text))
        unk = self.encoder.get("<unk>", 0)
        ids = [self.encoder.get(t, self.added_tokens.get(t, unk)) for t in tokens]
        if self.add_blank:
            # Id 0 between every character and at both ends.
            spaced = [0] * (len(ids) * 2 + 1)
            spaced[1::2] = ids
            ids = spaced
        return ids
