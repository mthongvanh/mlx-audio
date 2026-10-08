"""Verses from eBible chapter pages, as plain lines in NFC.

python prep_text.py <out.json> <chapter.html>...
"""

import html
import json
import re
import sys
import unicodedata

out = []
for path in sys.argv[2:]:
    t = open(path, encoding="utf-8").read()
    t = t.split("<div class='s'>", 1)[-1]  # past the navigation
    t = re.sub(r"<span class='[^']*v-num[^']*'>.*?</span>", "\n", t)
    t = re.sub(r"<div class='s'>.*?</div>", "\n", t)  # section headings
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    for line in t.split("\n"):
        line = unicodedata.normalize("NFC", re.sub(r"\s+", " ", line)).strip()
        line = re.sub(r"[◀▶☰﻿]", "", line).strip()
        if 6 <= len(line.split()) <= 40:
            out.append(line)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=0)
print(f"{len(out)} verses")
