"""Small helpers shared by the analysis and plotting scripts.

MFA writes Praat "long" TextGrids with two interval tiers, "words" and
"phones". Empty intervals are silence. Parsing that format needs only a few
regular expressions, so no extra TextGrid package is required.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SILENCE = {"", "sil", "sp", "<eps>"}
UNKNOWN = {"spn", "<unk>"}          # what MFA writes for words it has no pronunciation for


@dataclass
class Interval:
    start: float
    end: float
    label: str

    @property
    def dur(self) -> float:
        return self.end - self.start


_TIER = re.compile(r'name = "(?P<name>[^"]*)"(?P<body>.*?)(?=\n\s*item \[\d+\]:|\Z)', re.S)
_IV = re.compile(r'xmin = (?P<s>[\d.eE+-]+)\s+xmax = (?P<e>[\d.eE+-]+)\s+text = "(?P<t>(?:[^"]|"")*)"')


def read_textgrid(path: Path, keep_silence: bool = False) -> dict[str, list[Interval]]:
    """Return {tier name: intervals}. Silence is dropped unless keep_silence."""
    text = path.read_text(encoding="utf-8")
    tiers: dict[str, list[Interval]] = {}
    for tier in _TIER.finditer(text):
        ivs = [Interval(float(m["s"]), float(m["e"]), m["t"].replace('""', '"').strip())
               for m in _IV.finditer(tier["body"])]
        if not keep_silence:
            ivs = [iv for iv in ivs if iv.label not in SILENCE]
        tiers[tier["name"]] = ivs
    return tiers


def textgrids(outputs: Path) -> dict[str, dict[str, Path]]:
    """{experiment: {utterance id: TextGrid path}} for every run in outputs/."""
    runs: dict[str, dict[str, Path]] = {}
    for exp in sorted(p for p in outputs.iterdir() if p.is_dir()):
        tgs = {tg.stem: tg for tg in exp.rglob("*.TextGrid")}
        if tgs:
            runs[exp.name] = tgs
    return runs


def phones_in(word: Interval, phones: list[Interval]) -> list[Interval]:
    """Phones whose midpoint falls inside a word interval."""
    return [p for p in phones if word.start <= (p.start + p.end) / 2 < word.end]
