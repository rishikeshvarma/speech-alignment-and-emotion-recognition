"""Load the cached encoder features (see scripts/extract_features.py) as batches.

Every item carries both modalities, so the same loader serves all three
pipelines; each model simply ignores the inputs it does not use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data import EMO2ID, WORK_DIR, load_metadata

FEATURES = WORK_DIR / "features"
SPEECH_DIMS = {"wavlm": (13, 768), "mel": (1, 80)}     # (layers, feature size)
TEXT_DIMS = (13, 768)


def clip_id(row: dict) -> str:
    return Path(row["path"]).stem


def speaker_stats(kind: str, root: Path = FEATURES) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Mean and standard deviation of every feature dimension (per encoder
    layer), over all frames of all clips of each speaker. No labels are used.
    Cached in <features>/speaker_stats_<kind>.npz."""
    cache = root / f"speaker_stats_{kind}.npz"
    if not cache.exists():
        acc: dict[str, list] = {}
        for r in load_metadata():
            arr = np.load(root / kind / f"{clip_id(r)}.npy").astype(np.float64)
            if arr.ndim == 2:
                arr = arr[None]
            a = acc.setdefault(r["speaker"], [0, 0.0, 0.0])
            a[0] += arr.shape[1]
            a[1] = a[1] + arr.sum(1)
            a[2] = a[2] + (arr ** 2).sum(1)
        out = {}
        for spk, (n, s1, s2) in acc.items():
            mean = s1 / n
            out[f"{spk}_mean"] = mean.astype(np.float32)
            out[f"{spk}_std"] = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-8)).astype(np.float32)
        np.savez(cache, **out)
    z = np.load(cache)
    speakers = {k.rsplit("_", 1)[0] for k in z.files}
    return {s: (torch.from_numpy(z[f"{s}_mean"]), torch.from_numpy(z[f"{s}_std"])) for s in speakers}


class FeatureStore:
    """Reads features from disk once and keeps them in memory (fp16).

    speech=None skips the speech features (the text pipeline does not need
    the 5 GB of WavLM frames) and returns a 1x1x1 placeholder instead.

    speaker_norm=True standardises each speaker's speech features with that
    speaker's own mean and variance (speaker-level CMVN). This removes
    constant differences between the two voices (age, vocal tract, recording
    level) while keeping the variation within a speaker that carries emotion.
    It needs unlabelled audio of the test speaker, which is available here.
    """

    def __init__(self, speech: str | None = "wavlm", root: Path = FEATURES, speaker_norm: bool = False,
                 text_input: str = "transcript"):
        self.speech_kind = speech
        self.root = root
        # Controls for the fusion analysis:
        #   "zeros"    - the text branch gets an all-zero input (same architecture, no text)
        #   "shuffled" - every clip gets the transcript of a *different* word (a fixed
        #                derangement of the 200 transcripts): text that varies like the real
        #                one but does not match what was said
        self.text_input = text_input
        self._speech: dict[str, torch.Tensor] = {}
        self._dummy = torch.zeros(1, 1, 1, dtype=torch.float16)
        self._stats = speaker_stats(speech, root) if (speaker_norm and speech) else None
        self._speaker = {clip_id(r): r["speaker"] for r in load_metadata()} if self._stats else {}
        z = np.load(root / "bert_text.npz")
        self._text = {t: torch.from_numpy(z["feats"][i, :, :n]) for i, (t, n) in
                      enumerate(zip(z["texts"].tolist(), z["lengths"].tolist()))}
        texts = sorted(self._text)
        order = np.random.default_rng(0).permutation(len(texts))
        # rotate a random order by one: a permutation in which no text maps to itself
        self._shuffled = {texts[order[k]]: texts[order[(k + 1) % len(texts)]] for k in range(len(texts))}

    def speech(self, cid: str) -> torch.Tensor:                # (L, T, D)
        if self.speech_kind is None:
            return self._dummy
        if cid not in self._speech:
            arr = np.load(self.root / self.speech_kind / f"{cid}.npy")
            if arr.ndim == 2:                                  # mel: (T, 80) -> (1, T, 80)
                arr = arr[None]
            x = torch.tensor(arr)                              # own copy, not a view of the numpy buffer
            if self._stats:
                mean, std = self._stats[self._speaker[cid]]
                x = ((x.float() - mean[:, None]) / std[:, None]).half()
            self._speech[cid] = x
        return self._speech[cid]

    def text(self, text: str) -> torch.Tensor:                 # (L, tokens, D)
        if self.text_input == "zeros":
            return torch.zeros_like(self._text[text])
        if self.text_input == "shuffled":
            return self._text[self._shuffled[text]]
        return self._text[text]


def store_for(config: dict) -> FeatureStore:
    """The feature store a model with this config was trained on."""
    return FeatureStore(config.get("speech_features", "wavlm"), speaker_norm=config.get("speaker_norm", False),
                        text_input=config.get("text_input", "transcript"))


class EmotionDataset(Dataset):
    def __init__(self, rows: list[dict], store: FeatureStore):
        self.rows = rows
        self.store = store

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        cid = clip_id(r)
        return {"id": cid, "label": EMO2ID[r["emotion"]],
                "speech": self.store.speech(cid), "text": self.store.text(r["text"])}


def _pad(seqs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(L, T_i, D) tensors -> (B, L, T_max, D) plus a (B, T_max) mask of real steps."""
    t_max = max(s.shape[1] for s in seqs)
    out = torch.zeros(len(seqs), seqs[0].shape[0], t_max, seqs[0].shape[2], dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), t_max, dtype=torch.bool)
    for b, s in enumerate(seqs):
        out[b, :, :s.shape[1]] = s
        mask[b, :s.shape[1]] = True
    return out, mask


def collate(items: list[dict]) -> dict:
    speech, speech_mask = _pad([it["speech"] for it in items])
    text, text_mask = _pad([it["text"] for it in items])
    return {"id": [it["id"] for it in items], "label": torch.tensor([it["label"] for it in items]),
            "speech": speech, "speech_mask": speech_mask, "text": text, "text_mask": text_mask}


def make_loader(rows: list[dict], store: FeatureStore, batch_size: int, shuffle: bool,
                seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(EmotionDataset(rows, store), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=g, num_workers=0)
