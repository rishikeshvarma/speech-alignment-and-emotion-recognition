"""Building blocks used by the speech, text and fusion pipelines.

Shapes: B = batch, L = encoder layers, T = time steps / tokens, D = feature size.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LayerWeightedSum(nn.Module):
    """Mix the hidden layers of a frozen encoder with learned softmax weights.

    Different layers of WavLM / BERT hold different information (acoustic
    detail low down, phonetic and lexical content higher up), and which ones
    matter for emotion is not known in advance, so the model learns the mix
    (as in the SUPERB benchmark). The learned weights are plotted later.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, L, T, D) -> (B, T, D)
        return torch.einsum("l,bltd->btd", self.weights, x)


class AttentivePooling(nn.Module):
    """Weighted average over time; the weights come from a small scoring MLP.

    Emotion is often carried by a few frames (a stressed vowel, a burst of
    energy), so a learned weighting beats a plain mean. Padding is masked out.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf"))   # (B, T)
        a = torch.softmax(s, dim=1)
        return torch.einsum("bt,btd->bd", a, h), a


class SequenceEncoder(nn.Module):
    """Frozen-encoder features -> (layer mix) -> projection -> BiGRU -> attentive pooling.

    Used as the speech "temporal modelling" block (over 20 ms WavLM frames)
    and as the text "contextual modelling" block (over BERT word pieces).
    A BiGRU sees the whole sequence in both directions, which suits short
    clips (1-3 s / 5-7 tokens) and trains well on ~2k examples, where a deeper
    Transformer would overfit.
    """

    def __init__(self, n_layers: int, in_dim: int, proj: int = 256, hidden: int = 128,
                 gru_layers: int = 2, dropout: float = 0.3, time_mask: float = 0.0):
        super().__init__()
        self.mix = LayerWeightedSum(n_layers)
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, proj), nn.GELU(), nn.Dropout(dropout))
        # Single-layer GRUs stacked by hand, with dropout in between. This is
        # what nn.GRU(num_layers=2, dropout=p) computes, but the built-in
        # version makes this PyTorch 2.11 / CUDA 12.8 build crash at process
        # exit (0xC0000409, cuDNN RNN dropout state).
        self.rnns = nn.ModuleList(
            nn.GRU(proj if k == 0 else 2 * hidden, hidden, batch_first=True, bidirectional=True)
            for k in range(gru_layers))
        self.pool = AttentivePooling(2 * hidden)
        self.drop = nn.Dropout(dropout)
        self.time_mask = time_mask
        self.out_dim = 2 * hidden

    def _mask_time(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """SpecAugment-style: zero two random spans (each up to time_mask of the clip)."""
        if not self.training or self.time_mask <= 0:
            return x
        x = x.clone()
        for b, n in enumerate(lengths.tolist()):
            for _ in range(2):
                w = int(torch.randint(0, max(1, int(n * self.time_mask)) + 1, (1,)))
                if w == 0 or n - w <= 0:
                    continue
                s = int(torch.randint(0, n - w, (1,)))
                x[b, s:s + w] = 0
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        lengths = mask.sum(1)
        h = self.norm(self.mix(x.float()))
        h = self._mask_time(h, lengths)
        out = self.proj(h)
        for k, rnn in enumerate(self.rnns):
            if k > 0:
                out = self.drop(out)
            packed = nn.utils.rnn.pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = rnn(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=mask.size(1))
        emb, attn = self.pool(out, mask)
        return {"emb": self.drop(emb), "emb_clean": emb, "attn": attn, "layer_weights": self.mix.weights}


class ClassifierHead(nn.Module):
    """Two-layer MLP from the utterance representation to the 7 emotion logits."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al., 2017).

        h_s = tanh(W_s s)     h_t = tanh(W_t t)
        z   = sigmoid(W_z [s; t])          (one gate value per dimension)
        h   = z * h_s + (1 - z) * h_t

    The gate decides, per example and per dimension, how much to trust each
    modality, so an uninformative modality can be switched off. Its mean
    value is a readable "speech reliance" score.
    """

    def __init__(self, speech_dim: int, text_dim: int, dim: int = 256):
        super().__init__()
        self.s = nn.Linear(speech_dim, dim)
        self.t = nn.Linear(text_dim, dim)
        self.z = nn.Linear(speech_dim + text_dim, dim)
        self.out_dim = dim

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> dict:
        z = torch.sigmoid(self.z(torch.cat([s, t], -1)))
        h = z * torch.tanh(self.s(s)) + (1 - z) * torch.tanh(self.t(t))
        return {"fused": h, "gate": z}


class ConcatFusion(nn.Module):
    """Baseline: concatenate both vectors and mix them with one layer."""

    def __init__(self, speech_dim: int, text_dim: int, dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(speech_dim + text_dim, dim), nn.GELU())
        self.out_dim = dim

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> dict:
        return {"fused": self.net(torch.cat([s, t], -1)), "gate": None}
