"""Speech-only emotion model.

Block               Choice
------------------  -----------------------------------------------------------
Preprocessing       16 kHz mono, silence trimmed, RMS-normalised
                    (scripts/extract_features.py)
Feature extraction  frozen WavLM-Base+, all 13 layers mixed with learned weights
                    -> (time steps x 768) at 50 frames/s
                    [baseline: 80-band log-mel, (time steps x 80) at 100 frames/s]
Temporal modelling  linear projection -> 2-layer BiGRU -> attentive pooling (256-d)
Classifier          MLP 256 -> 128 -> 7
"""
from __future__ import annotations

import torch.nn as nn

from models.common.data import EMOTIONS
from models.common.features import SPEECH_DIMS
from models.common.layers import ClassifierHead, SequenceEncoder


class SpeechModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        n_layers, dim = SPEECH_DIMS[cfg["speech_features"]]
        self.encoder = SequenceEncoder(n_layers, dim, gru_layers=2, time_mask=0.1)
        self.head = ClassifierHead(self.encoder.out_dim, len(EMOTIONS))

    def forward(self, batch: dict) -> dict:
        enc = self.encoder(batch["speech"], batch["speech_mask"])
        return {"logits": self.head(enc["emb"]),
                "embeddings": {"temporal": enc["emb_clean"]},
                "layer_weights": {"speech": enc["layer_weights"]}}


def build_model(cfg: dict) -> SpeechModel:
    return SpeechModel(cfg)
