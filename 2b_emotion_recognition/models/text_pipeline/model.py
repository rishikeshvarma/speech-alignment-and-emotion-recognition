"""Text-only emotion model.

Block                Choice
-------------------  ----------------------------------------------------------
Preprocessing        transcript "Say the word <w>." -> BERT WordPiece tokens
                     (lower-cased, punctuation split; [CLS]/[SEP] added)
Feature extraction   frozen BERT-base-uncased, all 13 layers mixed with learned
                     weights -> (tokens x 768)
Contextual modelling linear projection -> 1-layer BiGRU -> attentive pooling (256-d)
Classifier           MLP 256 -> 128 -> 7

Note on TESS: each of the 200 target words is spoken in all 7 emotions by
both speakers, so a transcript carries no information about the label. This
pipeline is expected to sit at chance (1/7 = 14.3 %) on every split; it is
kept so the fusion model has a real text branch to learn to ignore.
"""
from __future__ import annotations

import torch.nn as nn

from models.common.data import EMOTIONS
from models.common.features import TEXT_DIMS
from models.common.layers import ClassifierHead, SequenceEncoder


class TextModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.encoder = SequenceEncoder(*TEXT_DIMS, gru_layers=1)
        self.head = ClassifierHead(self.encoder.out_dim, len(EMOTIONS))

    def forward(self, batch: dict) -> dict:
        enc = self.encoder(batch["text"], batch["text_mask"])
        return {"logits": self.head(enc["emb"]),
                "embeddings": {"contextual": enc["emb_clean"]},
                "layer_weights": {"text": enc["layer_weights"]}}


def build_model(cfg: dict) -> TextModel:
    return TextModel(cfg)
