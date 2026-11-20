"""Speech + text (multimodal) emotion model.

Both branches are the same blocks as the unimodal pipelines, trained jointly
from scratch with the fusion layer and classifier:

  speech: WavLM layer mix -> BiGRU x2 -> attentive pooling -> s (256-d)   [temporal modelling]
  text:   BERT layer mix  -> BiGRU x1 -> attentive pooling -> t (256-d)   [contextual modelling]
  fusion: Gated Multimodal Unit  h = z*tanh(Ws s) + (1-z)*tanh(Wt t),  z = sigmoid(Wz[s;t])
          [baseline: concatenation + one dense layer]
  classifier: MLP 256 -> 128 -> 7

Fusion happens at the utterance level (after pooling) because the two
sequences have different, unaligned time axes (50 speech frames/s vs. 5-7
word pieces), and the gate gives a per-example, readable measure of how much
the model relies on each modality.
"""
from __future__ import annotations

import torch.nn as nn

from models.common.data import EMOTIONS
from models.common.features import SPEECH_DIMS, TEXT_DIMS
from models.common.layers import ClassifierHead, ConcatFusion, GatedFusion, SequenceEncoder


class FusionModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.speech = SequenceEncoder(*SPEECH_DIMS[cfg["speech_features"]], gru_layers=2, time_mask=0.1)
        self.text = SequenceEncoder(*TEXT_DIMS, gru_layers=1)
        fusion_cls = {"gmu": GatedFusion, "concat": ConcatFusion}[cfg["fusion"]]
        self.fusion = fusion_cls(self.speech.out_dim, self.text.out_dim)
        self.drop = nn.Dropout(0.3)
        self.head = ClassifierHead(self.fusion.out_dim, len(EMOTIONS))

    def forward(self, batch: dict) -> dict:
        s = self.speech(batch["speech"], batch["speech_mask"])
        t = self.text(batch["text"], batch["text_mask"])
        f = self.fusion(s["emb"], t["emb"])
        return {"logits": self.head(self.drop(f["fused"])),
                "embeddings": {"temporal": s["emb_clean"], "contextual": t["emb_clean"], "fusion": f["fused"]},
                "gate": f["gate"],
                "layer_weights": {"speech": s["layer_weights"], "text": t["layer_weights"]}}


def build_model(cfg: dict) -> FusionModel:
    return FusionModel(cfg)
