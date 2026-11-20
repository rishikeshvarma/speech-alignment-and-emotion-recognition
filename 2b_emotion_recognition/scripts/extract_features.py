"""Preprocess every clip once and cache the frozen features the pipelines train on.

Speech (one file per clip in <work>/features/):
  1. load, mix to mono, resample to 16 kHz (TESS is 24.4 kHz, one clip 96 kHz;
     WavLM was trained on 16 kHz audio)
  2. trim leading/trailing silence (TESS clips have up to ~0.5 s of it)
  3. scale to a common RMS level so loudness differences between recording
     sessions do not leak into the features
  4a. wavlm/<id>.npy   all 13 hidden layers of WavLM-Base+  (13, T, 768) fp16, 50 frames/s
  4b. mel/<id>.npy     80-band log-mel spectrogram          (T, 80) fp16, 100 frames/s
      (a hand-crafted baseline for the speech pipeline)

Text (<work>/features/bert_text.npz):
  "Say the word <w>." -> all 13 hidden layers of BERT-base-uncased per token
  (13, L, 768) fp16 with L including [CLS]/[SEP].

The encoders are frozen, so caching once makes every training run take
seconds instead of minutes.

Usage:  python scripts/extract_features.py [--only speech|text] [--device cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.common.data import TESS_DIR, WORK_DIR, load_metadata  # noqa: E402

SR = 16000
TOP_DB = 30            # trim threshold relative to the clip's peak
TARGET_RMS = 0.05
SPEECH_MODEL = "microsoft/wavlm-base-plus"
TEXT_MODEL = "bert-base-uncased"


def clip_id(path: str) -> str:
    return Path(path).stem


def preprocess(path: Path) -> tuple[np.ndarray, dict]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    raw_len = len(audio)
    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB, frame_length=400, hop_length=160)
    rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    audio = np.clip(audio * (TARGET_RMS / rms), -1.0, 1.0)
    return audio, {"orig_sr": sr, "raw_s": raw_len / SR, "trimmed_s": len(audio) / SR}


def log_mel(audio: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=400, hop_length=160, win_length=400,
                                       n_mels=80, fmin=20, fmax=8000)
    return np.log(m + 1e-6).T.astype(np.float16)            # (T, 80)


@torch.no_grad()
def extract_speech(rows, out: Path, device: str) -> None:
    from transformers import AutoFeatureExtractor, WavLMModel

    fe = AutoFeatureExtractor.from_pretrained(SPEECH_MODEL)
    model = WavLMModel.from_pretrained(SPEECH_MODEL).to(device).eval()
    (out / "wavlm").mkdir(parents=True, exist_ok=True)
    (out / "mel").mkdir(parents=True, exist_ok=True)
    stats = []
    t0 = time.time()
    for k, r in enumerate(rows):
        cid = clip_id(r["path"])
        audio, info = preprocess(TESS_DIR / r["path"])
        stats.append(info)
        np.save(out / "mel" / f"{cid}.npy", log_mel(audio))
        inputs = fe(audio, sampling_rate=SR, return_tensors="pt")
        hs = model(inputs["input_values"].to(device), output_hidden_states=True).hidden_states
        layers = torch.stack(hs, 0)[:, 0]                    # (13, T, 768)
        np.save(out / "wavlm" / f"{cid}.npy", layers.to(torch.float16).cpu().numpy())
        if (k + 1) % 200 == 0:
            print(f"  speech {k + 1}/{len(rows)}  ({time.time() - t0:.0f} s)", flush=True)
    raw = np.array([s["raw_s"] for s in stats])
    trim = np.array([s["trimmed_s"] for s in stats])
    print(f"speech done: duration raw {raw.mean():.2f}s (min {raw.min():.2f}, max {raw.max():.2f}) -> "
          f"trimmed {trim.mean():.2f}s (min {trim.min():.2f}, max {trim.max():.2f})")
    np.savez(out / "speech_stats.npz", raw=raw, trimmed=trim,
             ids=np.array([clip_id(r["path"]) for r in rows]))


@torch.no_grad()
def extract_text(rows, out: Path, device: str) -> None:
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
    texts = sorted({r["text"] for r in rows})
    feats, tokens = {}, {}
    for text in texts:
        # preprocessing: BERT's uncased tokenizer lower-cases and splits
        # punctuation; nothing else is needed for these short prompts
        enc = tok(text, return_tensors="pt").to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        feats[text] = torch.stack(hs, 0)[:, 0].to(torch.float16).cpu().numpy()   # (13, L, 768)
        tokens[text] = tok.convert_ids_to_tokens(enc["input_ids"][0])
    # one padded array (n_texts, 13, L_max, 768) + true lengths
    lengths = np.array([feats[t].shape[1] for t in texts])
    padded = np.zeros((len(texts), feats[texts[0]].shape[0], lengths.max(), feats[texts[0]].shape[2]),
                      dtype=np.float16)
    for i, t in enumerate(texts):
        padded[i, :, :lengths[i]] = feats[t]
    np.savez(out / "bert_text.npz", texts=np.array(texts), feats=padded, lengths=lengths,
             tokens=np.array([" ".join(tokens[t]) for t in texts]))
    print(f"text done: {len(texts)} distinct texts, e.g. {texts[0]!r} -> {tokens[texts[0]]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["speech", "text"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"HF cache: {os.environ.get('HF_HOME', '(default ~/.cache/huggingface)')}  device: {args.device}")
    rows = load_metadata()
    out = WORK_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "text"):
        extract_text(rows, out, args.device)
    if args.only in (None, "speech"):
        extract_speech(rows, out, args.device)


if __name__ == "__main__":
    main()
