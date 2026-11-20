"""Training and testing loops shared by the three pipelines.

Each pipeline's train.py / test.py only supplies a function that builds its
model from a config dict, plus its own command-line options; everything else
(splits, seeds, early stopping, checkpoints, metrics files) lives here so the
three pipelines are trained and scored in exactly the same way.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from .data import CHECKPOINTS, EMOTIONS, PROTOCOLS, RESULTS, WORK_DIR, load_metadata, split_rows
from .features import make_loader, store_for

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRICS = RESULTS / "metrics.csv"
EMBEDDINGS = WORK_DIR / "embeddings"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=10, help="early stopping on validation accuracy")


def _to(batch: dict) -> dict:
    return {k: v.to(DEVICE, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def run_epoch(model, loader, opt=None) -> tuple[float, float]:
    train = opt is not None
    model.train(train)
    n, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        batch = _to(batch)
        with torch.set_grad_enabled(train):
            logits = model(batch)["logits"]
            loss = F.cross_entropy(logits, batch["label"], label_smoothing=0.05 if train else 0.0)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        n += len(batch["label"])
        correct += (logits.argmax(1) == batch["label"]).sum().item()
        loss_sum += loss.item() * len(batch["label"])
    return loss_sum / n, correct / n


def ckpt_path(variant: str, protocol: str, seed: int) -> Path:
    return CHECKPOINTS / f"{variant}_{protocol}_seed{seed}.pt"


def train_main(variant_of: Callable, build_model: Callable, config_of: Callable,
               add_args: Callable | None = None, description: str = "") -> None:
    ap = argparse.ArgumentParser(description=description)
    common_args(ap)
    if add_args:
        add_args(ap)
    args = ap.parse_args()
    config = config_of(args)
    variant = variant_of(args)
    store = store_for(config)
    rows = load_metadata()
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "training_logs").mkdir(parents=True, exist_ok=True)

    for protocol in args.protocols:
        train_rows, val_rows = split_rows(rows, protocol, "train"), split_rows(rows, protocol, "val")
        for seed in args.seeds:
            set_seed(seed)
            model = build_model(config).to(DEVICE)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            tr = make_loader(train_rows, store, args.batch_size, shuffle=True, seed=seed)
            va = make_loader(val_rows, store, 64, shuffle=False)
            best, best_loss, best_epoch, waited, history = -1.0, float("inf"), 0, 0, []
            t0 = time.time()
            for epoch in range(1, args.epochs + 1):
                tl, ta = run_epoch(model, tr, opt)
                vl, vacc = run_epoch(model, va)
                sched.step()
                history.append({"epoch": epoch, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": vacc})
                # select on validation accuracy; on ties (common once speech
                # models reach 100 % on TESS) prefer the lower validation loss
                if vacc > best or (vacc == best and vl < best_loss):
                    best, best_loss, best_epoch, waited = vacc, vl, epoch, 0
                    torch.save({"state_dict": model.state_dict(), "config": config, "variant": variant,
                                "protocol": protocol, "seed": seed, "best_epoch": epoch, "val_acc": vacc},
                               ckpt_path(variant, protocol, seed))
                else:
                    waited += 1
                    if waited >= args.patience:
                        break
            with open(RESULTS / "training_logs" / f"{variant}_{protocol}_seed{seed}.json", "w") as fh:
                json.dump({"variant": variant, "protocol": protocol, "seed": seed, "best_epoch": best_epoch,
                           "best_val_acc": best, "n_params_trainable": sum(p.numel() for p in params),
                           "history": history}, fh, indent=1)
            print(f"{variant:24s} {protocol:8s} seed {seed}: best val acc {best:.3f} at epoch {best_epoch:2d} "
                  f"({time.time() - t0:.0f} s, {sum(p.numel() for p in params)/1e6:.2f} M params)", flush=True)


@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    out = {"id": [], "y_true": [], "y_pred": [], "probs": [], "emb": {}, "gate": [], "layer_weights": {}}
    for batch in loader:
        batch = _to(batch)
        res = model(batch)
        probs = torch.softmax(res["logits"], 1)
        out["id"] += batch["id"]
        out["y_true"] += batch["label"].tolist()
        out["y_pred"] += probs.argmax(1).tolist()
        out["probs"].append(probs.cpu().numpy())
        for k, v in res.get("embeddings", {}).items():
            out["emb"].setdefault(k, []).append(v.cpu().numpy())
        if res.get("gate") is not None:
            out["gate"].append(res["gate"].mean(1).cpu().numpy())
        for k, v in res.get("layer_weights", {}).items():
            out["layer_weights"][k] = v.cpu().numpy().tolist()
    out["probs"] = np.concatenate(out["probs"])
    out["emb"] = {k: np.concatenate(v) for k, v in out["emb"].items()}
    out["gate"] = np.concatenate(out["gate"]) if out["gate"] else None
    return out


def _update_metrics(row: dict) -> None:
    rows = []
    if METRICS.exists():
        with open(METRICS, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh)
                    if (r["variant"], r["protocol"], r["seed"]) != (row["variant"], row["protocol"], str(row["seed"]))]
    rows.append(row)
    rows.sort(key=lambda r: (r["variant"], r["protocol"], int(r["seed"])))
    with open(METRICS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerows(rows)


def test_main(variant_of: Callable, build_model: Callable, add_args: Callable | None = None,
              description: str = "") -> None:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    if add_args:
        add_args(ap)
    args = ap.parse_args()
    variant = variant_of(args)
    rows = load_metadata()
    by_id = {Path(r["path"]).stem: r for r in rows}
    store = None
    (RESULTS / "predictions").mkdir(parents=True, exist_ok=True)
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)

    summary = []
    for protocol in args.protocols:
        test_rows = split_rows(rows, protocol, "test")
        for seed in args.seeds:
            path = ckpt_path(variant, protocol, seed)
            if not path.exists():
                print(f"missing checkpoint {path.name} - run train.py first")
                continue
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            if store is None:
                store = store_for(ckpt["config"])
            model = build_model(ckpt["config"]).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            res = evaluate(model, make_loader(test_rows, store, 64, shuffle=False))
            y, p = np.array(res["y_true"]), np.array(res["y_pred"])
            acc = float((y == p).mean())
            f1 = float(f1_score(y, p, average="macro"))
            per_class = {f"acc_{e}": round(float((p[y == k] == k).mean()), 4) for k, e in enumerate(EMOTIONS)}
            _update_metrics({"variant": variant, "protocol": protocol, "seed": seed,
                             "best_epoch": ckpt["best_epoch"], "val_acc": round(ckpt["val_acc"], 4),
                             "test_acc": round(acc, 4), "test_macro_f1": round(f1, 4), **per_class})
            with open(RESULTS / "predictions" / f"{variant}_{protocol}_seed{seed}.csv", "w",
                      newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "speaker", "word", "true", "pred", "confidence"]
                           + [f"p_{e}" for e in EMOTIONS] + (["speech_gate"] if res["gate"] is not None else []))
                for i, cid in enumerate(res["id"]):
                    r = by_id[cid]
                    w.writerow([cid, r["speaker"], r["word"], EMOTIONS[res["y_true"][i]], EMOTIONS[res["y_pred"][i]],
                                round(float(res["probs"][i].max()), 4)]
                               + [round(float(x), 4) for x in res["probs"][i]]
                               + ([round(float(res["gate"][i]), 4)] if res["gate"] is not None else []))
            np.savez(EMBEDDINGS / f"{variant}_{protocol}_seed{seed}.npz", ids=np.array(res["id"]),
                     y_true=y, y_pred=p, **{f"emb_{k}": v for k, v in res["emb"].items()},
                     layer_weights=json.dumps(res["layer_weights"]))
            summary.append((protocol, seed, acc, f1))
            print(f"{variant:24s} {protocol:8s} seed {seed}: test acc {acc:.3f}  macro-F1 {f1:.3f}")

    for protocol in args.protocols:
        accs = [a for pr, _, a, _ in summary if pr == protocol]
        if accs:
            print(f"  {variant} / {protocol}: test acc {np.mean(accs):.3f} +- {np.std(accs):.3f} over {len(accs)} seeds")
