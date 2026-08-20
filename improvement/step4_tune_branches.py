# ===============================================================
# step4_tune_branches.py
#
# Optuna hyperparameter search for the two branches.
#
# Split discipline (this is the part an examiner will probe):
#   TRAIN     - gradient updates only
#   VALIDATE  - epoch selection AND the Optuna objective
#   CALIBRATE - never touched here (reserved for fusion fitting)
#   TEST      - never touched here
#
# Because branch tuning consumes Validate and fusion fitting
# consumes Calibrate, the four-way split is doing exactly the job
# it was designed for. That is the argument for the design, and
# after this script it is demonstrated rather than asserted.
#
# Dependencies:
#   pip install optuna sentencepiece --break-system-packages
#
# ALWAYS smoke-test before committing GPU hours:
#   python improvement/step4_tune_branches.py --branch gru   --smoke
#   python improvement/step4_tune_branches.py --branch xlmr  --smoke
#
# Then the real runs:
#   python improvement/step4_tune_branches.py --branch gru   --trials 40
#   python improvement/step4_tune_branches.py --branch xlmr  --trials 12
# ===============================================================

import os
import sys
import gc
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_v2 import (  # noqa: E402
    ROOT, WORK_DIR, DEVICE, set_seed, load_canonical, class_weights,
    FocalLoss, macro_f1, save_json, save_pickle,
    GRUBranch, CNNTransformerBranch,
)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

TEXTS, Y, LABEL_NAMES, IDX = load_canonical()
NUM_CLASSES = len(LABEL_NAMES)
print(f"Classes: {LABEL_NAMES}   device: {DEVICE}")
print("Partition sizes:", {k: len(v) for k, v in IDX.items()})

Y_TRAIN, Y_VAL = Y[IDX["train"]], Y[IDX["val"]]
CW = class_weights(Y_TRAIN, NUM_CLASSES).to(DEVICE)
print("Class weights (inverse frequency):",
      dict(zip(LABEL_NAMES, np.round(CW.cpu().numpy(), 4))))


def texts_of(part):
    return [TEXTS[i] for i in IDX[part]]


# torch moved the AMP API in 2.4 (torch.cuda.amp -> torch.amp). These
# wrappers work on both so the script does not depend on your torch
# version.
_CUDA = DEVICE.type == "cuda"


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=_CUDA)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=_CUDA)


def autocast():
    try:
        return torch.amp.autocast("cuda", enabled=_CUDA)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=_CUDA)


def make_loss(name, gamma):
    if name == "focal":
        return FocalLoss(weight=CW, gamma=gamma)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=CW)
    return nn.CrossEntropyLoss()


# ===============================================================
# GRU branch
# ===============================================================
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_keras_tokenizer(vocab, max_len):
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok = Tokenizer(num_words=vocab, oov_token="<OOV>")
    tok.fit_on_texts(texts_of("train"))

    def encode(tl):
        return torch.tensor(
            pad_sequences(tok.texts_to_sequences(tl), maxlen=max_len),
            dtype=torch.long,
        )

    return tok, encode


def train_gru(p, epochs, verbose=False):
    set_seed(p["seed"])
    tok, encode = build_keras_tokenizer(p["vocab"], p["max_len"])
    Xtr, Xva = encode(texts_of("train")), encode(texts_of("val"))

    model = GRUBranch(
        vocab=p["vocab"], num_classes=NUM_CLASSES, embed_dim=p["embed_dim"],
        hidden=p["hidden"], layers=p["layers"],
        bidirectional=p["bidirectional"], dropout=p["dropout"],
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                            weight_decay=p["weight_decay"])
    lossf = make_loss(p["loss"], p.get("gamma", 2.0))
    tl = DataLoader(SeqDataset(Xtr, Y_TRAIN), batch_size=p["batch"], shuffle=True)
    vl = DataLoader(SeqDataset(Xva, Y_VAL), batch_size=256)

    best, best_state, best_epoch = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in vl:
                preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
        f1 = macro_f1(Y_VAL, torch.cat(preds).numpy())
        if f1 > best:
            best, best_epoch = f1, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    epoch {ep+1:2d}  val macro-F1 {f1:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best, best_state, best_epoch, tok


# ===============================================================
# XLM-R CNN branch
# ===============================================================
class TextDataset(Dataset):
    def __init__(self, texts, y):
        self.texts, self.y = texts, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.texts[i], int(self.y[i])


def collate_factory(tokenizer, max_len):
    def collate(batch):
        txt, lab = zip(*batch)
        # padding="max_length", NOT padding=True.
        #
        # This head max-pools over the conv output across ALL positions.
        # With dynamic padding the number of padded positions varies by
        # batch, so the same sentence would score differently depending
        # on what it was batched with - and training (dynamic) would
        # silently disagree with export (fixed). Fixed-length padding
        # makes training and inference identical, and matches what the
        # original CNN-BERT did.
        enc = tokenizer(list(txt), return_tensors="pt", padding="max_length",
                        truncation=True, max_length=max_len)
        return enc["input_ids"], enc["attention_mask"], torch.tensor(lab)
    return collate


def train_xlmr(p, epochs, verbose=False):
    from transformers import AutoTokenizer
    set_seed(p["seed"])
    tok = AutoTokenizer.from_pretrained(p["encoder"])
    coll = collate_factory(tok, p["max_len"])

    model = CNNTransformerBranch(
        encoder_name=p["encoder"], num_classes=NUM_CLASSES,
        n_filters=p["n_filters"], kernel_sizes=tuple(p["kernel_sizes"]),
        dropout=p["dropout"], freeze_embeddings=p["freeze_embeddings"],
        unfrozen_layers=p["unfrozen_layers"],
    ).to(DEVICE)

    params = [q for q in model.parameters() if q.requires_grad]
    n_train = sum(q.numel() for q in params)
    if verbose:
        print(f"    trainable params: {n_train/1e6:.1f}M")

    opt = torch.optim.AdamW(params, lr=p["lr"], weight_decay=p["weight_decay"])
    lossf = make_loss(p["loss"], p.get("gamma", 2.0))
    scaler = make_scaler()

    tl = DataLoader(TextDataset(texts_of("train"), Y_TRAIN),
                    batch_size=p["batch"], shuffle=True, collate_fn=coll)
    vl = DataLoader(TextDataset(texts_of("val"), Y_VAL),
                    batch_size=64, collate_fn=coll)

    total = max(1, len(tl) * epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=p["lr"], total_steps=total, pct_start=0.1)

    best, best_state, best_epoch = -1.0, None, -1
    for ep in range(epochs):
        model.train()
        for ids, mask, yb in tl:
            ids, mask, yb = ids.to(DEVICE), mask.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            with autocast():
                loss = lossf(model(ids, mask), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

        model.eval()
        preds = []
        with torch.no_grad():
            for ids, mask, _ in vl:
                with autocast():
                    preds.append(model(ids.to(DEVICE), mask.to(DEVICE)).argmax(1).cpu())
        f1 = macro_f1(Y_VAL, torch.cat(preds).numpy())
        if f1 > best:
            best, best_epoch = f1, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"    epoch {ep+1:2d}  val macro-F1 {f1:.4f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best, best_state, best_epoch, tok


# ===============================================================
# Search spaces
# ===============================================================
def suggest_gru(t, seed):
    return {
        "seed": seed,
        "vocab": t.suggest_categorical("vocab", [10000, 20000, 30000]),
        "max_len": t.suggest_categorical("max_len", [50, 80, 120]),
        "embed_dim": t.suggest_categorical("embed_dim", [128, 200, 300]),
        "hidden": t.suggest_categorical("hidden", [128, 192, 256]),
        "layers": t.suggest_int("layers", 1, 2),
        "bidirectional": t.suggest_categorical("bidirectional", [True, False]),
        "dropout": t.suggest_float("dropout", 0.0, 0.5),
        "lr": t.suggest_float("lr", 3e-4, 5e-3, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch": t.suggest_categorical("batch", [32, 64]),
        "loss": t.suggest_categorical("loss", ["weighted_ce", "focal", "ce"]),
        "gamma": t.suggest_float("gamma", 1.0, 3.0),
    }


def suggest_xlmr(t, seed, encoder):
    ks = t.suggest_categorical("kernel_sizes", ["3", "3,5", "2,3,4"])
    return {
        "seed": seed,
        "encoder": encoder,
        "max_len": t.suggest_categorical("max_len", [64, 96]),
        "n_filters": t.suggest_categorical("n_filters", [128, 256]),
        "kernel_sizes": [int(x) for x in ks.split(",")],
        "dropout": t.suggest_float("dropout", 0.0, 0.4),
        "lr": t.suggest_float("lr", 5e-6, 5e-5, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch": t.suggest_categorical("batch", [16, 32]),
        "freeze_embeddings": True,
        "unfrozen_layers": t.suggest_categorical("unfrozen_layers", [4, 8, 12]),
        "loss": t.suggest_categorical("loss", ["weighted_ce", "focal"]),
        "gamma": t.suggest_float("gamma", 1.0, 3.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=["gru", "xlmr"], required=True)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default="xlm-roberta-base")
    ap.add_argument("--smoke", action="store_true",
                    help="1 trial, 1 epoch - verifies the pipeline end to end")
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    epochs = args.epochs or (1 if args.smoke else (12 if args.branch == "gru" else 4))
    trials = 1 if args.smoke else args.trials
    tag = args.branch

    print(f"\n{'='*66}\nTUNING {tag.upper()}   trials={trials}  epochs/trial={epochs}")
    if args.branch == "xlmr":
        print(f"encoder: {args.encoder}")
    print("=" * 66)

    best_holder = {"f1": -1.0, "state": None, "params": None,
                   "epoch": None, "tok": None}

    def objective(trial):
        p = (suggest_gru(trial, args.seed) if args.branch == "gru"
             else suggest_xlmr(trial, args.seed, args.encoder))
        try:
            fn = train_gru if args.branch == "gru" else train_xlmr
            f1, state, ep, tok = fn(p, epochs, verbose=args.smoke)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print("    OOM - pruning this trial (reduce batch or max_len)")
            raise optuna.TrialPruned()

        print(f"  trial {trial.number:3d}  val macro-F1 {f1:.4f}  (best epoch {ep})")
        if f1 > best_holder["f1"]:
            best_holder.update({"f1": f1, "state": state, "params": p,
                                "epoch": ep, "tok": tok})
            print(f"    ^ new best")
        return f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=trials)

    out = os.path.join(WORK_DIR, tag)
    os.makedirs(out, exist_ok=True)

    torch.save(best_holder["state"], os.path.join(out, f"{tag}_best.pt"))
    save_json({
        "branch": tag,
        "best_val_macro_f1": best_holder["f1"],
        "best_epoch": best_holder["epoch"],
        "params": best_holder["params"],
        "n_trials": trials,
        "epochs_per_trial": epochs,
        "label_names": LABEL_NAMES,
        "smoke": bool(args.smoke),
    }, os.path.join(out, f"{tag}_best_params.json"))

    if args.branch == "gru":
        # Persist the fitted Keras tokenizer. The old pipeline refit it
        # from scratch in every downstream script and relied on that
        # refit being identical - a silent-corruption risk. Saving it
        # removes the possibility entirely.
        save_pickle(best_holder["tok"], os.path.join(out, "keras_tokenizer.pkl"))

    print(f"\nBest val macro-F1: {best_holder['f1']:.4f}")
    print(f"Params: {json.dumps(best_holder['params'], default=str)}")
    print(f"Saved to: {out}")
    if args.smoke:
        print("\nSMOKE TEST ONLY - results meaningless. Re-run without --smoke.")


if __name__ == "__main__":
    main()
