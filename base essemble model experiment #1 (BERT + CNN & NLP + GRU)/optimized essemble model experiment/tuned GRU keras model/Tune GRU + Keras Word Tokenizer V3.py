# ===============================================================
# STREAM A — GRU + Keras Word Tokenizer
# V3: Random-search hyperparameter tuning
#
# Replaces the V2 "epoch sweep" (EPOCH_LIST = [5..10]) with a proper
# random search over the full hyperparameter space, using:
#   - validation macro-F1 for model selection (not final-epoch val loss)
#   - early stopping with best-weight restoration, so the number of
#     epochs is tuned automatically instead of by brute force
#   - a saved config manifest + saved tokenizer, so the ensemble script
#     can rebuild the exact architecture and vocabulary
#
# Outputs (all under OUT_DIR):
#   gru_best.pt                 - best model weights
#   gru_best_config.json        - hyperparameters + architecture + metrics
#   gru_tokenizer.json          - the fitted Keras tokenizer
#   GRU_Tuning_Trials.xlsx      - full trial table (for the thesis appendix)
#   gru_trials.jsonl            - incremental log (allows resuming)
# ===============================================================

import os
import json
import copy
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SEED = 42
N_TRIALS = 25            # random-search budget for this branch
MAX_EPOCHS = 25          # upper bound; early stopping usually stops sooner
PATIENCE = 4             # epochs without val macro-F1 improvement before stop
DATA_PATH = "redo/canonical_dataset.csv"
SPLITS_PATH = "redo/splits.npz"

OUT_DIR = (
    "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
    "optimized essemble model experiment/tuned GRU keras model/V3"
)
os.makedirs(OUT_DIR, exist_ok=True)

BEST_CKPT = os.path.join(OUT_DIR, "gru_best.pt")
BEST_CFG = os.path.join(OUT_DIR, "gru_best_config.json")
TOKENIZER_PATH = os.path.join(OUT_DIR, "gru_tokenizer.json")
TRIALS_JSONL = os.path.join(OUT_DIR, "gru_trials.jsonl")
TRIALS_XLSX = os.path.join(OUT_DIR, "GRU_Tuning_Trials.xlsx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


def set_seed(seed):
    """Re-seed everything so each trial is individually reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Data — identical canonical dataset and split as V2
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
LABEL_NAMES = list(le.classes_)

splits = np.load(SPLITS_PATH)
train_idx, val_idx = splits["train"], splits["val"]

X_train_text = [texts[i] for i in train_idx]
X_val_text = [texts[i] for i in val_idx]
y_train_np = y[train_idx]
y_val_np = y[val_idx]

print(f"Train: {len(X_train_text)} | Val: {len(X_val_text)} | Classes: {NUM_CLASSES}")


# ---------------------------------------------------------------
# Search space
#
# Every entry here is a hyperparameter that was HARD-CODED in V2.
# Reporting this table in the thesis is what makes the tuning claim
# concrete: V2 fixed these by assumption, V3 searches them.
# ---------------------------------------------------------------
SEARCH_SPACE = {
    "vocab_size":    [10000, 20000, 30000],
    "max_len":       [40, 50, 64, 80],
    "embed_dim":     [64, 128, 200, 256],
    "hidden_dim":    [64, 128, 192, 256],
    "num_layers":    [1, 2],
    "bidirectional": [False, True],
    "pooling":       ["last", "meanmax"],
    "dropout":       [0.0, 0.2, 0.3, 0.5],
    "batch_size":    [16, 32, 64],
    "weight_decay":  [0.0, 1e-5, 1e-4],
    "class_weight":  [None, "balanced"],
    "grad_clip":     [0.0, 1.0],
}
LR_RANGE = (1e-4, 5e-3)   # sampled log-uniformly


def sample_config(rng):
    # Index-based sampling rather than rng.choice(list): choice coerces mixed
    # lists (e.g. [None, "balanced"]) into object arrays and returns numpy
    # scalars, which are awkward to JSON-serialise. This keeps native types.
    cfg = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
    lo, hi = np.log10(LR_RANGE[0]), np.log10(LR_RANGE[1])
    cfg["lr"] = float(10 ** rng.uniform(lo, hi))
    # dropout between GRU layers is a no-op for a 1-layer GRU; PyTorch warns.
    if cfg["num_layers"] == 1:
        cfg["rnn_dropout"] = 0.0
    else:
        cfg["rnn_dropout"] = float(cfg["dropout"])
    return cfg


# ---------------------------------------------------------------
# Model — architecture is now driven by cfg rather than hard-coded.
#
# NOTE: V2's `self.fc(h.squeeze(0))` silently breaks for multi-layer or
# bidirectional GRUs (squeeze(0) only works when num_layers*dirs == 1).
# The pooling logic below handles all cases correctly.
# ---------------------------------------------------------------
class GRU_Model(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"], padding_idx=0)
        self.gru = nn.GRU(
            cfg["embed_dim"],
            cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
            batch_first=True,
            bidirectional=cfg["bidirectional"],
            dropout=cfg["rnn_dropout"],
        )
        dirs = 2 if cfg["bidirectional"] else 1
        feat = cfg["hidden_dim"] * dirs
        if cfg["pooling"] == "meanmax":
            feat *= 2
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()          # (B, L, 1)
        emb = self.embedding(x)
        out, h = self.gru(emb)                          # out: (B, L, H*dirs)

        if self.cfg["pooling"] == "last":
            dirs = 2 if self.cfg["bidirectional"] else 1
            # h: (num_layers*dirs, B, H) -> take the final layer, all directions
            h = h.view(self.cfg["num_layers"], dirs, x.size(0), self.cfg["hidden_dim"])
            feat = torch.cat([h[-1, d] for d in range(dirs)], dim=1)
        else:
            # Mask-aware mean pooling + max pooling over time.
            denom = mask.sum(dim=1).clamp(min=1.0)
            mean = (out * mask).sum(dim=1) / denom
            mx = out.masked_fill(mask == 0, float("-inf")).max(dim=1).values
            mx = torch.nan_to_num(mx, neginf=0.0)       # rows that are all padding
            feat = torch.cat([mean, mx], dim=1)

        return self.fc(self.dropout(feat))              # raw logits, NO softmax


# ---------------------------------------------------------------
# Per-trial data preparation
#
# vocab_size and max_len are themselves tuned, so the tokenizer has to be
# refit per trial. It is still fit ONLY on the training partition — no
# leakage from val/test.
# ---------------------------------------------------------------
def build_loaders(cfg):
    tok = Tokenizer(num_words=cfg["vocab_size"], oov_token="<OOV>")
    tok.fit_on_texts(X_train_text)

    def encode(texts_list):
        seqs = tok.texts_to_sequences(texts_list)
        return torch.tensor(
            pad_sequences(seqs, maxlen=cfg["max_len"]), dtype=torch.long
        )

    Xtr, Xva = encode(X_train_text), encode(X_val_text)
    ytr = torch.tensor(y_train_np, dtype=torch.long)
    yva = torch.tensor(y_val_np, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True
    )
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=256)
    return tok, train_loader, val_loader


def make_criterion(cfg):
    if cfg["class_weight"] == "balanced":
        w = compute_class_weight(
            "balanced", classes=np.arange(NUM_CLASSES), y=y_train_np
        )
        return nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device)
        )
    return nn.CrossEntropyLoss()


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, preds, trues = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_loss += criterion(logits, yb).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(yb.cpu().tolist())
    return (
        total_loss / max(len(loader), 1),
        accuracy_score(trues, preds),
        f1_score(trues, preds, average="macro", zero_division=0),
    )


def run_trial(cfg, trial_id):
    """Train one configuration to convergence, return metrics + best weights."""
    set_seed(SEED + trial_id)
    tok, train_loader, val_loader = build_loaders(cfg)
    model = GRU_Model(cfg, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    criterion = make_criterion(cfg)

    best_f1, best_state, best_epoch, best_val_loss, best_acc = -1.0, None, 0, None, None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1,
        })
        print(f"  epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val acc {val_acc:.4f} | val macroF1 {val_f1:.4f}")

        # ---- Early stopping on validation macro-F1 ----
        # Macro-F1 rather than loss because sentiment classes are imbalanced;
        # loss can improve while the minority class collapses.
        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch = val_f1, epoch
            best_val_loss, best_acc = val_loss, val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    return {
        "val_macro_f1": best_f1,
        "val_acc": best_acc,
        "val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "history": history,
    }, best_state, tok


# ---------------------------------------------------------------
# Random search
# ---------------------------------------------------------------
rng = np.random.default_rng(SEED)          # seeded -> the search is reproducible

completed = []
if os.path.exists(TRIALS_JSONL):
    with open(TRIALS_JSONL) as f:
        completed = [json.loads(line) for line in f if line.strip()]
    print(f"Resuming: {len(completed)} trial(s) already logged.")

# Regenerate the sampled configs deterministically so a resumed run explores
# exactly the same sequence it would have on an uninterrupted run.
all_configs = [sample_config(rng) for _ in range(N_TRIALS)]

best_overall = max(completed, key=lambda r: r["val_macro_f1"]) if completed else None

for i, cfg in enumerate(all_configs, start=1):
    if i <= len(completed):
        continue
    print(f"\n=== GRU trial {i}/{N_TRIALS} ===")
    print("  config:", {k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in cfg.items()})
    t0 = time.time()
    metrics, state, tok = run_trial(cfg, i)
    elapsed = time.time() - t0

    record = {"trial": i, **cfg, **{k: v for k, v in metrics.items() if k != "history"},
              "seconds": round(elapsed, 1)}
    with open(TRIALS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    completed.append(record)

    print(f"  -> val macro-F1 {metrics['val_macro_f1']:.4f} "
          f"(best epoch {metrics['best_epoch']}, {elapsed:.0f}s)")

    if best_overall is None or metrics["val_macro_f1"] > best_overall["val_macro_f1"]:
        best_overall = record
        torch.save(state, BEST_CKPT)
        with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
            f.write(tok.to_json())
        with open(BEST_CFG, "w") as f:
            json.dump({
                "branch": "GRU + Keras Word Tokenizer",
                "trial": i,
                "config": cfg,
                "num_classes": NUM_CLASSES,
                "label_names": LABEL_NAMES,
                "val_macro_f1": metrics["val_macro_f1"],
                "val_acc": metrics["val_acc"],
                "val_loss": metrics["val_loss"],
                "best_epoch": metrics["best_epoch"],
                "checkpoint": BEST_CKPT,
                "tokenizer": TOKENIZER_PATH,
                "history": metrics["history"],
                "search": {"n_trials": N_TRIALS, "method": "random search",
                           "seed": SEED, "space": SEARCH_SPACE, "lr_range": LR_RANGE},
            }, f, indent=2)
        print(f"  *** new best — checkpoint + config + tokenizer saved")

# ---------------------------------------------------------------
# Export the trial table
# ---------------------------------------------------------------
trials_df = pd.DataFrame(completed).sort_values("val_macro_f1", ascending=False)
with pd.ExcelWriter(TRIALS_XLSX, engine="openpyxl") as writer:
    trials_df.to_excel(writer, sheet_name="All Trials", index=False)
    trials_df.head(10).to_excel(writer, sheet_name="Top 10", index=False)

print("\n" + "=" * 60)
print(f"Best GRU val macro-F1: {best_overall['val_macro_f1']:.4f} (trial {best_overall['trial']})")
print(f"Checkpoint : {BEST_CKPT}")
print(f"Config     : {BEST_CFG}")
print(f"Trial table: {TRIALS_XLSX}")
