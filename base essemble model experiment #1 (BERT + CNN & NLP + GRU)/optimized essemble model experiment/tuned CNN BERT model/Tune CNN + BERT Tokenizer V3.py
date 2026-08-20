# ===============================================================
# STREAM B — CNN + BERT Tokenizer
# V3: Random-search hyperparameter tuning
#
# Changes vs V2 (which only swept EPOCH_LIST = [5..10] with everything
# else fixed at lr=2e-5, one 3-wide kernel, 128 filters, max_len=50):
#   - discriminative learning rates (BERT encoder vs CNN head) — the single
#     shared lr in V2 is either too high for BERT or too low for the head
#   - multi-width kernels, tunable filter count and dropout
#   - optional partial freezing of the lower BERT layers (cheaper + often
#     better on small datasets)
#   - linear warmup/decay schedule
#   - mixed precision + pre-tokenisation for speed (V2 re-tokenised every
#     sample on every epoch inside __getitem__, which is a large waste)
#   - early stopping on validation macro-F1 with best-weight restoration
#
# Outputs (all under OUT_DIR):
#   cnn_bert_best.pt / cnn_bert_best_config.json / CNN_BERT_Tuning_Trials.xlsx
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

from transformers import BertTokenizerFast, BertModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
SEED = 42
N_TRIALS = 15            # BERT fine-tuning is the expensive branch
MAX_EPOCHS = 8
PATIENCE = 2             # BERT converges fast and overfits fast
BERT_NAME = "bert-base-cased"
DATA_PATH = "redo/canonical_dataset.csv"
SPLITS_PATH = "redo/splits.npz"

OUT_DIR = (
    "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
    "optimized essemble model experiment/tuned CNN BERT model/V3"
)
os.makedirs(OUT_DIR, exist_ok=True)

BEST_CKPT = os.path.join(OUT_DIR, "cnn_bert_best.pt")
BEST_CFG = os.path.join(OUT_DIR, "cnn_bert_best_config.json")
TRIALS_JSONL = os.path.join(OUT_DIR, "cnn_bert_trials.jsonl")
TRIALS_XLSX = os.path.join(OUT_DIR, "CNN_BERT_Tuning_Trials.xlsx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = device.type == "cuda"
print("Using:", device, "| AMP:", USE_AMP)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Data
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

tokenizer = BertTokenizerFast.from_pretrained(BERT_NAME)

# Tokenise once per distinct max_len and reuse across trials. V2 tokenised
# inside Dataset.__getitem__, i.e. once per sample per epoch per trial.
_ENC_CACHE = {}


def encode(text_list, max_len, key):
    ck = (key, max_len)
    if ck not in _ENC_CACHE:
        enc = tokenizer(
            text_list, padding="max_length", truncation=True,
            max_length=max_len, return_tensors="pt",
        )
        _ENC_CACHE[ck] = (enc["input_ids"], enc["attention_mask"])
    return _ENC_CACHE[ck]


# ---------------------------------------------------------------
# Search space
# ---------------------------------------------------------------
SEARCH_SPACE = {
    "max_len":        [50, 64, 96],
    "batch_size":     [8, 16, 32],
    "num_filters":    [64, 128, 256],
    "kernel_sizes":   [[3], [2, 3, 4], [3, 4, 5], [2, 3, 4, 5]],
    "dropout":        [0.1, 0.2, 0.3, 0.5],
    "freeze_layers":  [0, 4, 8],      # 0 = fully fine-tune all 12 layers
    "weight_decay":   [0.0, 0.01],
    "warmup_ratio":   [0.0, 0.1],
    "class_weight":   [None, "balanced"],
    "grad_clip":      [0.0, 1.0],
}
LR_BERT_RANGE = (5e-6, 5e-5)     # log-uniform, the usual fine-tuning band
LR_HEAD_RANGE = (1e-4, 3e-3)     # the randomly-initialised CNN head can go faster


def _loguniform(rng, lo, hi):
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def sample_config(rng):
    cfg = {k: v[int(rng.integers(len(v)))] for k, v in SEARCH_SPACE.items()}
    cfg["lr_bert"] = _loguniform(rng, *LR_BERT_RANGE)
    cfg["lr_head"] = _loguniform(rng, *LR_HEAD_RANGE)
    return cfg


# ---------------------------------------------------------------
# Model
# ---------------------------------------------------------------
class CNN_BERT(nn.Module):
    """Config-driven version of the V2 model.

    Two substantive fixes beyond tunability:
      1. padded positions are zeroed before the convolution, so the filters
         no longer slide over [PAD] representations;
      2. multiple kernel widths are concatenated, giving the head access to
         several n-gram scales instead of only trigrams.
    """

    def __init__(self, cfg, num_classes, bert_name=BERT_NAME):
        super().__init__()
        self.cfg = cfg
        self.bert = BertModel.from_pretrained(bert_name)
        hidden = self.bert.config.hidden_size
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden, cfg["num_filters"], kernel_size=k, padding=k // 2)
            for k in cfg["kernel_sizes"]
        ])
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["num_filters"] * len(cfg["kernel_sizes"]), num_classes)

    def freeze(self):
        n = self.cfg["freeze_layers"]
        if n <= 0:
            return
        for p in self.bert.embeddings.parameters():
            p.requires_grad = False
        for layer in self.bert.encoder.layer[:n]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state   # (B, L, H)
        x = x * mask.unsqueeze(-1).to(x.dtype)                      # kill padding
        x = x.permute(0, 2, 1)                                      # (B, H, L)
        feats = [self.pool(torch.relu(conv(x))).squeeze(2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))       # raw logits


# ---------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------
def build_loaders(cfg):
    tr_ids, tr_mask = encode(X_train_text, cfg["max_len"], "train")
    va_ids, va_mask = encode(X_val_text, cfg["max_len"], "val")
    train_ds = TensorDataset(tr_ids, tr_mask, torch.tensor(y_train_np, dtype=torch.long))
    val_ds = TensorDataset(va_ids, va_mask, torch.tensor(y_val_np, dtype=torch.long))
    return (
        DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True),
        DataLoader(val_ds, batch_size=64),
    )


def make_criterion(cfg):
    if cfg["class_weight"] == "balanced":
        w = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train_np)
        return nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device)
        )
    return nn.CrossEntropyLoss()


def make_optimizer(model, cfg):
    """Discriminative learning rates: slow for pretrained BERT, fast for the head."""
    bert_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bert_params if name.startswith("bert.") else head_params).append(p)
    return torch.optim.AdamW(
        [
            {"params": bert_params, "lr": cfg["lr_bert"]},
            {"params": head_params, "lr": cfg["lr_head"]},
        ],
        weight_decay=cfg["weight_decay"],
    )


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, preds, trues = 0.0, [], []
    for ids, mask, yb in loader:
        ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
        with torch.autocast(device_type=device.type, enabled=USE_AMP):
            logits = model(ids, mask)
            total += criterion(logits.float(), yb).item()
        preds.extend(logits.argmax(1).cpu().tolist())
        trues.extend(yb.cpu().tolist())
    return (
        total / max(len(loader), 1),
        accuracy_score(trues, preds),
        f1_score(trues, preds, average="macro", zero_division=0),
    )


def run_trial(cfg, trial_id):
    set_seed(SEED + trial_id)
    train_loader, val_loader = build_loaders(cfg)

    model = CNN_BERT(cfg, NUM_CLASSES).to(device)
    model.freeze()
    optimizer = make_optimizer(model, cfg)
    criterion = make_criterion(cfg)

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=USE_AMP)

    best_f1, best_state, best_epoch, best_loss, best_acc = -1.0, None, 0, None, None
    no_improve, history = 0, []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for ids, mask, yb in train_loader:
            ids, mask, yb = ids.to(device), mask.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=USE_AMP):
                loss = criterion(model(ids, mask), yb)
            scaler.scale(loss).backward()
            if cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_acc": val_acc, "val_macro_f1": val_f1})
        print(f"  epoch {epoch}/{MAX_EPOCHS} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val acc {val_acc:.4f} | val macroF1 {val_f1:.4f}")

        if val_f1 > best_f1 + 1e-5:
            best_f1, best_epoch = val_f1, epoch
            best_loss, best_acc = val_loss, val_acc
            best_state = copy.deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "val_macro_f1": best_f1, "val_acc": best_acc, "val_loss": best_loss,
        "best_epoch": best_epoch, "epochs_run": len(history), "history": history,
    }, best_state


# ---------------------------------------------------------------
# Random search
# ---------------------------------------------------------------
rng = np.random.default_rng(SEED)

completed = []
if os.path.exists(TRIALS_JSONL):
    with open(TRIALS_JSONL) as f:
        completed = [json.loads(line) for line in f if line.strip()]
    print(f"Resuming: {len(completed)} trial(s) already logged.")

all_configs = [sample_config(rng) for _ in range(N_TRIALS)]
best_overall = max(completed, key=lambda r: r["val_macro_f1"]) if completed else None

for i, cfg in enumerate(all_configs, start=1):
    if i <= len(completed):
        continue
    print(f"\n=== CNN-BERT trial {i}/{N_TRIALS} ===")
    print("  config:", {k: (round(v, 7) if isinstance(v, float) else v)
                        for k, v in cfg.items()})
    t0 = time.time()
    metrics, state = run_trial(cfg, i)
    elapsed = time.time() - t0

    record = {
        "trial": i,
        **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in cfg.items()},
        **{k: v for k, v in metrics.items() if k != "history"},
        "seconds": round(elapsed, 1),
    }
    with open(TRIALS_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    completed.append(record)

    print(f"  -> val macro-F1 {metrics['val_macro_f1']:.4f} "
          f"(best epoch {metrics['best_epoch']}, {elapsed/60:.1f} min)")

    if best_overall is None or metrics["val_macro_f1"] > best_overall["val_macro_f1"]:
        best_overall = record
        torch.save(state, BEST_CKPT)
        with open(BEST_CFG, "w") as f:
            json.dump({
                "branch": "CNN + BERT Tokenizer",
                "trial": i,
                "config": cfg,
                "bert_name": BERT_NAME,
                "num_classes": NUM_CLASSES,
                "label_names": LABEL_NAMES,
                "val_macro_f1": metrics["val_macro_f1"],
                "val_acc": metrics["val_acc"],
                "val_loss": metrics["val_loss"],
                "best_epoch": metrics["best_epoch"],
                "checkpoint": BEST_CKPT,
                "history": metrics["history"],
                "search": {"n_trials": N_TRIALS, "method": "random search",
                           "seed": SEED, "space": SEARCH_SPACE,
                           "lr_bert_range": LR_BERT_RANGE,
                           "lr_head_range": LR_HEAD_RANGE},
            }, f, indent=2)
        print("  *** new best — checkpoint + config saved")

trials_df = pd.DataFrame(completed).sort_values("val_macro_f1", ascending=False)
with pd.ExcelWriter(TRIALS_XLSX, engine="openpyxl") as writer:
    trials_df.to_excel(writer, sheet_name="All Trials", index=False)
    trials_df.head(10).to_excel(writer, sheet_name="Top 10", index=False)

print("\n" + "=" * 60)
print(f"Best CNN-BERT val macro-F1: {best_overall['val_macro_f1']:.4f} "
      f"(trial {best_overall['trial']})")
print(f"Checkpoint : {BEST_CKPT}")
print(f"Config     : {BEST_CFG}")
print(f"Trial table: {TRIALS_XLSX}")
