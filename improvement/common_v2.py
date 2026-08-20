# ===============================================================
# common_v2.py
#
# Shared plumbing for step 4 (tuning) and step 5 (re-export).
# Nothing here trains anything - it exists so the tuner and the
# exporter cannot possibly disagree about the split, the label
# encoding, or the tokenizer.
#
# The single most important invariant: NOTHING in this file or
# anything that imports it ever calls train_test_split. The
# partition comes from redo/splits.npz and nowhere else.
# ===============================================================

import os
import json
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_CSV = os.path.join(ROOT, "redo", "canonical_dataset.csv")
SPLITS_NPZ = os.path.join(ROOT, "redo", "splits.npz")
WORK_DIR = os.path.join(ROOT, "improvement", "v2")
os.makedirs(WORK_DIR, exist_ok=True)

TEXT_COL = "comment/tweet"
LABEL_COL = "majority_sent"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_canonical():
    """Returns texts, y, label_names, and the four index arrays."""
    df = pd.read_csv(DATA_CSV)
    texts = df[TEXT_COL].astype(str).tolist()

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COL].tolist())
    label_names = list(le.classes_)

    s = np.load(SPLITS_NPZ)
    idx = {k: s[k] for k in ["train", "val", "cal", "test"]}

    keys = list(idx)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ov = np.intersect1d(idx[keys[i]], idx[keys[j]]).size
            assert ov == 0, f"LEAKAGE: {keys[i]}/{keys[j]} overlap by {ov}"

    return texts, y, label_names, idx


def class_weights(y_train, num_classes):
    """Inverse-frequency weights, normalised to mean 1.

    This is the lever that targets the neutral collapse at the LOSS
    level rather than post-hoc at the decision boundary. Post-hoc bias
    correction (step 2's M2) could only trade positive recall away for
    neutral recall. Weighting the loss lets the branch actually learn
    better neutral features instead of just relabelling its outputs.
    """
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    w = counts.sum() / (num_classes * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Optional alternative to weighted CE: down-weights easy examples."""

    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits, target):
        ce = self.ce(logits, target)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
class GRUBranch(nn.Module):
    """Keras-tokenised GRU branch, now with tunable capacity.

    The original was fixed at embed=128, hidden=128, 1 layer,
    unidirectional, no dropout. Every one of those is now a search
    dimension.
    """

    def __init__(self, vocab, num_classes, embed_dim=128, hidden=128,
                 layers=1, bidirectional=False, dropout=0.0, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), num_classes)
        self.bidirectional = bidirectional
        self.layers = layers

    def forward(self, x):
        e = self.embedding(x)
        _, h = self.gru(e)
        if self.bidirectional:
            h = torch.cat([h[-2], h[-1]], dim=1)
        else:
            h = h[-1]
        return self.fc(self.drop(h))


class CNNTransformerBranch(nn.Module):
    """CNN head over a transformer encoder.

    Architecturally the same idea as the original CNN-BERT (conv over
    token states, max-pool, linear) so the thesis narrative carries
    over. What changes: the encoder can be XLM-R, multiple kernel
    widths are supported, and the embedding matrix can be frozen.

    Freezing embeddings matters on a 6 GB card: xlm-roberta-base has
    ~192M of its ~278M parameters in the embedding table alone.
    Freezing it removes those from the Adam optimiser state and cuts
    memory roughly in half with little accuracy cost for fine-tuning.
    """

    def __init__(self, encoder_name, num_classes, n_filters=128,
                 kernel_sizes=(3,), dropout=0.1, freeze_embeddings=True,
                 unfrozen_layers=None):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hid = self.encoder.config.hidden_size

        if freeze_embeddings:
            for p in self.encoder.embeddings.parameters():
                p.requires_grad = False

        if unfrozen_layers is not None:
            layers = getattr(self.encoder.encoder, "layer", None)
            if layers is not None:
                total = len(layers)
                for i, layer in enumerate(layers):
                    req = i >= total - unfrozen_layers
                    for p in layer.parameters():
                        p.requires_grad = req

        self.convs = nn.ModuleList(
            [nn.Conv1d(hid, n_filters, kernel_size=k, padding=k // 2)
             for k in kernel_sizes]
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), num_classes)

    def forward(self, ids, mask):
        h = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        # Zero out padded positions before convolving. Without this the
        # conv sees whatever the encoder emits at pad positions and the
        # max-pool can select those activations. The original CNN-BERT
        # had this flaw; it was harmless only because padding length was
        # constant. Masking makes the head correct regardless.
        h = h * mask.unsqueeze(-1).to(h.dtype)
        h = h.permute(0, 2, 1)
        feats = [self.pool(torch.relu(c(h))).squeeze(2) for c in self.convs]
        return self.fc(self.drop(torch.cat(feats, dim=1)))
