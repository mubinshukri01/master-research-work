# ===============================================================
# common_baseline.py
#
# Shared plumbing for the canonical-split re-run of the eight
# single-model baselines:
#     {Keras word tokenizer, BERT tokenizer} x {CNN, GRU, LSTM, RNN}
#
# Why this file exists
# --------------------
# The originals in ../NLP and ../BERT each carried their own copy of
# ~150 lines of data loading, splitting, training and export. They
# drifted: 5 epochs on one side and 3 on the other, left-padding on
# one side and right-padding on the other, and one file that was a
# copy of another with the epoch count changed. Centralising the
# plumbing is what stops that happening again.
#
# The single most important invariant, inherited from common_v2:
# NOTHING here ever calls train_test_split. The partition comes from
# redo/splits.npz and nowhere else.
#
# The originals are left untouched - they are the provenance record
# for the numbers currently in the thesis.
# ===============================================================

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ../.. from this file is the experiment root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "improvement"))

# Reuse the canonical loader rather than reimplementing it. load_canonical
# asserts zero pairwise overlap between train/val/cal/test, so a leak
# cannot survive an import of this module.
from common_v2 import (  # noqa: E402
    load_canonical,
    class_weights,
    macro_f1,
    set_seed,
    DEVICE,
)

OUT_DIR = os.path.join(ROOT, "Baseline experiment", "canonical rerun", "results")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_LEN = 100
NUM_WORDS = 20000
EMBED_DIM = 128
HIDDEN = 128
BATCH = 32
LR = 1e-3
PAD_IDX = 0

# The baseline originals used bert-base-uncased (BERT + CNN.py:65-66).
# Kept as-is so this is a corrected re-run of the same experiment and
# not a different one. Note it differs from the ensemble branch, which
# uses bert-base-cased - a real inconsistency in the thesis, but one
# that belongs to Stage 2, not here.
BERT_CKPT = "bert-base-uncased"


# ---------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------
def build_keras_inputs(texts, train_idx):
    """Keras word tokenisation, fit on TRAIN ROWS ONLY.

    Returns (ids, lengths, vocab_size).

    Two deliberate departures from the originals:

    * padding='post' / truncating='post'. The originals used the Keras
      defaults, which are 'pre' - i.e. left-padding. The BERT side
      right-pads. Since every recurrent baseline read h_n[-1], the two
      families were reading fundamentally different things: the state
      after the last real word (NLP) versus the state after up to 99
      [PAD] steps (BERT). Right-padding everywhere, combined with the
      length-aware readout below, makes the readout correct and the two
      families comparable.

    * lengths are returned, so the models can find the true final
      timestep instead of assuming it is the last one.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    texts = np.asarray(texts, dtype=object)

    tk = Tokenizer(num_words=NUM_WORDS, oov_token="<OOV>")
    tk.fit_on_texts(texts[train_idx].tolist())

    seqs = tk.texts_to_sequences(texts.tolist())
    lengths = np.array([min(len(s), MAX_LEN) for s in seqs], dtype=np.int64)

    ids = pad_sequences(
        seqs, maxlen=MAX_LEN, padding="post", truncating="post", value=PAD_IDX
    )

    # Keras reserves index 0 for padding and never assigns it to a word,
    # so num_words is an exclusive upper bound on the indices emitted.
    vocab_size = min(NUM_WORDS, len(tk.word_index) + 1)
    return ids.astype(np.int64), lengths, vocab_size, tk


def build_bert_inputs(texts):
    """WordPiece tokenisation. Returns (ids, lengths, vocab_size).

    Unlike the originals, this keeps the attention_mask - they took
    ["input_ids"] only (BERT + CNN.py:65-83), which left nothing
    downstream able to mask padding even in principle.
    """
    from transformers import BertTokenizer

    tok = BertTokenizer.from_pretrained(BERT_CKPT)
    enc = tok(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="np",
    )
    ids = enc["input_ids"].astype(np.int64)
    lengths = enc["attention_mask"].sum(axis=1).astype(np.int64)
    return ids, lengths, tok.vocab_size, tok


# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------
def _mask_from_lengths(lengths, max_len):
    """(B,) lengths -> (B, max_len) float mask, 1.0 on real tokens."""
    ar = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return (ar < lengths.unsqueeze(1)).float()


class BaselineCNN(nn.Module):
    """Conv1d + global max-pool, matching the original architecture.

    The correctness fix over the original is masking: pad positions are
    zeroed before convolving, so the max-pool cannot select an
    activation produced entirely by padding. This mirrors the idiom
    already used in improvement/common_v2.py:198, whose comment notes
    the original CNN-BERT had the same flaw.
    """

    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        self.conv = nn.Conv1d(EMBED_DIM, HIDDEN, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(HIDDEN, num_classes)

    def forward(self, ids, lengths):
        e = self.emb(ids)                                   # (B, L, E)
        e = e * _mask_from_lengths(lengths, ids.size(1)).unsqueeze(-1)
        h = torch.relu(self.conv(e.transpose(1, 2)))        # (B, H, L)
        return self.fc(self.pool(h).squeeze(2))


class BaselineRNN(nn.Module):
    """GRU / LSTM / vanilla RNN with a length-aware readout.

    The originals returned self.fc(h_n[-1]). With right-padding that
    reads the state after the pad run; with left-padding it happens to
    be correct. Gathering at index (length - 1) is correct for
    right-padded input regardless of how much padding there is, which
    is the whole point of switching both families to right-padding.

    Sequences of length 0 (possible when Keras' default filters strip a
    tweet of all tokens - plausible for emoji-only Manglish posts) clamp
    to index 0 rather than crashing.
    """

    def __init__(self, vocab_size, num_classes, cell="gru"):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=PAD_IDX)
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}[cell]
        self.rnn = rnn_cls(EMBED_DIM, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, num_classes)

    def forward(self, ids, lengths):
        out, _ = self.rnn(self.emb(ids))                    # (B, L, H)
        idx = (lengths - 1).clamp(min=0)
        h = out[torch.arange(out.size(0), device=out.device), idx]
        return self.fc(h)


def make_model(head, vocab_size, num_classes):
    if head == "cnn":
        return BaselineCNN(vocab_size, num_classes)
    return BaselineRNN(vocab_size, num_classes, cell=head)


# ---------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------
def _loader(ids, lengths, y, idx, shuffle):
    ds = TensorDataset(
        torch.from_numpy(ids[idx]),
        torch.from_numpy(lengths[idx]),
        torch.from_numpy(np.asarray(y)[idx]),
    )
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, trues = [], []
    for ids, lengths, yb in loader:
        logits = model(ids.to(DEVICE), lengths.to(DEVICE))
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train_and_select(model, ids, lengths, y, idx, max_epochs=10, weighted=True):
    """Train on `train`, select the best epoch on `val` by macro-F1.

    This is the methodological fix. The originals had no validation set
    at all - a single train_test_split whose test half drove metrics,
    figures and exports alike, which also meant the 5-vs-3 epoch choice
    was made with only test-set feedback. Here the epoch count is a
    SELECTED quantity, chosen on val, so the 5-vs-3 confound between the
    two families disappears rather than being carried forward.

    Returns (best_state_dict, best_epoch, history).
    """
    model = model.to(DEVICE)
    train_loader = _loader(ids, lengths, y, idx["train"], shuffle=True)
    val_loader = _loader(ids, lengths, y, idx["val"], shuffle=False)

    w = class_weights(np.asarray(y)[idx["train"]], 3).to(DEVICE) if weighted else None
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_epoch, best_state, history = -1.0, -1, None, []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for bids, blen, by in train_loader:
            opt.zero_grad()
            loss = crit(model(bids.to(DEVICE), blen.to(DEVICE)), by.to(DEVICE))
            loss.backward()
            opt.step()
            total += loss.item() * by.size(0)

        vp, vt = predict(model, val_loader)
        vf1 = macro_f1(vt, vp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / len(idx["train"]),
                "val_macro_f1": vf1,
                "val_accuracy": accuracy_score(vt, vp),
            }
        )
        if vf1 > best_f1:
            best_f1, best_epoch = vf1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"    epoch {epoch:>2}  loss {history[-1]['train_loss']:.4f}  val macro-F1 {vf1:.4f}")

    return best_state, best_epoch, history


def evaluate(model, ids, lengths, y, idx, split, label_names):
    """Metrics on one split. Call with split='test' exactly once."""
    preds, trues = predict(model, _loader(ids, lengths, y, idx[split], shuffle=False))

    row = {"accuracy": accuracy_score(trues, preds)}
    for avg in ("macro", "micro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            trues, preds, average=avg, zero_division=0
        )
        row[f"precision_{avg}"] = p
        row[f"recall_{avg}"] = r
        row[f"f1_{avg}"] = f

    p, r, f, s = precision_recall_fscore_support(
        trues, preds, average=None, labels=range(len(label_names)), zero_division=0
    )
    per_class = [
        {
            "class": label_names[i],
            "precision": p[i],
            "recall": r[i],
            "f1": f[i],
            "support": int(s[i]),
        }
        for i in range(len(label_names))
    ]

    # micro-F1 is identically accuracy in single-label multiclass. The
    # old NLP/performance comparison.py carried commented arrays that
    # violated this, proving they came from a different run.
    assert abs(row["f1_micro"] - row["accuracy"]) < 1e-9, "micro-F1 != accuracy"

    return row, per_class, confusion_matrix(trues, preds), preds
