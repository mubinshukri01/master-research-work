# ===============================================================
# step1_export_logits.py
#
# Runs the two trained branches ONCE over the val / cal / test
# partitions and dumps their RAW LOGITS to disk.
#
# Why raw logits and not probabilities: temperature scaling is
# applied to logits. Dumping logits means every downstream
# experiment (temperature, alpha, per-class weights, prior
# correction) becomes a CPU-only script that runs in seconds,
# with no GPU and no re-inference.
#
# This is the LAST script that needs the GPU for steps 1-3.
#
# Run from the "Final experiment" folder:
#     python improvement/step1_export_logits.py
# ===============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------
# CONFIG - adjust only if your folder layout differs
# ---------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CSV   = os.path.join(ROOT, "redo", "canonical_dataset.csv")
SPLITS_NPZ = os.path.join(ROOT, "redo", "splits.npz")

GRU_CKPT = os.path.join(
    ROOT,
    "base essemble model experiment #1 (BERT + CNN & NLP + GRU)",
    "optimized essemble model experiment",
    "epoch GRU keras model", "epoch Version 2", "gru_best.pt",
)
CNN_CKPT = os.path.join(
    ROOT,
    "base essemble model experiment #1 (BERT + CNN & NLP + GRU)",
    "optimized essemble model experiment",
    "epoch CNN BERT model", "epoch Version 2", "cnn_bert_best.pt",
)

OUT_DIR  = os.path.join(ROOT, "improvement")
OUT_NPZ  = os.path.join(OUT_DIR, "logits_dump.npz")

TEXT_COL  = "comment/tweet"
LABEL_COL = "majority_sent"
MAX_LEN   = 50
VOCAB     = 20000
BERT_NAME = "bert-base-cased"
BATCH     = 32

# The base ensemble's published numbers, used as a self-check that
# this dump faithfully reproduces the existing pipeline.
BASE_ALPHA_CNN  = 0.6
EXPECTED_BASE_ACC = 0.634146
EXPECTED_BASE_F1  = 0.621843

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ===============================================================
# 1. Data + canonical split
# ===============================================================
df = pd.read_csv(DATA_CSV)
texts  = df[TEXT_COL].astype(str).tolist()
labels = df[LABEL_COL].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
label_names = list(le.classes_)
NUM_CLASSES = len(label_names)
print("Classes (encoded order):", label_names)

splits = np.load(SPLITS_NPZ)
train_idx = splits["train"]
val_idx   = splits["val"]
cal_idx   = splits["cal"]
test_idx  = splits["test"]

for a in ["train", "val", "cal", "test"]:
    for b in ["train", "val", "cal", "test"]:
        if a < b:
            ov = len(set(splits[a].tolist()) & set(splits[b].tolist()))
            assert ov == 0, f"LEAKAGE: {a}/{b} overlap by {ov}"
print("Overlap check passed.")
print(f"Sizes  train={len(train_idx)}  val={len(val_idx)}  "
      f"cal={len(cal_idx)}  test={len(test_idx)}")

# ===============================================================
# 2. Model definitions - must match the training scripts exactly
# ===============================================================
class GRU_Model(nn.Module):
    def __init__(self, vocab_size=VOCAB, embed_dim=128, hidden=128,
                 num_classes=NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


class CNN_BERT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_NAME)
        self.conv = nn.Conv1d(768, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(2)
        return self.fc(x)


gru = GRU_Model().to(device)
gru.load_state_dict(torch.load(GRU_CKPT, map_location=device))
gru.eval()
print("Loaded GRU checkpoint.")

cnn = CNN_BERT().to(device)
cnn.load_state_dict(torch.load(CNN_CKPT, map_location=device))
cnn.eval()
print("Loaded CNN-BERT checkpoint.")

bert_tok = BertTokenizer.from_pretrained(BERT_NAME)

# ===============================================================
# 3. Keras tokenizer - refit on the canonical TRAIN partition,
#    identical to how Train_GRU.py fit it, so token IDs line up
#    with the weights actually stored in gru_best.pt.
# ===============================================================
keras_tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
keras_tok.fit_on_texts([texts[i] for i in train_idx])


def keras_tensor(text_list):
    seqs = keras_tok.texts_to_sequences(text_list)
    seqs = pad_sequences(seqs, maxlen=MAX_LEN)
    return torch.tensor(seqs, dtype=torch.long)


# ===============================================================
# 4. Ordered inference
#
# Both branches are driven from the SAME python list, batched by
# explicit slicing. No DataLoader, no zip() of two iterators - the
# original scripts paired a DataLoader with torch.split() and relied
# on both yielding batches in identical order. That happens to hold
# with shuffle=False, but it is a silent-corruption risk not worth
# carrying. Row i of every returned array corresponds to row i of
# the partition index array.
# ===============================================================
@torch.no_grad()
def run_partition(idx_array, name):
    part_texts = [texts[i] for i in idx_array]
    part_y = y[idx_array]
    n = len(part_texts)

    xk = keras_tensor(part_texts)

    gru_logits, cnn_logits = [], []
    for s in range(0, n, BATCH):
        e = min(s + BATCH, n)

        gru_logits.append(gru(xk[s:e].to(device)).cpu())

        enc = bert_tok(
            part_texts[s:e],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )
        cnn_logits.append(
            cnn(enc["input_ids"].to(device),
                enc["attention_mask"].to(device)).cpu()
        )

        if (s // BATCH) % 10 == 0:
            print(f"  {name}: {e}/{n}", end="\r")

    gru_logits = torch.cat(gru_logits).numpy().astype(np.float64)
    cnn_logits = torch.cat(cnn_logits).numpy().astype(np.float64)

    assert gru_logits.shape == (n, NUM_CLASSES)
    assert cnn_logits.shape == (n, NUM_CLASSES)
    print(f"  {name}: {n}/{n} done          ")
    return gru_logits, cnn_logits, part_y


print("\nRunning inference...")
out = {}
for name, idx in [("val", val_idx), ("cal", cal_idx), ("test", test_idx)]:
    g, c, yy = run_partition(idx, name)
    out[f"gru_logits_{name}"] = g
    out[f"cnn_logits_{name}"] = c
    out[f"y_{name}"] = yy
    out[f"idx_{name}"] = idx

# ===============================================================
# 5. VERIFICATION - reproduce the published base ensemble numbers
#
# The base script fuses UNSCALED softmax probabilities at
# alpha=0.6 (CNN) / 0.4 (GRU) on the test partition. If this dump
# is faithful, recomputing that from the saved logits must land on
# the numbers already in Base_Ensemble_Results.xlsx. If it does
# not, something upstream is misaligned and nothing downstream
# should be trusted.
# ===============================================================
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


p_cnn = softmax(out["cnn_logits_test"])
p_gru = softmax(out["gru_logits_test"])
base_pred = np.argmax(BASE_ALPHA_CNN * p_cnn + (1 - BASE_ALPHA_CNN) * p_gru, axis=1)

acc = accuracy_score(out["y_test"], base_pred)
f1m = f1_score(out["y_test"], base_pred, average="macro")

print("\n===== VERIFICATION vs published base ensemble =====")
print(f"  accuracy  recomputed {acc:.6f}   expected {EXPECTED_BASE_ACC:.6f}   "
      f"diff {abs(acc - EXPECTED_BASE_ACC):.6f}")
print(f"  macro-F1  recomputed {f1m:.6f}   expected {EXPECTED_BASE_F1:.6f}   "
      f"diff {abs(f1m - EXPECTED_BASE_F1):.6f}")

ok = abs(acc - EXPECTED_BASE_ACC) < 5e-3 and abs(f1m - EXPECTED_BASE_F1) < 5e-3
if ok:
    print("  PASS - dump reproduces the existing pipeline.")
else:
    print("  FAIL - dump does NOT match. Do not run step 2 yet.")
    print("  Most likely causes: wrong checkpoint path, a different")
    print("  bert-base-cased revision, or the Keras tokenizer being")
    print("  fit on something other than the canonical train partition.")

# ===============================================================
# 6. Save
# ===============================================================
np.savez_compressed(OUT_NPZ, label_names=np.array(label_names), **out)

with open(os.path.join(OUT_DIR, "logits_dump_manifest.json"), "w") as f:
    json.dump({
        "label_names": label_names,
        "num_classes": int(NUM_CLASSES),
        "sizes": {k: int(len(out[f"y_{k}"])) for k in ["val", "cal", "test"]},
        "gru_ckpt": GRU_CKPT,
        "cnn_ckpt": CNN_CKPT,
        "bert_name": BERT_NAME,
        "max_len": MAX_LEN,
        "vocab": VOCAB,
        "verification": {
            "base_alpha_cnn": BASE_ALPHA_CNN,
            "recomputed_accuracy": float(acc),
            "recomputed_macro_f1": float(f1m),
            "expected_accuracy": EXPECTED_BASE_ACC,
            "expected_macro_f1": EXPECTED_BASE_F1,
            "passed": bool(ok),
        },
    }, f, indent=2)

print(f"\nSaved logits to: {OUT_NPZ}")
print("Next: python improvement/step2_diagnose_and_fit.py")
