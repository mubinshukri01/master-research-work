# ================================================================
#  Dual-Representation Sentiment Ensemble (DRSE) — V3 TUNED
#  CNN + BERT Tokenizer  ⊕  GRU + Keras Tokenizer
#
#  This is the tuned counterpart to "experiment V2 (base ensemble model).py".
#  Three things changed:
#
#  1. Both branches are rebuilt from the config manifests written by the
#     two V3 tuning scripts, so the architecture is whatever the search
#     found — not the hard-coded 128-dim GRU / single-3-kernel CNN of V2.
#  2. The Keras tokenizer is LOADED from disk rather than refit. V2 refit it
#     here and relied on that refit landing on identical word->index IDs;
#     that only held because the hyperparameters happened to match. Once
#     vocab_size and max_len are tuned it would silently corrupt every input.
#  3. The fusion weight ALPHA (and a per-branch temperature) is now SELECTED
#     ON THE VALIDATION SET instead of being fixed at 0.6/0.4 by assumption.
#     The test set is touched exactly once, for final reporting.
#
#  Inference is batched, so this runs in seconds rather than looping over
#  one test sample at a time as V2 did.
# ================================================================

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizerFast, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
)
from collections import Counter

# ---------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DATA_PATH = "redo/canonical_dataset.csv"
SPLITS_PATH = "redo/splits.npz"

GRU_DIR = ("base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
           "optimized essemble model experiment/tuned GRU keras model/V3")
CNN_DIR = ("base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
           "optimized essemble model experiment/tuned CNN BERT model/V3")

GRU_CFG_PATH = os.path.join(GRU_DIR, "gru_best_config.json")
GRU_CKPT = os.path.join(GRU_DIR, "gru_best.pt")
GRU_TOKENIZER = os.path.join(GRU_DIR, "gru_tokenizer.json")
CNN_CFG_PATH = os.path.join(CNN_DIR, "cnn_bert_best_config.json")
CNN_CKPT = os.path.join(CNN_DIR, "cnn_bert_best.pt")

RESULT_DIR = ("base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
              "Result/Version 3 tuned results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Baseline weights from V2, kept only so the report can show what the
# untuned fusion would have scored on the same checkpoints.
BASE_ALPHA, BASE_BETA = 0.6, 0.4

BATCH_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
label_names = list(le.classes_)
NUM_CLASSES = len(label_names)

splits = np.load(SPLITS_PATH)
val_idx, test_idx = splits["val"], splits["test"]

X_val_texts = [texts[i] for i in val_idx]
X_test_texts = [texts[i] for i in test_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print(f"Val: {len(X_val_texts)} | Test: {len(X_test_texts)} | Classes: {NUM_CLASSES}")

gru_cfg_blob = json.load(open(GRU_CFG_PATH))
cnn_cfg_blob = json.load(open(CNN_CFG_PATH))
gru_cfg = gru_cfg_blob["config"]
cnn_cfg = cnn_cfg_blob["config"]

print("\nGRU  best config :", gru_cfg)
print("     val macro-F1:", round(gru_cfg_blob["val_macro_f1"], 4))
print("CNN  best config :", cnn_cfg)
print("     val macro-F1:", round(cnn_cfg_blob["val_macro_f1"], 4))


# ===============================================================
#  Model definitions — must stay byte-for-byte equivalent to the
#  classes in the two tuning scripts, or load_state_dict will fail.
# ===============================================================
class GRU_Model(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"], padding_idx=0)
        self.gru = nn.GRU(
            cfg["embed_dim"], cfg["hidden_dim"],
            num_layers=cfg["num_layers"], batch_first=True,
            bidirectional=cfg["bidirectional"], dropout=cfg["rnn_dropout"],
        )
        dirs = 2 if cfg["bidirectional"] else 1
        feat = cfg["hidden_dim"] * dirs
        if cfg["pooling"] == "meanmax":
            feat *= 2
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(feat, num_classes)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        emb = self.embedding(x)
        out, h = self.gru(emb)
        if self.cfg["pooling"] == "last":
            dirs = 2 if self.cfg["bidirectional"] else 1
            h = h.view(self.cfg["num_layers"], dirs, x.size(0), self.cfg["hidden_dim"])
            feat = torch.cat([h[-1, d] for d in range(dirs)], dim=1)
        else:
            denom = mask.sum(dim=1).clamp(min=1.0)
            mean = (out * mask).sum(dim=1) / denom
            mx = out.masked_fill(mask == 0, float("-inf")).max(dim=1).values
            mx = torch.nan_to_num(mx, neginf=0.0)
            feat = torch.cat([mean, mx], dim=1)
        return self.fc(self.dropout(feat))


class CNN_BERT(nn.Module):
    def __init__(self, cfg, num_classes, bert_name):
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

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x * mask.unsqueeze(-1).to(x.dtype)
        x = x.permute(0, 2, 1)
        feats = [self.pool(torch.relu(conv(x))).squeeze(2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(feats, dim=1)))


# ---------------------------------------------------------------
# Load branch A: GRU + Keras tokenizer (tokenizer LOADED, not refit)
# ---------------------------------------------------------------
with open(GRU_TOKENIZER, encoding="utf-8") as f:
    _tok_json = f.read()
try:
    keras_tok = tokenizer_from_json(_tok_json)
except Exception as e:
    # Some Keras builds serialise/deserialise the tokenizer differently.
    # Rebuilding from word_index gives identical token IDs either way.
    print(f"tokenizer_from_json failed ({e}); rebuilding from word_index")
    _blob = json.loads(_tok_json)
    _cfg = json.loads(_blob["config"]) if "config" in _blob else _blob
    _wi = _cfg["word_index"]
    if isinstance(_wi, str):
        _wi = json.loads(_wi)
    keras_tok = Tokenizer(num_words=gru_cfg["vocab_size"], oov_token="<OOV>")
    keras_tok.word_index = {w: int(i) for w, i in _wi.items()}
    keras_tok.index_word = {i: w for w, i in keras_tok.word_index.items()}

gru = GRU_Model(gru_cfg, NUM_CLASSES).to(device)
gru.load_state_dict(torch.load(GRU_CKPT, map_location=device))
gru.eval()


def keras_encode(text_list):
    seqs = keras_tok.texts_to_sequences(text_list)
    return torch.tensor(pad_sequences(seqs, maxlen=gru_cfg["max_len"]), dtype=torch.long)


# ---------------------------------------------------------------
# Load branch B: CNN + BERT
# ---------------------------------------------------------------
bert_name = cnn_cfg_blob.get("bert_name", "bert-base-cased")
bert_tok = BertTokenizerFast.from_pretrained(bert_name)

cnn = CNN_BERT(cnn_cfg, NUM_CLASSES, bert_name).to(device)
cnn.load_state_dict(torch.load(CNN_CKPT, map_location=device))
cnn.eval()


# ---------------------------------------------------------------
# Batched logit extraction
# ---------------------------------------------------------------
@torch.no_grad()
def branch_logits(text_list):
    """Return (gru_logits, cnn_logits) as float64 numpy arrays."""
    gru_out, cnn_out = [], []
    for s in range(0, len(text_list), BATCH_SIZE):
        chunk = text_list[s:s + BATCH_SIZE]

        k = keras_encode(chunk).to(device)
        gru_out.append(gru(k).float().cpu().numpy())

        enc = bert_tok(chunk, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=cnn_cfg["max_len"])
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        cnn_out.append(cnn(ids, mask).float().cpu().numpy())

    return (np.concatenate(gru_out).astype(np.float64),
            np.concatenate(cnn_out).astype(np.float64))


def softmax_T(logits, T):
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


print("\nExtracting branch logits...")
gru_val_logits, cnn_val_logits = branch_logits(X_val_texts)
gru_test_logits, cnn_test_logits = branch_logits(X_test_texts)

# ===============================================================
#  FUSION-WEIGHT TUNING — on the VALIDATION set only
#
#  Searches ALPHA (weight on CNN+BERT; BETA = 1 - ALPHA) jointly with a
#  temperature per branch. Temperature scaling matters here because the
#  two branches are differently calibrated: a fine-tuned BERT tends to be
#  far more confident than a small GRU, so an un-scaled 0.6/0.4 blend is
#  effectively dominated by whichever branch is sharper, regardless of the
#  nominal weights.
# ===============================================================
ALPHA_GRID = np.round(np.arange(0.0, 1.001, 0.01), 3)
TEMP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

gru_val_probs = {T: softmax_T(gru_val_logits, T) for T in TEMP_GRID}
cnn_val_probs = {T: softmax_T(cnn_val_logits, T) for T in TEMP_GRID}

fusion_rows = []
best = {"macro_f1": -1.0}

for t_cnn in TEMP_GRID:
    for t_gru in TEMP_GRID:
        pc, pg = cnn_val_probs[t_cnn], gru_val_probs[t_gru]
        for a in ALPHA_GRID:
            preds = (a * pc + (1 - a) * pg).argmax(axis=1)
            f1m = f1_score(y_val, preds, average="macro", zero_division=0)
            acc = accuracy_score(y_val, preds)
            fusion_rows.append({"alpha": float(a), "beta": float(1 - a),
                                "T_cnn": t_cnn, "T_gru": t_gru,
                                "val_macro_f1": f1m, "val_accuracy": acc})
            if f1m > best["macro_f1"]:
                best = {"macro_f1": f1m, "acc": acc, "alpha": float(a),
                        "T_cnn": t_cnn, "T_gru": t_gru}

fusion_df = pd.DataFrame(fusion_rows).sort_values("val_macro_f1", ascending=False)

ALPHA = best["alpha"]
BETA = 1.0 - ALPHA
T_CNN, T_GRU = best["T_cnn"], best["T_gru"]

print("\n--- Tuned fusion (selected on validation) ---")
print(f"ALPHA (CNN+BERT) = {ALPHA:.2f} | BETA (GRU) = {BETA:.2f}")
print(f"T_cnn = {T_CNN} | T_gru = {T_GRU}")
print(f"Validation macro-F1 = {best['macro_f1']:.4f} | accuracy = {best['acc']:.4f}")

# ===============================================================
#  FINAL EVALUATION — test set, touched once
# ===============================================================
true = y_test.tolist()

tuned_probs = (ALPHA * softmax_T(cnn_test_logits, T_CNN)
               + BETA * softmax_T(gru_test_logits, T_GRU))
final_preds = tuned_probs.argmax(axis=1).tolist()

# Untuned reference: V2's fixed 0.6/0.4 blend at temperature 1, on the
# same checkpoints — this isolates the gain attributable to fusion tuning.
base_preds = (BASE_ALPHA * softmax_T(cnn_test_logits, 1.0)
              + BASE_BETA * softmax_T(gru_test_logits, 1.0)).argmax(axis=1).tolist()

# Single-branch references, for the ablation table.
gru_only = gru_test_logits.argmax(axis=1).tolist()
cnn_only = cnn_test_logits.argmax(axis=1).tolist()


def metric_row(name, preds):
    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(
        true, preds, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        true, preds, average="weighted", zero_division=0)
    return {"Model": name, "Accuracy": accuracy_score(true, preds),
            "Precision (Macro)": p_ma, "Recall (Macro)": r_ma, "F1 (Macro)": f_ma,
            "Precision (Weighted)": p_w, "Recall (Weighted)": r_w,
            "F1 (Weighted)": f_w}


comparison_df = pd.DataFrame([
    metric_row("GRU + Keras (single branch)", gru_only),
    metric_row("CNN + BERT (single branch)", cnn_only),
    metric_row(f"Base ensemble (a={BASE_ALPHA}, b={BASE_BETA}, T=1)", base_preds),
    metric_row(f"Tuned ensemble (a={ALPHA:.2f}, T_cnn={T_CNN}, T_gru={T_GRU})", final_preds),
])

print("\n=== Test-set comparison ===")
print(comparison_df.to_string(index=False))

print("\nTuned ensemble sentiment distribution:", Counter(final_preds))

# ---------- Overall metrics for the tuned ensemble ----------
acc = accuracy_score(true, final_preds)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    true, final_preds, average="macro", zero_division=0)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    true, final_preds, average="micro", zero_division=0)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    true, final_preds, average="weighted", zero_division=0)

overall_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision (Macro)": prec_macro, "Recall (Macro)": rec_macro, "F1 (Macro)": f1_macro,
    "Precision (Micro)": prec_micro, "Recall (Micro)": rec_micro, "F1 (Micro)": f1_micro,
    "Precision (Weighted)": prec_weighted, "Recall (Weighted)": rec_weighted,
    "F1 (Weighted)": f1_weighted,
}])

prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
    true, final_preds, labels=range(NUM_CLASSES), zero_division=0)
per_class_df = pd.DataFrame({"Sentiment": label_names, "Precision": prec_cls,
                             "Recall": rec_cls, "F1": f1_cls, "Support": support})

human_counts = pd.Series(true).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
ens_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

dist_counts = pd.DataFrame({"Sentiment": label_names,
                            "Human Count": human_counts.values,
                            "Ensemble Count": ens_counts.values})
dist_pct_df = pd.DataFrame({
    "Sentiment": label_names,
    "Human (%)": (human_counts / human_counts.sum() * 100).values,
    "Ensemble (%)": (ens_counts / ens_counts.sum() * 100).values})

report_dict = classification_report(true, final_preds, target_names=label_names,
                                    output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).T.reset_index().rename(
    columns={"index": "Metric/Class"})

hp_df = pd.DataFrame([
    {"Branch": "GRU + Keras", "Hyperparameter": k, "Selected value": str(v)}
    for k, v in gru_cfg.items()
] + [
    {"Branch": "CNN + BERT", "Hyperparameter": k, "Selected value": str(v)}
    for k, v in cnn_cfg.items()
] + [
    {"Branch": "Ensemble", "Hyperparameter": "alpha (CNN+BERT)", "Selected value": f"{ALPHA:.2f}"},
    {"Branch": "Ensemble", "Hyperparameter": "beta (GRU)", "Selected value": f"{BETA:.2f}"},
    {"Branch": "Ensemble", "Hyperparameter": "temperature (CNN)", "Selected value": str(T_CNN)},
    {"Branch": "Ensemble", "Hyperparameter": "temperature (GRU)", "Selected value": str(T_GRU)},
])

OUTPUT_PATH = os.path.join(RESULT_DIR, "Tuned_Ensemble_Results.xlsx")
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    comparison_df.to_excel(writer, sheet_name="Base vs Tuned", index=False)
    hp_df.to_excel(writer, sheet_name="Selected Hyperparameters", index=False)
    per_class_df.to_excel(writer, sheet_name="Per-Class PRF", index=False)
    dist_counts.to_excel(writer, sheet_name="Sentiment Counts", index=False)
    dist_pct_df.to_excel(writer, sheet_name="Sentiment Percentages", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)
    fusion_df.head(200).to_excel(writer, sheet_name="Fusion Search (top 200)", index=False)

print(f"\nAll results saved to: {OUTPUT_PATH}")

# ===============================================================
#  FIGURE 1 — fusion weight sensitivity (validation)
# ===============================================================
best_temp_slice = fusion_df[(fusion_df["T_cnn"] == T_CNN) & (fusion_df["T_gru"] == T_GRU)]
best_temp_slice = best_temp_slice.sort_values("alpha")

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(best_temp_slice["alpha"], best_temp_slice["val_macro_f1"], linewidth=2)
ax.axvline(ALPHA, linestyle="--", color="red",
           label=f"selected α = {ALPHA:.2f}")
ax.axvline(BASE_ALPHA, linestyle=":", color="grey",
           label=f"V2 fixed α = {BASE_ALPHA}")
ax.set_xlabel("α  (weight on CNN + BERT;  β = 1 − α)")
ax.set_ylabel("Validation macro-F1")
ax.set_title("Ensemble fusion weight sensitivity")
ax.grid(alpha=0.4, linestyle="--")
ax.legend()
plt.tight_layout()
ALPHA_FIG = os.path.join(RESULT_DIR, "fusion_weight_sensitivity.png")
plt.savefig(ALPHA_FIG, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {ALPHA_FIG}")

# ===============================================================
#  FIGURE 2 — sentiment distribution, human vs tuned ensemble
# ===============================================================
human_counts.index = label_names
ens_counts.index = label_names
dist_df = pd.DataFrame({"Human": human_counts, "Ensemble": ens_counts}).T
dist_pct = dist_df.div(dist_df.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(9, 6))
bottom = np.zeros(len(dist_df))
for sentiment in label_names:
    values = dist_pct[sentiment].values
    bars = ax.bar(dist_pct.index, values, bottom=bottom, label=sentiment)
    for i, bar in enumerate(bars):
        if values[i] > 0:
            count = dist_df.loc[dist_pct.index[i], sentiment]
            ax.text(bar.get_x() + bar.get_width() / 2, bottom[i] + values[i] / 2,
                    f"{count}\n{values[i]*100:.1f}%", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
    bottom += values

ax.set_title("Sentiment Distribution: Human vs Tuned DRSE Ensemble")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
FIG_PATH = os.path.join(RESULT_DIR, "tuned_sentiment_distribution.png")
plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {FIG_PATH}")

DIST_OUTPUT_PATH = os.path.join(RESULT_DIR, "tuned_sentiment_distribution.xlsx")
with pd.ExcelWriter(DIST_OUTPUT_PATH, engine="openpyxl") as writer:
    dist_df.reset_index().rename(columns={"index": "Source"}).to_excel(
        writer, sheet_name="Counts", index=False)
    (dist_pct * 100).reset_index().rename(columns={"index": "Source"}).to_excel(
        writer, sheet_name="Percentages", index=False)
print(f"Distribution data saved to: {DIST_OUTPUT_PATH}")

# ---------------------------------------------------------------
# Machine-readable record of the final tuned system
# ---------------------------------------------------------------
with open(os.path.join(RESULT_DIR, "tuned_ensemble_manifest.json"), "w") as f:
    json.dump({
        "gru": {"config": gru_cfg, "checkpoint": GRU_CKPT,
                "val_macro_f1": gru_cfg_blob["val_macro_f1"]},
        "cnn_bert": {"config": cnn_cfg, "checkpoint": CNN_CKPT,
                     "val_macro_f1": cnn_cfg_blob["val_macro_f1"]},
        "fusion": {"alpha": ALPHA, "beta": BETA, "T_cnn": T_CNN, "T_gru": T_GRU,
                   "selected_on": "validation macro-F1",
                   "val_macro_f1": best["macro_f1"]},
        "test": {"accuracy": acc, "macro_f1": f1_macro,
                 "weighted_f1": f1_weighted},
    }, f, indent=2)

print("\nDone.")
