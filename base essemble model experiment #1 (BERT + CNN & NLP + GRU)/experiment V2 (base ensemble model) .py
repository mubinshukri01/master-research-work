# ================================================================
#  Dual-Representation Sentiment Ensemble (DRSE)
#  CNN + BERT Tokenizer  ⊕  GRU + Keras Tokenizer
# ================================================================

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = False
cudnn.benchmark = True


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load dataset — from the CLEANED canonical file, not the raw Excel
# ---------------------------
df = pd.read_csv("redo/canonical_dataset.csv")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
label_names = list(le.classes_)
NUM_CLASSES = len(label_names)

# ---------------------------
# Load canonical split — evaluate on the SAME test set the optimized
# script will use. X_train_texts is only rebuilt to refit the Keras
# tokenizer identically to Train_GRU.py (needed so word->index IDs
# match what gru_best.pt actually learned) — no training happens here.
# ---------------------------
splits = np.load("redo/splits.npz")
train_idx = splits["train"]
test_idx = splits["test"]

X_train_texts = [texts[i] for i in train_idx]
X_test_texts = [texts[i] for i in test_idx]
y_test = y[test_idx]

# ===============================================================
#  STREAM A: Keras Word Tokenizer + GRU  (load the tuned checkpoint —
#  no training here, so base and optimized ensembles use the SAME
#  underlying GRU model)
# ===============================================================

MAX_LEN = 50
VOCAB = 20000

keras_tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
# Refit on the canonical train partition — identical to how Train_GRU.py
# fit it, so token IDs line up with what gru_best.pt actually learned.
keras_tok.fit_on_texts(X_train_texts)

X_keras_test = keras_tok.texts_to_sequences(X_test_texts)
X_keras_test = pad_sequences(X_keras_test, maxlen=MAX_LEN)
X_keras_test = torch.tensor(X_keras_test, dtype=torch.long)


# GRU model — layer names must match Train_GRU.py exactly (self.embedding,
# not self.embed) or load_state_dict fails to restore the trained
# embedding weights. Also drop the extra Dropout layer Train_GRU.py never
# had, so the architecture is identical, not just eval-mode-equivalent.
class GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))  # ❗ NO softmax


gru = GRU_Model().to(device)
gru.load_state_dict(torch.load("base essemble model experiment #1 (BERT + CNN & NLP + GRU)/optimized essemble model experiment/epoch GRU keras model/epoch Version 2/gru_best.pt", map_location=device))
gru.eval()

# ===============================================================
#  STREAM B: BERT Tokenizer + CNN  (load the tuned checkpoint — same
#  reasoning as Stream A: no training here, and importantly no BERT
#  freezing either, since cnn_bert_best.pt was trained with BERT fully
#  fine-tuned; freezing here would have been architecturally harmless
#  for loading but conceptually wrong to leave in a script that no
#  longer trains anything)
# ===============================================================

bert_tok = BertTokenizer.from_pretrained("bert-base-cased")

class CNN_BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.conv = nn.Conv1d(768, 128, 3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x.permute(0, 2, 1)              # (batch, hidden, seq_len)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)         # (batch, 128)
        return self.fc(x)                   # raw logits


cnn = CNN_BERT().to(device)
cnn.load_state_dict(torch.load("base essemble model experiment #1 (BERT + CNN & NLP + GRU)/optimized essemble model experiment/epoch CNN BERT model/epoch Version 2/cnn_bert_best.pt", map_location=device))
cnn.eval()

# Both models are loaded from disk and already in eval mode above —
# nothing left to switch.

# ===============================================================
#  ENSEMBLE INFERENCE
# ===============================================================

ALPHA = 0.6   # CNN+BERT
BETA = 0.4    # GRU+Keras

final_preds = []
true = []

with torch.no_grad():
    for i in range(len(X_test_texts)):
        k = X_keras_test[i].unsqueeze(0).to(device)
        bert_in = bert_tok(X_test_texts[i], return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        ids = bert_in["input_ids"].to(device)
        mask = bert_in["attention_mask"].to(device)

        p_gru = torch.softmax(gru(k), dim=1)
        p_cnn = torch.softmax(cnn(ids, mask), dim=1)

        ensemble = ALPHA * p_cnn + BETA * p_gru
        final_preds.append(torch.argmax(ensemble, dim=1).item())
        true.append(int(y_test[i]))

# ===============================================================
#  Step 7 : Evaluation
# ===============================================================

print("Ensemble Accuracy:", accuracy_score(true, final_preds))
print("Ensemble Macro-F1:", f1_score(true, final_preds, average="macro"))
from sklearn.metrics import precision_score, recall_score
print("Ensemble Macro-Precision:", precision_score(true, final_preds, average="macro"))
print("Ensemble Macro-Recall:", recall_score(true, final_preds, average="macro"))

print("Ensemble Sentiment Distribution:", Counter(final_preds))

# ---------- Overall metrics ----------
acc = accuracy_score(true, final_preds)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    true, final_preds, average="macro", zero_division=0
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    true, final_preds, average="micro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    true, final_preds, average="weighted", zero_division=0
)

overall_df = pd.DataFrame([{
    "Accuracy": acc,
    "Precision (Macro)": prec_macro,
    "Recall (Macro)": rec_macro,
    "F1 (Macro)": f1_macro,
    "Precision (Micro)": prec_micro,
    "Recall (Micro)": rec_micro,
    "F1 (Micro)": f1_micro,
    "Precision (Weighted)": prec_weighted,
    "Recall (Weighted)": rec_weighted,
    "F1 (Weighted)": f1_weighted
}])

# ---------- Per-class Precision & Recall ----------
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
    true, final_preds, labels=range(NUM_CLASSES), zero_division=0
)

per_class_df = pd.DataFrame({
    "Sentiment": label_names,
    "Precision": prec_cls,
    "Recall": rec_cls,
    "F1": f1_cls,
    "Support": support
})

# ---------- Sentiment Distribution (Counts & %) ----------
human_counts = pd.Series(true).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
ens_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

dist_counts = pd.DataFrame({
    "Sentiment": label_names,
    "Human Count": human_counts.values,
    "Ensemble Count": ens_counts.values
})

dist_pct = pd.DataFrame({
    "Sentiment": label_names,
    "Human (%)": (human_counts / human_counts.sum() * 100).values,
    "Ensemble (%)": (ens_counts / ens_counts.sum() * 100).values
})

# ---------- Classification report (string → table) ----------
report_dict = classification_report(
    true, final_preds, target_names=label_names, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "Metric/Class"})

# ---------- Write everything to ONE Excel file ----------
OUTPUT_PATH = "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/Result/Version 2 results/Base_Ensemble_Results.xlsx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    per_class_df.to_excel(writer, sheet_name="Per-Class PRF", index=False)
    dist_counts.to_excel(writer, sheet_name="Sentiment Counts", index=False)
    dist_pct.to_excel(writer, sheet_name="Sentiment Percentages", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)

print(f"All results saved to: {OUTPUT_PATH}")

# ===============================================================
#  SENTIMENT DISTRIBUTION — STACKED BAR (WITH VALUES & %)
# ===============================================================



# Map numeric labels back to sentiment names
label_names = list(le.classes_)  # e.g., ["negative", "neutral", "positive"]

# ----- Human (gold) distribution -----
human_counts = pd.Series(true).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
human_counts.index = label_names

# ----- Ensemble distribution -----
ens_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
ens_counts.index = label_names

# Build dataframe
dist_df = pd.DataFrame({
    "Human": human_counts,
    "Ensemble": ens_counts
}).T

# Percentages
dist_pct = dist_df.div(dist_df.sum(axis=1), axis=0)

# ---- Plot (stacked bar) ----
fig, ax = plt.subplots(figsize=(9, 6))
bottom = np.zeros(len(dist_df))

for sentiment in label_names:
    values = dist_pct[sentiment].values
    bars = ax.bar(dist_pct.index, values, bottom=bottom, label=sentiment)
    
    # Add labels inside each stack: count + percentage
    for i, bar in enumerate(bars):
        if values[i] > 0:
            count = dist_df.loc[dist_pct.index[i], sentiment]
            pct = values[i] * 100
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bottom[i] + values[i]/2,
                f"{count}\n{pct:.1f}%",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold"
            )
    bottom += values

ax.set_title("Sentiment Distribution: Human vs DRSE Ensemble")
ax.set_ylabel("Proportion")
ax.set_ylim(0, 1)
ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()

FIG_PATH = "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/Result/Version 2 results/base_sentiment_distribution.png"
os.makedirs(os.path.dirname(FIG_PATH), exist_ok=True)
plt.savefig(FIG_PATH, dpi=200, bbox_inches="tight")
print(f"Figure saved to: {FIG_PATH}")
plt.show()

# ---------- Export distribution data to Excel ----------
dist_df_out = dist_df.reset_index().rename(columns={"index": "Source"})
dist_pct_out = (dist_pct * 100).reset_index().rename(columns={"index": "Source"})
DIST_OUTPUT_PATH = "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/Result/Version 2 results/base_sentiment_distribution.xlsx"
with pd.ExcelWriter(DIST_OUTPUT_PATH, engine="openpyxl") as writer:
    dist_df_out.to_excel(writer, sheet_name="Counts", index=False)
    dist_pct_out.to_excel(writer, sheet_name="Percentages", index=False)
print(f"Distribution data saved to: {DIST_OUTPUT_PATH}")