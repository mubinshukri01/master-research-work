# ================================================================
#  Dual-Representation Sentiment Ensemble (DRSE)
#  LSTM + BERT Tokenizer  ⊕  GRU + Keras Tokenizer
#  not standardize
# ================================================================

from unittest import loader
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader as TorchDataLoader
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
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = False   # IMPORTANT: allow randomness
cudnn.benchmark = True       # IMPORTANT: allow kernel variation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load dataset
# ---------------------------
df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
label_names = list(le.classes_)
NUM_CLASSES = len(label_names)


# ===============================================================
#  STREAM A: Keras Word Tokenizer + GRU
# ===============================================================

MAX_LEN = 50
VOCAB = 20000

keras_tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
keras_tok.fit_on_texts(texts)

X_keras = keras_tok.texts_to_sequences(texts)
X_keras = pad_sequences(X_keras, maxlen=MAX_LEN)
X_keras = torch.tensor(X_keras, dtype=torch.long)

# GRU model
class GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))  # raw logits


gru = GRU_Model().to(device)
opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train GRU
for epoch in range(5):
    for i in range(0, len(X_keras), 32):
        xb = X_keras[i:i+32].to(device)
        yb = torch.tensor(y[i:i+32]).to(device)

        opt_gru.zero_grad()
        pred = gru(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt_gru.step()

# ===============================================================
#  STREAM B (UPDATED): BERT Tokenizer + LSTM
# ===============================================================

bert_tok = BertTokenizer.from_pretrained("bert-base-cased")
bert = BertModel.from_pretrained("bert-base-cased")

for param in bert.parameters():
    param.requires_grad = False

class LSTM_BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, ids, mask):
        with torch.no_grad():  # BERT frozen
            x = self.bert(ids, attention_mask=mask).last_hidden_state

        _, (h_n, _) = self.lstm(x)
        h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        h = self.dropout(h)

        return self.fc(h)  # ❗ NO softmax here


lstm_bert = LSTM_BERT().to(device)
opt_lstm = torch.optim.AdamW(lstm_bert.parameters(), lr=2e-5)

class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = bert_tok(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            self.labels[idx]
        )


train_loader = TorchDataLoader(
    BertDataset(texts, y),
    batch_size=16,
    shuffle=True
)




# train LSTM + BERT
for epoch in range(3):
    for ids, mask, lab in train_loader:
        ids = ids.to(device)
        mask = mask.to(device)
        lab = torch.tensor(lab).to(device)

        opt_lstm.zero_grad()
        pred = lstm_bert(ids, mask)
        loss = loss_fn(pred, lab)
        loss.backward()
        opt_lstm.step()

# ---------------------------
# Switch to evaluation mode
# ---------------------------
gru.eval()
lstm_bert.eval()

# ===============================================================
#  ENSEMBLE INFERENCE
# ===============================================================

ALPHA = 0.6   # LSTM+BERT
BETA = 0.4    # GRU+Keras

final_preds = []
true = []

with torch.no_grad():
    for i in range(len(texts)):
        k = X_keras[i].unsqueeze(0).to(device)
        bert_in = bert_tok(texts[i], return_tensors="pt", padding="max_length", truncation=True, max_length=50)
        ids = bert_in["input_ids"].to(device)
        mask = bert_in["attention_mask"].to(device)

        p_gru = torch.softmax(gru(k), dim=1)
        p_lstm = torch.softmax(lstm_bert(ids, mask), dim=1)



        # Step 5: Weighted Weighted Soft-Voting Ensemble
        ensemble = ALPHA * p_lstm + BETA * p_gru
        # Step 6 Final sentiment decision
        final_preds.append(torch.argmax(ensemble, dim=1).item())
        true.append(y[i])

# ===============================================================
#  Step 7 : Evaluation
# ===============================================================

print("Ensemble Accuracy:", accuracy_score(true, final_preds))
print("Ensemble Macro-F1:", f1_score(true, final_preds, average="macro"))

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
OUTPUT_PATH = "essemble model experiment #2/Freeze Bert/DRSE_Ensemble_Results5.xlsx"
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
plt.show()

# Optional: export to Excel
dist_df_out = dist_df.reset_index().rename(columns={"index": "Source"})
dist_pct_out = (dist_pct * 100).reset_index().rename(columns={"index": "Source"})
with pd.ExcelWriter("ensemble_sentiment_distribution.xlsx", engine="openpyxl") as writer:
    dist_df_out.to_excel(writer, sheet_name="Counts", index=False)
    dist_pct_out.to_excel(writer, sheet_name="Percentages", index=False)
