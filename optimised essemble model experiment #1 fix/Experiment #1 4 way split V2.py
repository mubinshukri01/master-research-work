# ===============================================================
# Dual-Representation Sentiment Ensemble (DRSE)
# Unified Experimental Framework
# Optimized Ensemble Model: CNN-BERT + GRU-Keras
# Model: NLP-GRU + CNN-BERT
# Split: 60(train) / 20(validation) / 10(calibration) / 10(test)
# ===============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from torch.utils.data import DataLoader, Dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
)

import warnings
# Narrowed from a blanket ignore — a full suppression would have hidden
# exactly the kind of subtle issues found throughout this pipeline review
# (e.g. tensor-construction warnings, deprecation notices pointing at
# real bugs). Only muting the known-noisy HuggingFace/transformers ones.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*resume_download.*")

# ---------------------------------------------------------------
# Device
# ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================================================
# 1. LOAD DATASET — from the CLEANED canonical file, not the raw Excel
# ===============================================================

df = pd.read_csv("redo/canonical_dataset.csv")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
label_names = list(le.classes_)

# ===============================================================
# 2. LOAD THE CANONICAL 4-WAY SPLIT (60/20/10/10)
# ===============================================================
# IMPORTANT: these indices must come from the SAME splits.npz used to
# pretrain gru_best.pt / cnn_bert_best.pt. Regenerating the split here —
# even with identical logic and random_state — would silently produce a
# DIFFERENT partition if run against different underlying data (e.g. the
# raw, uncleaned Excel), reintroducing the exact leakage this pipeline
# was fixed to remove.

splits = np.load("redo/splits.npz")
train_idx = splits["train"]
val_idx = splits["val"]
cal_idx = splits["cal"]
test_idx = splits["test"]

y_train = y[train_idx]
y_val = y[val_idx]
y_cal = y[cal_idx]
y_test = y[test_idx]

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Calibration:", len(cal_idx))
print("Test:", len(test_idx))

# Sanity check — should always pass, since split_generator.py already
# verified this when splits.npz was created. Kept here as a cheap guard
# in case the file is ever regenerated or swapped for the wrong one.
all_splits = {"train": train_idx, "val": val_idx, "cal": cal_idx, "test": test_idx}
keys = list(all_splits.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        overlap = len(set(all_splits[keys[i]]) & set(all_splits[keys[j]]))
        assert overlap == 0, f"LEAKAGE: {keys[i]} and {keys[j]} overlap by {overlap} indices"
print("Overlap check passed — no leakage between partitions.")
# ===============================================================
# 3. MODEL DEFINITIONS
# ===============================================================

class GRU_Model(nn.Module):
    def __init__(self, vocab_size=20000, embed_dim=128, hidden=128, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


class CNN_BERT(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.conv = nn.Conv1d(768, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(2)
        return self.fc(x)

# ===============================================================
# 4. LOAD TRAINED MODELS
# ===============================================================

gru_model = GRU_Model(num_classes=NUM_CLASSES).to(device)
gru_model.load_state_dict(torch.load("base essemble model experiment #1 (BERT + CNN & NLP + GRU)\\optimized essemble model experiment\\epoch GRU keras model\\epoch Version 2\\gru_best.pt", map_location=device))
gru_model.eval()

cnn_model = CNN_BERT(num_classes=NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load("base essemble model experiment #1 (BERT + CNN & NLP + GRU)\\optimized essemble model experiment\\epoch CNN BERT model\\epoch Version 2\\cnn_bert_best.pt", map_location=device))
cnn_model.eval()

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# ===============================================================
# 5. DATASET CLASS
# ===============================================================

class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = bert_tokenizer(
            self.texts[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            self.labels[idx]
        )

# BERT Data
X_val = [texts[i] for i in val_idx]
X_cal = [texts[i] for i in cal_idx]
X_test = [texts[i] for i in test_idx]

val_loader = DataLoader(BertDataset(X_val, y_val), batch_size=32)
cal_loader = DataLoader(BertDataset(X_cal, y_cal), batch_size=32)
test_loader = DataLoader(BertDataset(X_test, y_test), batch_size=32)

# ===============================================================
# GRU sequences — refit the Keras tokenizer on canonical train text
# (identical to Train_GRU.py) instead of loading a precomputed tensor
# of unknown provenance. This guarantees the word->index mapping and
# row alignment are both correct, with no external file to keep in sync.
# ===============================================================
MAX_LEN = 50
VOCAB = 20000

X_train_texts = [texts[i] for i in train_idx]

keras_tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
keras_tok.fit_on_texts(X_train_texts)

def to_keras_tensor(text_list):
    seqs = keras_tok.texts_to_sequences(text_list)
    seqs = pad_sequences(seqs, maxlen=MAX_LEN)
    return torch.tensor(seqs, dtype=torch.long)

X_val_keras = to_keras_tensor(X_val)
X_cal_keras = to_keras_tensor(X_cal)
X_test_keras = to_keras_tensor(X_test)

# ===============================================================
# 6. TEMPERATURE SCALING
# ===============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return logits / torch.exp(self.log_temp)

def fit_temperature(logits, labels):
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS([scaler.log_temp], lr=0.01, max_iter=50)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

def collect_logits_cnn(model, loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, yb in loader:
            ids, mask = ids.to(device), mask.to(device)
            logits.append(model(ids, mask))
            labels.append(yb.to(device))
    return torch.cat(logits), torch.cat(labels)

def collect_logits_gru(model, X, y):
    logits, labels = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = X[i:i+32].to(device)
            logits.append(model(xb))
            labels.append(torch.tensor(y[i:i+32]).to(device))
    return torch.cat(logits), torch.cat(labels)

# Fit temperatures using calibration set
cnn_cal_logits, cnn_cal_labels = collect_logits_cnn(cnn_model, cal_loader)
gru_cal_logits, gru_cal_labels = collect_logits_gru(gru_model, X_cal_keras, y_cal)

T_cnn = fit_temperature(cnn_cal_logits, cnn_cal_labels)
T_gru = fit_temperature(gru_cal_logits, gru_cal_labels)

print("CNN Temperature:", torch.exp(T_cnn.log_temp).item())
print("GRU Temperature:", torch.exp(T_gru.log_temp).item())

# ===============================================================
# 7. ALPHA GRID SEARCH (CALIBRATION SET)
# ===============================================================
# This must run on the Calibration partition, not Validation. Validation
# is already spent on epoch-count tuning in Train_GRU.py/Train_CNN_BERT.py;
# the whole point of the four-way split is that ensemble-weight learning
# happens on a SEPARATE dedicated calibration set, decoupled from that
# validation signal, to avoid overfitting the fusion weights to it.

def evaluate_alpha(alpha):
    preds = []

    with torch.no_grad():
        for (ids, mask, yb), xg in zip(
            cal_loader,
            torch.split(X_cal_keras, 32)
        ):
            ids, mask = ids.to(device), mask.to(device)
            xg = xg.to(device)

            probs_cnn = torch.softmax(
                T_cnn(cnn_model(ids, mask)), dim=1
            )
            probs_gru = torch.softmax(
                T_gru(gru_model(xg)), dim=1
            )

            P = alpha * probs_cnn + (1 - alpha) * probs_gru
            preds.extend(torch.argmax(P, dim=1).cpu().numpy())

    return f1_score(y_cal, preds, average="macro")

alphas = np.linspace(0, 1, 21)
scores = [evaluate_alpha(a) for a in alphas]
best_alpha = alphas[np.argmax(scores)]

print("Best Alpha:", best_alpha)
# Save alpha grid-search results to Excel
df_alphas = pd.DataFrame({
    "alpha": alphas.tolist() if isinstance(alphas, np.ndarray) else alphas,
    "f1_macro": scores
})
df_alphas["is_best"] = np.isclose(df_alphas["alpha"], best_alpha)
ALPHA_RESULTS_PATH = "optimised essemble model experiment #1 fix/result 4 way V2/alpha_grid_search_results.xlsx"
os.makedirs(os.path.dirname(ALPHA_RESULTS_PATH), exist_ok=True)
df_alphas.to_excel(ALPHA_RESULTS_PATH, index=False)
print(f"Saved alpha grid search results to: {ALPHA_RESULTS_PATH}")
# ===============================================================
# 8. FINAL TEST EVALUATION
# ===============================================================

final_preds = []

with torch.no_grad():
    for (ids, mask, yb), xg in zip(
        test_loader,
        torch.split(X_test_keras, 32)
    ):
        ids, mask = ids.to(device), mask.to(device)
        xg = xg.to(device)

        probs_cnn = torch.softmax(
            T_cnn(cnn_model(ids, mask)), dim=1
        )
        probs_gru = torch.softmax(
            T_gru(gru_model(xg)), dim=1
        )

        P = best_alpha * probs_cnn + (1 - best_alpha) * probs_gru
        final_preds.extend(torch.argmax(P, dim=1).cpu().numpy())

acc = accuracy_score(y_test, final_preds)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, final_preds, average="macro", zero_division=0
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_test, final_preds, average="micro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, final_preds, average="weighted", zero_division=0
)

print("\n===== FINAL TEST RESULTS =====")
print("Accuracy:", acc)
print("Macro Precision:", prec_macro)
print("Macro Recall:", rec_macro)
print("Macro F1:", f1_macro)

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
    "F1 (Weighted)": f1_weighted,
    "Best Alpha": best_alpha,
    "CNN Temperature": torch.exp(T_cnn.log_temp).item(),
    "GRU Temperature": torch.exp(T_gru.log_temp).item(),
}])

# ---------- Per-class Precision & Recall (matches base script) ----------
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
    y_test, final_preds, labels=range(NUM_CLASSES), zero_division=0
)
per_class_df = pd.DataFrame({
    "Sentiment": label_names,
    "Precision": prec_cls,
    "Recall": rec_cls,
    "F1": f1_cls,
    "Support": support
})

# ---------- Full classification report ----------
report_dict = classification_report(
    y_test, final_preds, target_names=label_names, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "Metric/Class"})

# ===============================================================
# 9. SENTIMENT DISTRIBUTION (TEST SET)
# ===============================================================

# Convert predictions back to label names
pred_labels = le.inverse_transform(final_preds)
true_labels = le.inverse_transform(y_test)

# Create DataFrame for analysis
results_df = pd.DataFrame({
    "True_Label": true_labels,
    "Predicted_Label": pred_labels
})

# ------------------------------
# Predicted Sentiment Counts
# ------------------------------
pred_counts = results_df["Predicted_Label"].value_counts().sort_index()
pred_percent = (pred_counts / len(results_df)) * 100

sentiment_summary = pd.DataFrame({
    "Count": pred_counts,
    "Percentage (%)": pred_percent.round(2)
})

print("\n===== PREDICTED SENTIMENT DISTRIBUTION =====")
print(sentiment_summary)

# ------------------------------
# True Sentiment Counts (Gold Standard)
# ------------------------------
true_counts = results_df["True_Label"].value_counts().sort_index()
true_percent = (true_counts / len(results_df)) * 100

true_summary = pd.DataFrame({
    "Count": true_counts,
    "Percentage (%)": true_percent.round(2)
})

print("\n===== TRUE SENTIMENT DISTRIBUTION =====")
print(true_summary)

# ===============================================================
# 10. SAVE ALL RESULTS TO ONE EXCEL FILE
# ===============================================================
RESULTS_PATH = "optimised essemble model experiment #1 fix/result 4 way V2/Optimized_Ensemble_Results.xlsx"
os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
with pd.ExcelWriter(RESULTS_PATH, engine="openpyxl") as writer:
    overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    per_class_df.to_excel(writer, sheet_name="Per-Class PRF", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)
    sentiment_summary.to_excel(writer, sheet_name="Predicted Distribution")
    true_summary.to_excel(writer, sheet_name="True Distribution")

print(f"\nAll results saved to: {RESULTS_PATH}")