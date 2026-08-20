# ===============================================================
# Dual-Representation Sentiment Ensemble (DRSE)
# Unified Experimental Framework
# Optimized Ensemble Model: CNN-BERT + GRU-Keras
# Model: NLP-GRU + CNN-BERT
# Split: 60(train) / 30(validation) / 10(test)
# ===============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# Device
# ---------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================================================
# 1. LOAD DATASET
# ===============================================================

df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
label_names = list(le.classes_)

# ===============================================================
# 2. 3-WAY STRATIFIED SPLIT (60/30/10)
# ===============================================================

indices = np.arange(len(texts))

train_idx, temp_idx, y_train, y_temp = train_test_split(
    indices,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

val_idx, test_idx, y_val, y_test = train_test_split(
    temp_idx,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print("Train:", len(train_idx))
print("Validation:", len(val_idx))
print("Test:", len(test_idx))

print("Train-Test overlap:",
      len(set(train_idx) & set(test_idx)))

print("Validation-Test overlap:",
      len(set(val_idx) & set(test_idx)))
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
gru_model.load_state_dict(torch.load(
    "Final experiment/base essemble model experiment #1 (BERT + CNN & NLP + GRU)\optimized essemble model experiment\epoch GRU keras model\epoch_5\gru_keras5.pt"
))
gru_model.eval()

cnn_model = CNN_BERT(num_classes=NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load(
    "Final experiment/base essemble model experiment #1 (BERT + CNN & NLP + GRU)\optimized essemble model experiment\epoch CNN BERT model\epoch_5\cnn_bert5.pt"
))
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
X_test = [texts[i] for i in test_idx]

val_loader = DataLoader(BertDataset(X_val, y_val), batch_size=32)
test_loader = DataLoader(BertDataset(X_test, y_test), batch_size=32)

# GRU sequences
X_all_keras = torch.load(
    "Final experiment/base essemble model experiment #1 (BERT + CNN & NLP + GRU)\optimized essemble model experiment\origin\keras_sequences.pt"
)

X_val_keras = X_all_keras[val_idx]
X_test_keras = X_all_keras[test_idx]

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
cnn_val_logits, cnn_val_labels = collect_logits_cnn(
    cnn_model,
    val_loader
)

gru_val_logits, gru_val_labels = collect_logits_gru(
    gru_model,
    X_val_keras,
    y_val
)

T_cnn = fit_temperature(cnn_val_logits, cnn_val_labels)
T_gru = fit_temperature(gru_val_logits, gru_val_labels)

print("CNN Temperature:", torch.exp(T_cnn.log_temp).item())
print("GRU Temperature:", torch.exp(T_gru.log_temp).item())

# ===============================================================
# 7. ALPHA GRID SEARCH (VALIDATION SET)
# ===============================================================

def evaluate_alpha(alpha):
    preds = []

    with torch.no_grad():
        for (ids, mask, yb), xg in zip(
            val_loader,
            torch.split(X_val_keras, 32)
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

    return f1_score(y_val, preds, average="macro")

alphas = np.linspace(0, 1, 21)
scores = [evaluate_alpha(a) for a in alphas]
best_alpha = alphas[np.argmax(scores)]

print("Best Alpha:", best_alpha)

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
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, final_preds, average="macro"
)

print("\n===== FINAL TEST RESULTS =====")
print("Accuracy:", acc)
print("Macro Precision:", prec)
print("Macro Recall:", rec)
print("Macro F1:", f1)

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
# 10. EXPORT RESULTS TO EXCEL
# ===============================================================

export_filename = "Final experiment/optimised essemble model experiment #1 fix/ablation study/ablation_study_3way_vs_4way_split_epoch5.xlsx"

summary_df = pd.DataFrame({
    "Metric": ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
    "Value": [acc, prec, rec, f1]
})

meta_df = pd.DataFrame({
    "Study": ["Ablation Study 3-way split vs 4-way split"],
    "Description": ["Compare 60/30/10 split results against 4-way split results"]
})

with pd.ExcelWriter(export_filename, engine="openpyxl") as writer:
    meta_df.to_excel(writer, index=False, sheet_name="Metadata")
    summary_df.to_excel(writer, index=False, sheet_name="Metrics")
    results_df.to_excel(writer, index=False, sheet_name="Test_Results")
    sentiment_summary.to_excel(writer, sheet_name="Predicted_Distribution")
    true_summary.to_excel(writer, sheet_name="True_Distribution")

print(f"\nResults exported to {export_filename}")