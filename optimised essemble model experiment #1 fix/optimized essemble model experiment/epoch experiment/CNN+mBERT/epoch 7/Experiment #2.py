# ===============================================================
# Dual-Representation Sentiment Ensemble (DRSE)
# With Proper Probability Calibration
# Experiment #2: Calibrated Ensemble on Epoch 7 Models
# ===============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================
# Load dataset
# ===============================================================

df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

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
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.conv = nn.Conv1d(768, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(2)
        return self.fc(x)

# ===============================================================
# Train / Validation split (needed for calibration)
# ===============================================================

indices = np.arange(len(texts))

train_idx, val_idx, _, _ = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

X_val = [texts[i] for i in val_idx]
y_val = y[val_idx]


# ===============================================================
# Placeholder: load your trained models here
# ===============================================================

gru_model = GRU_Model().to(device)
gru_model.load_state_dict(torch.load("essemble model experiment #1\optimized essemble model experiment\epoch GRU keras model\epoch_7\gru_keras7.pt"))
gru_model.eval()

cnn_model = CNN_BERT().to(device)
cnn_model.load_state_dict(torch.load("essemble model experiment #1\optimized essemble model experiment\epoch CNN BERT model (mBERT)\epoch_7\cnn_bert7.pt"))
cnn_model.eval()

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# ===============================================================
# Validation Dataset
# ===============================================================

class BertValDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = bert_tokenizer(self.texts[idx], return_tensors="pt",
                             padding="max_length", truncation=True, max_length=50)
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), self.labels[idx]

bert_val_loader = DataLoader(BertValDataset(X_val, y_val), batch_size=32)

# Keras-tokenized sequences already saved
X_all_keras = torch.load("essemble model experiment #1\optimized essemble model experiment\origin\keras_sequences.pt")
X_val_keras = X_all_keras[val_idx]

print(X_val_keras.max(), X_val_keras.min())


# ===============================================================
# Temperature Scaling Module
# ===============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

# ===============================================================
# Collect logits for calibration
# ===============================================================

def collect_logits_cnn(model, loader):
    logits, labels = [], []
    with torch.no_grad():
        for ids, mask, yb in loader:
            ids, mask = ids.to(device), mask.to(device)
            logit = model(ids, mask)
            logits.append(logit)
            labels.append(yb.to(device))
    return torch.cat(logits), torch.cat(labels)

def collect_logits_gru(model, X, y):
    logits, labels = [], []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            xb = X[i:i+32].to(device)
            logit = model(xb)
            logits.append(logit)
            labels.append(torch.tensor(y[i:i+32]).to(device))
    return torch.cat(logits), torch.cat(labels)

# ===============================================================
# Fit temperature
# ===============================================================

def fit_temperature(logits, labels):
    scaler = TemperatureScaler().to(device)
    optimizer = optim.LBFGS([scaler.temperature], lr=0.01, max_iter=50)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler

# ===============================================================
# Learn temperatures
# ===============================================================

cnn_logits, cnn_labels = collect_logits_cnn(cnn_model, bert_val_loader)
gru_logits, gru_labels = collect_logits_gru(gru_model, X_val_keras, y_val)

T_cnn = fit_temperature(cnn_logits, cnn_labels)
T_gru = fit_temperature(gru_logits, gru_labels)

print("CNN temperature:", T_cnn.temperature.item())
print("GRU temperature:", T_gru.temperature.item())

assert cnn_logits.shape[0] == cnn_labels.shape[0]
assert gru_logits.shape[0] == gru_labels.shape[0]

# ===============================================================
# Calibrated Ensemble Prediction
# ===============================================================

alpha = 0.6
final_preds = []

for i, idx in enumerate(val_idx):
    # CNN
    enc = bert_tokenizer(texts[idx], return_tensors="pt",
                         padding="max_length", truncation=True, max_length=50)
    ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    probs_cnn = torch.softmax(T_cnn(cnn_model(ids, mask)), dim=1)

    # GRU
    x = X_val_keras[i].unsqueeze(0).to(device)
    probs_gru = torch.softmax(T_gru(gru_model(x)), dim=1)

    P = alpha * probs_cnn + (1 - alpha) * probs_gru
    final_preds.append(torch.argmax(P).item())


# ===============================================================
# Evaluation
# ===============================================================

print("Calibrated Ensemble Accuracy:", accuracy_score(y_val, final_preds))
print("Calibrated Ensemble Macro-F1:", f1_score(y_val, final_preds, average="macro"))

label_names = list(le.classes_)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Human counts
human_counts = pd.Series(y_val).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
# Ensemble counts
ens_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

count_df = pd.DataFrame({
    "Human": human_counts.values,
    "DRSE": ens_counts.values
}, index=label_names).T

prop_df = count_df.div(count_df.sum(axis=1), axis=0)

# ---- Plot stacked bar ----
fig, ax = plt.subplots(figsize=(9,6))
prop_df.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=1.2)

plt.title("Sentiment Distribution: Human vs DRSE")
plt.ylabel("Proportion")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02,1), loc="upper left")

# Add counts + percentages
for i, sentiment in enumerate(prop_df.columns):
    for j, src in enumerate(prop_df.index):
        pct = prop_df.loc[src, sentiment]
        cnt = count_df.loc[src, sentiment]

        bar = ax.containers[i][j]
        x = bar.get_x() + bar.get_width()/2
        y0 = bar.get_y() + bar.get_height()/2

        ax.text(x, y0, f"{pct*100:.1f}%\n({cnt})",
                ha="center", va="center", color="white",
                fontsize=10, fontweight="bold")

plt.tight_layout()
plt.show()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

accuracy_score(y_val, final_preds)
acc = accuracy_score(y_val, final_preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, final_preds, average="macro")

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1 (Macro)"],
    "Score": [acc, prec, rec, f1]
})

fig, ax = plt.subplots(figsize=(8,5))

bars = ax.bar(metrics_df["Metric"], metrics_df["Score"])
ax.set_ylim(0,1)
ax.set_title("DRSE Performance Metrics")
ax.grid(axis="y", linestyle="--", alpha=0.6)

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x()+bar.get_width()/2, h, f"{h:.3f}",
            ha="center", va="bottom", fontsize=11)

plt.tight_layout()
plt.show()

import numpy as np

labels = ["Accuracy", "Precision", "Recall", "F1"]
values = [acc, prec, rec, f1]

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
values += values[:1]

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

ax.plot(angles, values, linewidth=2)
ax.fill(angles, values, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0,1)
ax.set_title("DRSE Performance Radar")

plt.tight_layout()
plt.show()

# ===============================================================
# EXPORT ALL RESULTS TO EXCEL
# ===============================================================

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report

# ----- Overall metrics -----
acc = accuracy_score(y_val, final_preds)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_val, final_preds, average="macro", zero_division=0
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_val, final_preds, average="micro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_val, final_preds, average="weighted", zero_division=0
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

# ----- Per-class metrics -----
label_names = list(le.classes_)
prec_c, rec_c, f1_c, support = precision_recall_fscore_support(
    y_val, final_preds, labels=range(NUM_CLASSES), zero_division=0
)

per_class_df = pd.DataFrame({
    "Sentiment": label_names,
    "Precision": prec_c,
    "Recall": rec_c,
    "F1": f1_c,
    "Support": support
})

# ----- Sentiment distribution -----
human_counts = pd.Series(y_val).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
ens_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

dist_counts = pd.DataFrame({
    "Sentiment": label_names,
    "Human Count": human_counts.values,
    "DRSE Count": ens_counts.values
})

dist_pct = pd.DataFrame({
    "Sentiment": label_names,
    "Human (%)": (human_counts / human_counts.sum() * 100).values,
    "DRSE (%)": (ens_counts / ens_counts.sum() * 100).values
})

# ----- Classification report -----
report = classification_report(
    y_val, final_preds, target_names=label_names, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "Metric/Class"})

# ----- Write to Excel -----
OUTPUT_FILE = "essemble model experiment #1\optimized essemble model experiment\epoch experiment\epoch 7\DRSE_Calibrated_Results_epoch7.xlsx"
with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    per_class_df.to_excel(writer, sheet_name="Per-Class Metrics", index=False)
    dist_counts.to_excel(writer, sheet_name="Sentiment Counts", index=False)
    dist_pct.to_excel(writer, sheet_name="Sentiment Percentages", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)

print(f"Results exported to {OUTPUT_FILE}")
