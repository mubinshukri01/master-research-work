# ===============================================================
# BERT TOKENIZER + CNN (GPU ENABLED)
# ===============================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

from transformers import BertTokenizer

# ===============================================================
# Reproducibility
# ===============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ===============================================================
# Device
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ===============================================================
# Load Data
# ===============================================================
df = pd.read_excel(
    "original data/TweeterManglishDS - filtered.xlsx"
)[["comment/tweet", "majority_sent"]].dropna()

df.columns = ["text", "label"]

le = LabelEncoder()
df["y"] = le.fit_transform(df["label"])

X = df["text"]
y = df["y"]
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# ===============================================================
# BERT Tokenizer
# ===============================================================
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

X_train = tokenizer(
    X_train.tolist(),
    padding="max_length",
    truncation=True,
    max_length=100,
    return_tensors="pt"
)["input_ids"]

X_test = tokenizer(
    X_test.tolist(),
    padding="max_length",
    truncation=True,
    max_length=100,
    return_tensors="pt"
)["input_ids"]

# ===============================================================
# Dataset
# ===============================================================
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.long()
        self.y = torch.tensor(y.values, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(
    TextDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    TextDataset(X_test, y_test),
    batch_size=32
)

# ===============================================================
# CNN Model
# ===============================================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(128, 128, kernel_size=5)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.emb(x)              # (B, T, E)
        x = x.transpose(1, 2)        # (B, E, T)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

model = TextCNN(tokenizer.vocab_size, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# ===============================================================
# Train
# ===============================================================
EPOCHS = 3
model.train()

for epoch in range(EPOCHS):
    for x, y_batch in train_loader:
        x = x.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(x), y_batch)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

# ===============================================================
# Evaluate
# ===============================================================
model.eval()
preds, trues = [], []

with torch.no_grad():
    for x, y_batch in test_loader:
        x = x.to(device)
        y_batch = y_batch.to(device)

        logits = model(x)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(y_batch.cpu().numpy())

accuracy = accuracy_score(trues, preds)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    trues, preds, average="macro", zero_division=0
)
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    trues, preds, average="micro", zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    trues, preds, average="weighted", zero_division=0
)

print("\nEvaluation Results")
print(f"Accuracy         : {accuracy:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro)   : {recall_macro:.4f}")
print(f"F1 (Macro)       : {f1_macro:.4f}")
print(f"F1 (Micro)       : {f1_micro:.4f}")
print(f"F1 (Weighted)    : {f1_weighted:.4f}")

# ===============================================================
# Sentiment Distribution
# ===============================================================
human_counts = np.bincount(y_test.values, minlength=num_classes)
model_counts = np.bincount(preds, minlength=num_classes)

human_pct = human_counts / human_counts.sum() * 100
model_pct = model_counts / model_counts.sum() * 100

sentiments = le.classes_

dist_df = pd.DataFrame({
    "Sentiment": sentiments,
    "Human_Count": human_counts,
    "Human_Percentage": human_pct,
    "Model_Count": model_counts,
    "Model_Percentage": model_pct
})

# ===============================================================
# STACKED BAR CHART (STYLE MATCHES REFERENCE)
# ===============================================================

fig, ax = plt.subplots(figsize=(7, 6))

x = np.arange(2)  # Human vs Model
bar_width = 0.6

colors = {
    "positive": "green",
    "negative": "red",
    "neutral": "gray"
}

bottom_human = 0
bottom_model = 0

for i, sent in enumerate(sentiments):
    # Human
    ax.bar(
        x[0],
        human_pct[i],
        bottom=bottom_human,
        color=colors[sent],
        width=bar_width
    )
    ax.text(
        x[0],
        bottom_human + human_pct[i] / 2,
        f"{human_pct[i]:.1f}%\n({human_counts[i]})",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    # Model
    ax.bar(
        x[1],
        model_pct[i],
        bottom=bottom_model,
        color=colors[sent],
        width=bar_width
    )
    ax.text(
        x[1],
        bottom_model + model_pct[i] / 2,
        f"{model_pct[i]:.1f}%\n({model_counts[i]})",
        ha="center",
        va="center",
        color="white",
        fontsize=10,
        fontweight="bold"
    )

    bottom_human += human_pct[i]
    bottom_model += model_pct[i]

human_pct = human_counts / human_counts.sum() * 100
model_pct = model_counts / model_counts.sum() * 100
sentiments = le.classes_

# -------------------------
# Sentiment Distribution Table
# -------------------------
dist_df = pd.DataFrame({
    "Sentiment": sentiments,
    "Human_Count": human_counts,
    "Human_Percentage": human_pct,
    "Model_Count": model_counts,
    "Model_Percentage": model_pct
})


# ===============================================================
# AXIS & LABELS
# ===============================================================

ax.set_xticks(x)
ax.set_xticklabels(["Human Annotation", "CNN model Prediction"])
ax.set_ylabel("Percentage of Instances (%)")
ax.set_ylim(0, 100)

ax.set_title("Sentiment Distribution: Human vs CNN Model")

plt.tight_layout()
plt.show()


# # ===============================================================
# # Save Results
# # ===============================================================
# metrics_df = pd.DataFrame({
#     "Model": ["BERT Tokenizer + CNN"],
#     "Accuracy": [accuracy],
#     "Precision-Macro": [precision_macro],
#     "Recall-Macro": [recall_macro],
#     "F1-Macro": [f1_macro],
#     "F1-Micro": [f1_micro],
#     "F1-Weighted": [f1_weighted]
# })

# output_file = "Phase 3/Sentiment distribution/BERT/BERT_tokenizer_cnn.xlsx"

# with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
#     metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
#     dist_df.to_excel(writer, sheet_name="Sentiment_Distribution", index=False)

# print("\nResults saved to:", output_file)
