# KERAS WORD TOKENIZER + RNN

import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load Data
# -------------------------
df = pd.read_excel(
    "original data\TweeterManglishDS - filtered.xlsx"
)[['comment/tweet','majority_sent']].dropna()

df.columns = ['text','label']

le = LabelEncoder()
df['y'] = le.fit_transform(df['label'])
X = df['text']
y = df['y']

num_classes = len(le.classes_)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)



# -------------------------
# Keras Tokenizer
# -------------------------
tk = Tokenizer(num_words=20000, oov_token="<OOV>")
tk.fit_on_texts(X_train)

X_train = pad_sequences(tk.texts_to_sequences(X_train), maxlen=100)
X_test  = pad_sequences(tk.texts_to_sequences(X_test),  maxlen=100)

# -------------------------
# Dataset
# -------------------------
class DS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)  # ✅ FIX

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(DS(X_train, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(DS(X_test, y_test), batch_size=32)

# -------------------------
# RNN Model
# -------------------------
class TextRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(20000, 128)
        self.rnn = nn.RNN(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.emb(x)
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

model = TextRNN()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# -------------------------
# Train
# -------------------------
for _ in range(5):
    for x,y in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()


from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support
)


# -------------------------
# Evaluate
# -------------------------
preds, trues = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        preds.extend(logits.argmax(1).numpy())
        trues.extend(y.numpy())

# Accuracy
accuracy = accuracy_score(trues, preds)

# Precision, Recall, F1 (macro, micro, weighted)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    trues, preds, average="macro", zero_division=0
)

precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    trues, preds, average="micro", zero_division=0
)

precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    trues, preds, average="weighted", zero_division=0
)

print(f"Accuracy        : {accuracy:.4f}")
print(f"Precision (Macro): {precision_macro:.4f}")
print(f"Recall (Macro)   : {recall_macro:.4f}")
print(f"F1 (Macro)       : {f1_macro:.4f}")
print(f"F1 (Micro)       : {f1_micro:.4f}")
print(f"F1 (Weighted)    : {f1_weighted:.4f}")

# ===============================================================
# PREPARE SENTIMENT DISTRIBUTIONS
# ===============================================================

# --- Human annotation (test set only) ---
y_true = y_test.values
human_counts = np.bincount(y_true, minlength=num_classes)

# --- Model prediction ---
model_counts = np.bincount(preds, minlength=num_classes)

# Convert to percentages
human_pct = human_counts / human_counts.sum() * 100
model_pct = model_counts / model_counts.sum() * 100

sentiments = le.classes_  # e.g. ['negative','positive','neutral']

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
ax.set_xticklabels(["Human Annotation", "RNN model Prediction"])
ax.set_ylabel("Percentage of Instances (%)")
ax.set_ylim(0, 100)

ax.set_title("Sentiment Distribution: Human vs RNN Model")

plt.tight_layout()
plt.show()

# -------------------------
# Save Results
# -------------------------
metrics_df = pd.DataFrame({
    "Model": ["Keras Word Tokenizer + RNN"],
    "Accuracy": [accuracy],
    "Precision-Macro": [precision_macro],
    "Recall-Macro": [recall_macro],
    "F1-Macro": [f1_macro],
    "F1-Micro": [f1_micro],
    "F1-Weighted": [f1_weighted]
})

with pd.ExcelWriter("Phase 3/Sentiment distribution/Keras Word Token/keras_word_rnn.xlsx", engine="xlsxwriter") as writer:
    metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
    dist_df.to_excel(writer, sheet_name="Sentiment_Distribution", index=False)


