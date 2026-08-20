import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================================================
# 1. LOAD DATASET
# ===============================================================

df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)
label_names = list(le.classes_)

# ===============================================================
# 2. TRAIN / CALIBRATION / TEST SPLIT (60 / 20 / 20)
# ===============================================================

indices = np.arange(len(texts))

train_temp_idx, test_idx, y_train_temp, y_test = train_test_split(
    indices, y, test_size=0.2, random_state=42, stratify=y
)

train_idx, calib_idx, y_train, y_calib = train_test_split(
    train_temp_idx, y_train_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_train_temp
)

X_calib = [texts[i] for i in calib_idx]
X_test = [texts[i] for i in test_idx]

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
gru_model.load_state_dict(torch.load("essemble model experiment #1\optimized essemble model experiment\epoch GRU keras model\epoch_5\gru_keras5.pt"))
gru_model.eval()

cnn_model = CNN_BERT(num_classes=NUM_CLASSES).to(device)
cnn_model.load_state_dict(torch.load("essemble model experiment #1\optimized essemble model experiment\epoch CNN BERT model\epoch_5\cnn_bert5.pt"))
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

X_all_keras = torch.load("essemble model experiment #1\optimized essemble model experiment\origin\keras_sequences.pt")
X_calib_keras = X_all_keras[calib_idx]
X_test_keras = X_all_keras[test_idx]

calib_loader = DataLoader(BertDataset(X_calib, y_calib), batch_size=32)
test_loader = DataLoader(BertDataset(X_test, y_test), batch_size=32)

# ===============================================================
# 6. SAFE TEMPERATURE SCALING
# ===============================================================

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        temperature = torch.exp(self.log_temp)
        return logits / temperature


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

# ===============================================================
# 7. COLLECT LOGITS
# ===============================================================

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

# ===============================================================
# 8. CALIBRATION
# ===============================================================

cnn_logits_cal, cnn_labels_cal = collect_logits_cnn(cnn_model, calib_loader)
gru_logits_cal, gru_labels_cal = collect_logits_gru(gru_model, X_calib_keras, y_calib)

T_cnn = fit_temperature(cnn_logits_cal, cnn_labels_cal)
T_gru = fit_temperature(gru_logits_cal, gru_labels_cal)

print("CNN Temperature:", torch.exp(T_cnn.log_temp).item())
print("GRU Temperature:", torch.exp(T_gru.log_temp).item())

# ===============================================================
# 9. MANUAL ALPHA
# ===============================================================

ALPHA = 0.85  # Adjust this value to see how it affects the final results

# ===============================================================
# 10. FINAL TEST EVALUATION
# ===============================================================

cnn_logits_test, _ = collect_logits_cnn(cnn_model, test_loader)
gru_logits_test, _ = collect_logits_gru(gru_model, X_test_keras, y_test)

probs_cnn_test = torch.softmax(T_cnn(cnn_logits_test), dim=1)
probs_gru_test = torch.softmax(T_gru(gru_logits_test), dim=1)

P_test = ALPHA * probs_cnn_test + (1 - ALPHA) * probs_gru_test
final_preds = torch.argmax(P_test, dim=1).cpu().numpy()

acc = accuracy_score(y_test, final_preds)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, final_preds, average="macro"
)

print("\n===== FINAL TEST RESULTS =====")
print("Alpha:", ALPHA)
print("Accuracy:", acc)
print("Macro Precision:", prec)
print("Macro Recall:", rec)
print("Macro F1:", f1)

# ===============================================================
# 11. SENTIMENT DISTRIBUTION ANALYSIS (TEST SET)
# ===============================================================

human_counts = pd.Series(y_test).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
drse_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

count_df = pd.DataFrame({
    "Sentiment": label_names,
    "Human Count": human_counts.values,
    "DRSE Count": drse_counts.values
})

percentage_df = pd.DataFrame({
    "Sentiment": label_names,
    "Human (%)": (human_counts / human_counts.sum() * 100).values,
    "DRSE (%)": (drse_counts / drse_counts.sum() * 100).values
})

print("\nSentiment Distribution (Counts)")
print(count_df)

print("\nSentiment Distribution (%)")
print(percentage_df)

# ----- Stacked Proportion Bar Plot with % and Counts -----

prop_df = pd.DataFrame({
    "Human": human_counts.values,
    "DRSE": drse_counts.values
}, index=label_names).T

prop_df = prop_df.div(prop_df.sum(axis=1), axis=0)

fig, ax = plt.subplots(figsize=(9,6))
prop_df.plot(kind="bar", stacked=True, ax=ax, edgecolor="white", linewidth=1.2)

plt.title("Sentiment Distribution: Human vs DRSE (Test Set)")
plt.ylabel("Proportion")
plt.ylim(0,1)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend(title="Sentiment", bbox_to_anchor=(1.02,1), loc="upper left")

# ----------------------------------------------------------
# Add Percentage + Count Labels
# ----------------------------------------------------------

for i, sentiment in enumerate(prop_df.columns):
    for j, source in enumerate(prop_df.index):

        pct = prop_df.loc[source, sentiment]
        cnt = count_df.loc[
            count_df["Sentiment"] == sentiment,
            "Human Count" if source == "Human" else "DRSE Count"
        ].values[0]

        bar = ax.containers[i][j]

        x = bar.get_x() + bar.get_width()/2
        y = bar.get_y() + bar.get_height()/2

        if pct > 0:  # avoid cluttering zero values
            ax.text(
                x, y,
                f"{pct*100:.1f}%\n({cnt})",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold"
            )

plt.tight_layout()
plt.show()

# # ===============================================================
# # 12. EXPORT RESULTS
# # ===============================================================

# OUTPUT_FILE = "essemble model experiment #1 fix/DRSE_ManualAlpha1.0_TestResults.xlsx"

# with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
#     pd.DataFrame([{
#         "Alpha": ALPHA,
#         "Accuracy": acc,
#         "Precision (Macro)": prec,
#         "Recall (Macro)": rec,
#         "F1 (Macro)": f1
#     }]).to_excel(writer, sheet_name="Overall Metrics", index=False)

#     count_df.to_excel(writer, sheet_name="Sentiment Counts", index=False)
#     percentage_df.to_excel(writer, sheet_name="Sentiment Percentages", index=False)

# print(f"\nResults exported to {OUTPUT_FILE}")
