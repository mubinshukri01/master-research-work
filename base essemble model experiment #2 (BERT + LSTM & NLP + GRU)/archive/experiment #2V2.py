# ================================================================
#  Dual-Representation Sentiment Ensemble (DRSE)
#  GRU + Keras Tokenizer  ⊕  LSTM + BERT Tokenizer
#  With Proper Probability Calibration (3-way 60/20/20 split)
#  suggested by AI
# ================================================================

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter
import matplotlib.pyplot as plt

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

# ================================================================
# Load Dataset
# ================================================================
df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
label_names = list(le.classes_)
NUM_CLASSES = len(label_names)

# ---------------------------
# 80 / 20 Split
# ---------------------------
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y, test_size=0.4, stratify=y, random_state=SEED
)

# ---------------------------
# Split test into calibration / final evaluation (50 / 50)
# ---------------------------
X_calib_texts, X_eval_texts, y_calib, y_eval = train_test_split(
    X_test_texts,
    y_test,
    test_size=0.5,
    stratify=y_test,
    random_state=SEED
)

# ================================================================
# STREAM A: Keras Tokenizer + GRU
# ================================================================
MAX_LEN = 50
VOCAB = 20000

keras_tok = Tokenizer(num_words=VOCAB, oov_token="<OOV>")
keras_tok.fit_on_texts(X_train_texts)

def keras_encode(texts):
    seq = keras_tok.texts_to_sequences(texts)
    seq = pad_sequences(seq, maxlen=MAX_LEN)
    return torch.tensor(seq, dtype=torch.long)

X_keras_train = keras_encode(X_train_texts)
X_keras_calib = keras_encode(X_calib_texts)
X_keras_eval  = keras_encode(X_eval_texts)

class GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.embed(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))  # logits

gru = GRU_Model().to(device)
opt_gru = torch.optim.Adam(gru.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train GRU
gru.train()
for epoch in range(5):
    for i in range(0, len(X_keras_train), 32):
        xb = X_keras_train[i:i+32].to(device)
        yb = torch.tensor(y_train[i:i+32]).to(device)

        opt_gru.zero_grad()
        loss = loss_fn(gru(xb), yb)
        loss.backward()
        opt_gru.step()

# ================================================================
# STREAM B: BERT Tokenizer + LSTM (Frozen BERT)
# ================================================================
bert_tok = BertTokenizer.from_pretrained("bert-base-cased")
bert = BertModel.from_pretrained("bert-base-cased")

for p in bert.parameters():
    p.requires_grad = False

class LSTM_BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, ids, mask):
        with torch.no_grad():
            x = self.bert(ids, attention_mask=mask).last_hidden_state
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)  # logits

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

train_loader = DataLoader(
    BertDataset(X_train_texts, y_train),
    batch_size=32,
    shuffle=True
)

# Train LSTM+BERT
lstm_bert.train()
for epoch in range(5):
    for ids, mask, lab in train_loader:
        ids, mask = ids.to(device), mask.to(device)
        lab = torch.tensor(lab).to(device)

        opt_lstm.zero_grad()
        loss = loss_fn(lstm_bert(ids, mask), lab)
        loss.backward()
        opt_lstm.step()

gru.eval()
lstm_bert.eval()

# ================================================================
# Temperature Scaling (Calibration)
# ================================================================
class TemperatureScaler(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1, device=device))

    def forward(self, logits):
        return logits / self.temperature


def calibrate_temperature(logits, labels, device):
    scaler = TemperatureScaler(device).to(device)
    optimizer = torch.optim.LBFGS([scaler.temperature], lr=0.01)
    loss_fn = nn.CrossEntropyLoss().to(device)

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler



# Collect calibration logits
gru_logits = []

with torch.no_grad():
    for i in range(len(X_keras_calib)):
        xb = X_keras_calib[i].unsqueeze(0).to(device)
        gru_logits.append(gru(xb))

    gru_logits = torch.cat(gru_logits).to(device)
    y_calib_tensor = torch.tensor(y_calib, device=device)


    lstm_logits = []

with torch.no_grad():
    for i in range(len(X_calib_texts)):
        enc = bert_tok(
            X_calib_texts[i],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        lstm_logits.append(lstm_bert(ids, mask))

lstm_logits = torch.cat(lstm_logits).to(device)



gru_scaler = calibrate_temperature(
    gru_logits, y_calib_tensor, device
)

lstm_scaler = calibrate_temperature(
    lstm_logits, y_calib_tensor, device
)



# ================================================================
# Final Calibrated Ensemble Evaluation
# ================================================================
ALPHA = 0.6
BETA  = 0.4

final_preds = []

with torch.no_grad():
    for i in range(len(X_eval_texts)):
        k = X_keras_eval[i].unsqueeze(0).to(device)

        enc = bert_tok(
            X_eval_texts[i],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=50
        )
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        p_gru = torch.softmax(
            gru_scaler(gru(k)), dim=1
        )

        p_lstm = torch.softmax(
            lstm_scaler(lstm_bert(ids, mask)), dim=1
        )

        ensemble = ALPHA * p_lstm + BETA * p_gru
        final_preds.append(torch.argmax(ensemble, dim=1).item())


# ================================================================
# Evaluation
# ================================================================
print("Accuracy:", accuracy_score(y_eval, final_preds))
print("Macro-F1:", f1_score(y_eval, final_preds, average="macro"))
print("Prediction Distribution:", Counter(final_preds))

print(classification_report(y_eval, final_preds, target_names=label_names))

# ================================================================
# Export Results to Excel
# ================================================================

# Overall metrics
acc = accuracy_score(y_eval, final_preds)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_eval, final_preds, average="macro", zero_division=0
)
prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
    y_eval, final_preds, average="micro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_eval, final_preds, average="weighted", zero_division=0
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

# Per-class metrics
prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
    y_eval, final_preds, labels=range(NUM_CLASSES), zero_division=0
)

per_class_df = pd.DataFrame({
    "Sentiment": label_names,
    "Precision": prec_cls,
    "Recall": rec_cls,
    "F1": f1_cls,
    "Support": support
})

# Sentiment distribution (Counts & %)
human_counts = pd.Series(y_eval).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
pred_counts = pd.Series(final_preds).value_counts().reindex(range(NUM_CLASSES), fill_value=0)

dist_counts = pd.DataFrame({
    "Sentiment": label_names,
    "Human Count": human_counts.values,
    "Prediction Count": pred_counts.values
})

dist_pct = pd.DataFrame({
    "Sentiment": label_names,
    "Human (%)": (human_counts / human_counts.sum() * 100).values,
    "Prediction (%)": (pred_counts / pred_counts.sum() * 100).values
})

# Classification report
report_dict = classification_report(
    y_eval, final_preds, target_names=label_names, output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "Metric/Class"})

# Write to Excel
OUTPUT_PATH = "essemble model experiment #2/result/DRSE_Calibrated_Results#2V2-2.xlsx"
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    overall_df.to_excel(writer, sheet_name="Overall Metrics", index=False)
    per_class_df.to_excel(writer, sheet_name="Per-Class PRF", index=False)
    dist_counts.to_excel(writer, sheet_name="Sentiment Counts", index=False)
    dist_pct.to_excel(writer, sheet_name="Sentiment Percentages", index=False)
    report_df.to_excel(writer, sheet_name="Classification Report", index=False)

print(f"\nAll results saved to: {OUTPUT_PATH}")
