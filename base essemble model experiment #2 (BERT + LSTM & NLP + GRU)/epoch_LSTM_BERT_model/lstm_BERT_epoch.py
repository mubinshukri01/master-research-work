# ===============================================================
# LSTM + BERT Tokenizer
# Epoch Comparison Experiment (5–10)
# ===============================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

train_texts, val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)

# -----------------------
# Dataset
# -----------------------
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=50,
            return_tensors="pt"
        )
        return (
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

train_loader = DataLoader(
    BertDataset(train_texts, y_train),
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    BertDataset(val_texts, y_val),
    batch_size=16,
    shuffle=False
)

# -----------------------
# LSTM + BERT Model
# -----------------------
class LSTM_BERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, ids, mask):
        x = self.bert(
            input_ids=ids,
            attention_mask=mask
        ).last_hidden_state

        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

# ===============================================================
# Epoch Comparison Experiment
# ===============================================================
EPOCH_LIST = [5, 6, 7, 8, 9, 10]
all_losses = []

for TOTAL_EPOCHS in EPOCH_LIST:

    print(f"\nTraining LSTM-BERT for {TOTAL_EPOCHS} epochs")

    model = LSTM_BERT(NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(TOTAL_EPOCHS):
        model.train()
        total_loss = 0.0

        for ids, mask, labels in train_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ids, mask, labels in val_loader:
                ids = ids.to(device)
                mask = mask.to(device)
                labels = labels.to(device)

                logits = model(ids, mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        all_losses.append({
            "Total_Epochs": TOTAL_EPOCHS,
            "Epoch": epoch + 1,
            "Train_Loss": avg_loss,
            "Validation_Loss": avg_val_loss
        })

        print(
            f"Epoch {epoch + 1}/{TOTAL_EPOCHS} "
            f"- Train Loss: {avg_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f}"
        )

    # ---- Save checkpoint ----
    save_dir = (
        f"essemble model experiment #2/"
        f"epoch_LSTM_BERT/epoch_{TOTAL_EPOCHS}"
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        f"{save_dir}/lstm_bert.pt"
    )

    print(f"Saved: {save_dir}/lstm_bert.pt")

# ===============================================================
# Export Training Losses
# ===============================================================
loss_df = pd.DataFrame(all_losses)

OUTPUT_PATH = (
    "essemble model experiment #2\epoch_LSTM_BERT\LSTM_BERT_Training_Losses.xlsx"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

loss_df.to_excel(
    OUTPUT_PATH,
    sheet_name="Training Losses",
    index=False
)

print(f"\nTraining losses exported to: {OUTPUT_PATH}")
print(f"Total epochs tracked: {len(all_losses)}")
