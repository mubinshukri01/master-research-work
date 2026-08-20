# ===============================================================
# CNN + BERT Tokenizer
# Output: cnn_bert_model.pt
# ===============================================================

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

# -----------------------
# Train / Validation split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    texts, y, test_size=0.15, random_state=42, stratify=y
)

# ===============================================================
# Dataset
# ===============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

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
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), self.labels[idx]

train_loader = DataLoader(BertDataset(X_train, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(BertDataset(X_val, y_val), batch_size=16)

# ===============================================================
# CNN + BERT Model
# ===============================================================

class CNN_BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.conv = nn.Conv1d(768, 128, kernel_size=3)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        x = x.permute(0,2,1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(2)
        return self.fc(x)

model = CNN_BERT().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ===============================================================
# Training
# ===============================================================

EPOCHS = 9
all_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for ids, mask, yb in train_loader:
        ids, mask, yb = ids.to(device), mask.to(device), torch.tensor(yb).to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for ids, mask, yb in val_loader:
            ids = ids.to(device)
            mask = mask.to(device)
            yb = torch.tensor(yb).to(device)

            logits = model(ids, mask)
            loss = criterion(logits, yb)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    all_losses.append({
        "Epoch": epoch + 1,
        "Train_Loss": avg_train_loss,
        "Validation_Loss": avg_val_loss
    })

    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# ===============================================================
# Export losses to Excel
# ===============================================================

losses_df = pd.DataFrame(all_losses)
OUTPUT_PATH = "essemble model experiment #1/optimized essemble model experiment/epoch CNN BERT model (mBERT)/epoch_9/CNN_BERT_Training_Losses9.xlsx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

losses_df.to_excel(OUTPUT_PATH, sheet_name="Training Losses", index=False)
print(f"\nTraining and validation losses exported to: {OUTPUT_PATH}")

# ===============================================================
# Save model
# ===============================================================

torch.save(model.state_dict(), "essemble model experiment #1/optimized essemble model experiment/epoch CNN BERT model (mBERT)/epoch_9/cnn_bert9.pt")
print("Model saved as essemble model experiment #1/optimized essemble model experiment/epoch CNN BERT model (mBERT)/epoch_9/cnn_bert9.pt")
