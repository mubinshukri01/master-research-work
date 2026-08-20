# ===============================================================
# Optimized CNN + BERT Sentiment Classifier
# With Regularization, Scheduler, and Early Stopping
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

# ------------------------------------------------
# Reproducibility
# ------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ------------------------------------------------
# Load dataset
# ------------------------------------------------

df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)

NUM_CLASSES = len(le.classes_)

# ------------------------------------------------
# Train / Validation split
# ------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# Tokenizer
# ------------------------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# ------------------------------------------------
# Dataset
# ------------------------------------------------

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
    BertDataset(X_train, y_train),
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    BertDataset(X_val, y_val),
    batch_size=8
)

# ===============================================================
# CNN + BERT Model
# ===============================================================

class CNN_BERT(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = BertModel.from_pretrained("bert-base-cased")

        # Freeze most BERT layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Fine-tune last 2 transformer layers
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.conv = nn.Conv1d(768, 128, kernel_size=3)

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, ids, mask):

        outputs = self.bert(ids, attention_mask=mask)

        x = outputs.last_hidden_state

        x = x.permute(0, 2, 1)

        x = torch.relu(self.conv(x))

        x = self.pool(x).squeeze(2)

        x = self.dropout(x)

        return self.fc(x)

# ------------------------------------------------
# Initialize model
# ------------------------------------------------

model = CNN_BERT().to(device)

# ------------------------------------------------
# Optimizer + Loss
# ------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss()

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=1,
    factor=0.5
)

# ===============================================================
# Training Settings
# ===============================================================

EPOCHS = 10

best_val_loss = float("inf")

patience = 2
early_stop_counter = 0

all_losses = []

# ===============================================================
# Training Loop
# ===============================================================

for epoch in range(EPOCHS):

    # ------------------------
    # Training
    # ------------------------

    model.train()

    total_loss = 0

    for ids, mask, yb in train_loader:

        ids = ids.to(device)
        mask = mask.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(ids, mask)

        loss = criterion(logits, yb)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # ------------------------
    # Validation
    # ------------------------

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for ids, mask, yb in val_loader:

            ids = ids.to(device)
            mask = mask.to(device)
            yb = yb.to(device)

            logits = model(ids, mask)

            loss = criterion(logits, yb)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    all_losses.append({
        "Epoch": epoch + 1,
        "Train_Loss": avg_train_loss,
        "Validation_Loss": avg_val_loss
    })

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"- Train Loss: {avg_train_loss:.4f} "
        f"- Val Loss: {avg_val_loss:.4f}"
    )

    # # ------------------------
    # # Early stopping
    # # ------------------------

    # if avg_val_loss < best_val_loss:

    #     best_val_loss = avg_val_loss
    #     early_stop_counter = 0

    #     torch.save(model.state_dict(), "essemble model experiment #1\optimized essemble model experiment\epoch CNN BERT model\ best_cnn_bert_model_epoch10.pt")

    # else:

    #     early_stop_counter += 1

    #     if early_stop_counter >= patience:
    #         print("Early stopping triggered")
    #         break

# # ===============================================================
# # Export Losses
# # ===============================================================

# losses_df = pd.DataFrame(all_losses)

# OUTPUT_PATH = "CNN_BERT_Training_Losses.xlsx"

# losses_df.to_excel(OUTPUT_PATH, index=False)

# print("\nTraining losses exported to:", OUTPUT_PATH)

# ===============================================================
# Save final model
# ===============================================================

torch.save(model.state_dict(), "essemble model experiment #1\optimized essemble model experiment\epoch CNN BERT model\ best_cnn_bert_model_epoch10.pt")

print("Model saved as best_cnn_bert_model_epoch10")
print("Best Validation Loss:", best_val_loss)