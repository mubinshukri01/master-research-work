# ===============================================================
# CNN + BERT Tokenizer
# Output: cnn_bert_model.pt
# ===============================================================

import os
import json
import shutil
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
# Load dataset — from the CLEANED canonical file, not the raw Excel
# -----------------------
df = pd.read_csv("redo\canonical_dataset.csv")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

# -----------------------
# Load canonical split (replaces the old train_test_split — this is the fix)
# -----------------------
splits = np.load("redo\splits.npz")
train_idx = splits["train"]
val_idx = splits["val"]

X_train = [texts[i] for i in train_idx]
y_train = y[train_idx]
X_val = [texts[i] for i in val_idx]
y_val = y[val_idx]

# ===============================================================
# Dataset
# ===============================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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

# # inspect one batch from train_loader
# batch = next(iter(train_loader))
# ids, masks, labels = batch  # ids: (B, L), masks: (B, L)

# print("input_ids shape:", ids.shape)
# print("attention_mask shape:", masks.shape)

# # first sample (as lists)
# print("input_ids (first sample):", ids[0].tolist())
# print("attention_mask (first sample):", masks[0].tolist())

# # decode only the non-padded tokens
# nonpad_ids = [int(i) for i, m in zip(ids[0].tolist(), masks[0].tolist()) if m == 1]
# tokens = tokenizer.convert_ids_to_tokens(nonpad_ids)
# print("tokens (first sample):", tokens)

# # optionally: print first N samples
# N = 3
# for i in range(min(N, ids.size(0))):
#     ids_i = ids[i].tolist()
#     mask_i = masks[i].tolist()
#     tokens_i = tokenizer.convert_ids_to_tokens([int(x) for x, m in zip(ids_i, mask_i) if m == 1])
#     print(f"sample {i} tokens:", tokens_i)

# ===============================================================
# One sample text for demonstration
# ===============================================================
example_idx = 1
example_text = X_train[example_idx]
example_label = y_train[example_idx]
print("Example text:", example_text)
print("Example label:", example_label)

example_enc = tokenizer(
    example_text,
    padding="max_length",
    truncation=True,
    max_length=50,
    return_tensors="pt"
)

input_ids = example_enc["input_ids"]
attention_mask = example_enc["attention_mask"]

print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)

tokens = tokenizer.convert_ids_to_tokens(
    [int(tok) for tok, m in zip(input_ids[0].tolist(), attention_mask[0].tolist()) if m == 1]
)
print("tokens:", tokens)

# ===============================================================
# CNN + BERT Model
# ===============================================================

class CNN_BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
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


model.eval()
with torch.no_grad():
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    bert_output = model.bert(input_ids, attention_mask=attention_mask)
    last_hidden = bert_output.last_hidden_state

print("last_hidden_state shape:", last_hidden.shape)
# first token ([CLS]) embedding
print("CLS embedding shape:", last_hidden[:, 0, :].shape)
print("CLS embedding sample values:", last_hidden[0, 0, :8].cpu().numpy())

with torch.no_grad():
    logits = model(input_ids, attention_mask)

print("logits shape:", logits.shape)
print("logits:", logits.cpu().numpy())

probs = torch.softmax(logits, dim=-1)
print("probabilities:", probs.cpu().numpy())
print("predicted class:", logits.argmax(dim=-1).item())

# ===============================================================
# CNN + BERT Training for Epoch Comparison (5–10)
# ===============================================================

EPOCH_LIST = [5, 6, 7, 8, 9, 10]

# Track all losses for export
all_losses = []

for EPOCHS in EPOCH_LIST:

    print(f"\nTraining CNN-BERT for {EPOCHS} epochs")

    # ---- reinitialize model ----
    model = CNN_BERT().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
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
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Validation ----
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

        all_losses.append({
            "Total_Epochs": EPOCHS,
            "Epoch": epoch + 1,
            "Train_Loss": avg_loss,
            "Validation_Loss": avg_val_loss
        })

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"- Train Loss: {avg_loss:.4f} "
            f"- Val Loss: {avg_val_loss:.4f}"
        )

    # ---- Save model ----
    save_dir = (
        f"base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
        f"optimized essemble model experiment/epoch CNN BERT model/epoch Version 2/epoch_{EPOCHS}"
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{save_dir}/cnn_bert{EPOCHS}.pt")
    print(f"Saved: {save_dir}/cnn_bert{EPOCHS}.pt")

# ===============================================================
# Export losses
# ===============================================================

losses_df = pd.DataFrame(all_losses)
OUTPUT_PATH = (
    "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
    "optimized essemble model experiment/epoch CNN BERT model/epoch Version 2/CNN_BERT_Training_Losses.xlsx"
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    losses_df.to_excel(writer, sheet_name="Training Losses", index=False)

print(f"\nAll training losses exported to: {OUTPUT_PATH}")
print(f"Total epochs tracked: {len(all_losses)}")

# ===============================================================
# Select best checkpoint and save it under a fixed, predictable name
# ===============================================================
# Same logic as Train_GRU.py: only the FINAL epoch of each EPOCHS run was
# actually checkpointed, so compare using each run's final validation loss.
# The canonical filename below is what permanently fixes the old
# cnn_bert.pt / cnn_bert10.pt mismatch — the ensemble script only ever
# needs to know one name: cnn_bert_best.pt.
final_rows = losses_df[losses_df["Epoch"] == losses_df["Total_Epochs"]]
best_row = final_rows.loc[final_rows["Validation_Loss"].idxmin()]
best_epochs = int(best_row["Total_Epochs"])
best_val_loss = float(best_row["Validation_Loss"])

best_checkpoint_path = (
    f"base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
    f"optimized essemble model experiment/epoch CNN BERT model/epoch Version 2/"
    f"epoch_{best_epochs}/cnn_bert{best_epochs}.pt"
)

CANONICAL_BEST_PATH = "cnn_bert_best.pt"
shutil.copy(best_checkpoint_path, CANONICAL_BEST_PATH)

manifest = {
    "best_epochs": best_epochs,
    "validation_loss": best_val_loss,
    "source_checkpoint": best_checkpoint_path,
    "canonical_checkpoint": CANONICAL_BEST_PATH,
}
with open("cnn_bert_best_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nBest epoch count: {best_epochs} (Validation_Loss = {best_val_loss:.4f})")
print(f"Copied best checkpoint to: {CANONICAL_BEST_PATH}")
print(f"Manifest written to: cnn_bert_best_manifest.json")