# =============================================================== 
# GRU + Keras Tokenizer (Corrected: Split → Tokenize)
# Output: gru_keras_model.pt
# ===============================================================

import os
import json
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
df = pd.read_csv("redo/canonical_dataset.csv")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

# Encode labels
le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

# -----------------------
# Load canonical split (replaces the old train_test_split — this is the fix)
# -----------------------
splits = np.load("redo/splits.npz")
train_idx = splits["train"]

X_train_text = [texts[i] for i in train_idx]
y_train = y[train_idx]

val_idx = splits["val"]
X_val_text = [texts[i] for i in val_idx]
y_val = y[val_idx]
# -----------------------
# Keras Tokenizer (FIT ONLY ON TRAINING)
# -----------------------
VOCAB_SIZE = 20000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)   # ✅ only training data

# Convert texts to sequences
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)

# Padding
X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_val = pad_sequences(X_val, maxlen=MAX_LEN)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ===============================================================
# GRU Model
# ===============================================================

class GRU_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 128)
        self.gru = nn.GRU(128, 128, batch_first=True)
        self.fc = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# ===============================================================
# Sample
# ===============================================================

sample_idx = 0
sample_text = X_train_text[sample_idx]
print("Sample text:", sample_text)

sample_seq = tokenizer.texts_to_sequences([sample_text])
sample_pad = pad_sequences(sample_seq, maxlen=MAX_LEN)
print("Token IDs:", sample_pad[0])

sample_tokens = [
    tokenizer.index_word.get(token_id, "<PAD/OOV>")
    for token_id in sample_pad[0]
]
print("Tokens:", sample_tokens)

sample_tensor = torch.tensor(sample_pad, dtype=torch.long).to(device)
model = GRU_Model().to(device)

model.eval()
with torch.no_grad():
    sample_logits = model(sample_tensor)
    sample_probs = torch.softmax(sample_logits, dim=-1)

print("Logits:", sample_logits.cpu().numpy())
print("Probabilities:", sample_probs.cpu().numpy())
pred_label = torch.argmax(sample_logits, dim=-1).item()
print("Predicted class index:", pred_label)
print("Predicted label:", le.inverse_transform([pred_label])[0])

with torch.no_grad():
    sample_embedding = model.embedding(sample_tensor)  # shape [1, MAX_LEN, 128]

for token_id, token_str, embed_vec in zip(sample_pad[0], sample_tokens, sample_embedding[0]):
    print(token_id, token_str, embed_vec[:8].cpu().numpy())

xb, yb = next(iter(val_loader))
xb, yb = xb.to(device), yb.to(device)

with torch.no_grad():
    logits = model(xb[:1])
print("First batch token IDs:", xb[0].cpu().numpy())
print("First batch logits:", logits.cpu().numpy())
# ===============================================================
# Training (Epoch comparison 5–10)
# ===============================================================

EPOCH_LIST = [5, 6, 7, 8, 9, 10]
all_losses = []

for EPOCHS in EPOCH_LIST:

    print(f"\nTraining GRU for {EPOCHS} epochs")

    model = GRU_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        # ---- Training ----
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # # get embedding matrix
        # embeddings = model.embedding.weight.detach().cpu().numpy()
        # print("embeddings shape:", embeddings.shape)  # e.g. (20000, 128)

        # # inspect some indices (Keras Tokenizer reserves index 0 for padding)
        # print("embedding[0] (padding):", embeddings[0][:8])
        # print("embedding[1]:", embeddings[1][:8])

        # # look up a word -> index -> vector
        # word = "covid"
        # idx = tokenizer.word_index.get(word)            # word -> index (1-based)
        # if idx is None or idx >= VOCAB_SIZE:
        #     print(f"{word} not in tokenizer top {VOCAB_SIZE}")
        # else:
        #     print(f"{word} idx={idx}, vector[:8]={embeddings[idx][:8]}")

        # # save embeddings
        # np.save("embeddings.npy", embeddings)
        # pd.DataFrame(embeddings).to_csv("embeddings.csv", index=False)

        # ---- Validation ----
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
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
    save_dir = f"base essemble model experiment #1 (BERT + CNN & NLP + GRU)/optimized essemble model experiment/epoch GRU keras model/epoch Version 2/epoch_{EPOCHS}"
    os.makedirs(save_dir, exist_ok=True)

    torch.save(model.state_dict(), f"{save_dir}/gru_keras{EPOCHS}.pt")
    print(f"Saved: {save_dir}/gru_keras{EPOCHS}.pt")

# ===============================================================
# Export losses
# ===============================================================

losses_df = pd.DataFrame(all_losses)
OUTPUT_PATH = "base essemble model experiment #1 (BERT + CNN & NLP + GRU)/optimized essemble model experiment/epoch GRU keras model/epoch Version 2/GRU_Training_Losses2.xlsx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    losses_df.to_excel(writer, sheet_name="Training Losses", index=False)

print(f"\nAll training losses exported to: {OUTPUT_PATH}")
print(f"Total epochs tracked: {len(all_losses)}")

# ===============================================================
# Select best checkpoint and save it under a fixed, predictable name
# ===============================================================
# Only ONE checkpoint was actually saved per EPOCHS value (at the end of
# that run's training loop), so "best" must compare each run's FINAL
# epoch validation loss — not the minimum loss seen mid-run, since no
# checkpoint exists for those intermediate epochs.
final_rows = losses_df[losses_df["Epoch"] == losses_df["Total_Epochs"]]
best_row = final_rows.loc[final_rows["Validation_Loss"].idxmin()]
best_epochs = int(best_row["Total_Epochs"])
best_val_loss = float(best_row["Validation_Loss"])

best_checkpoint_path = (
    f"base essemble model experiment #1 (BERT + CNN & NLP + GRU)/"
    f"optimized essemble model experiment/epoch GRU keras model/epoch Version 2/"
    f"epoch_{best_epochs}/gru_keras{best_epochs}.pt"
)

CANONICAL_BEST_PATH = "gru_best.pt"
shutil.copy(best_checkpoint_path, CANONICAL_BEST_PATH)

# Small manifest recording the decision — useful as methodology evidence,
# and lets the ensemble script confirm it loaded the intended checkpoint.
manifest = {
    "best_epochs": best_epochs,
    "validation_loss": best_val_loss,
    "source_checkpoint": best_checkpoint_path,
    "canonical_checkpoint": CANONICAL_BEST_PATH,
}
with open("gru_best_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nBest epoch count: {best_epochs} (Validation_Loss = {best_val_loss:.4f})")
print(f"Copied best checkpoint to: {CANONICAL_BEST_PATH}")
print(f"Manifest written to: gru_best_manifest.json")