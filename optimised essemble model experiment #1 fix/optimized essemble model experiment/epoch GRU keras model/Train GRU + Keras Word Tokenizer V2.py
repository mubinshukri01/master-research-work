# ===============================================================
# Improved GRU + Keras Tokenizer Training Pipeline
# Regularized + Early Stopping + LR Scheduling
# ===============================================================

import os
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

# ===============================================================
# Load dataset
# ===============================================================

df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")

texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

# Label encoding
le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

# ===============================================================
# Train / Validation split FIRST
# ===============================================================

X_train_text, X_val_text, y_train, y_val = train_test_split(
    texts, y, test_size=0.15, random_state=42, stratify=y
)

# ===============================================================
# Tokenizer (fit only on training)
# ===============================================================

VOCAB_SIZE = 20000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)

X_train = pad_sequences(X_train, maxlen=MAX_LEN)
X_val = pad_sequences(X_val, maxlen=MAX_LEN)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# ===============================================================
# Improved GRU Model
# ===============================================================

class GRU_Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(VOCAB_SIZE, 64)

        self.gru = nn.GRU(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(64, NUM_CLASSES)

    def forward(self, x):

        x = self.embedding(x)

        _, h = self.gru(x)

        h = self.dropout(h.squeeze(0))

        return self.fc(h)

# ===============================================================
# Training Setup
# ===============================================================

EPOCHS = 6

model = GRU_Model().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-5
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=2,
    factor=0.5
)

# ===============================================================
# Early Stopping
# ===============================================================

best_val_loss = float("inf")
patience = 4
counter = 0

history = []

# ===============================================================
# Training Loop
# ===============================================================

for epoch in range(EPOCHS):

    # -----------------
    # Training
    # -----------------
    model.train()

    total_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        logits = model(xb)

        loss = criterion(logits, yb)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    # -----------------
    # Validation
    # -----------------
    model.eval()

    val_loss_total = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)

            loss = criterion(logits, yb)

            val_loss_total += loss.item()

    val_loss = val_loss_total / len(val_loader)

    scheduler.step(val_loss)

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # -----------------
    # Save best model
    # -----------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        counter = 0

        torch.save(model.state_dict(), "essemble model experiment #1\\optimized essemble model experiment\\epoch GRU keras model\\best_gru_model_epoch6.pt")

    else:

        counter += 1

        if counter >= patience:

            print("Early stopping triggered")
            break

# ===============================================================
# Export training history
# ===============================================================

history_df = pd.DataFrame(history)

OUTPUT_PATH = "essemble model experiment #1\\optimized essemble model experiment\\epoch GRU keras model\\GRU_training_history_epoch6.xlsx"

history_df.to_excel(OUTPUT_PATH, index=False)

print("Training history saved to:", OUTPUT_PATH)
print("Best validation loss:", best_val_loss)
print("model saved successfully" )