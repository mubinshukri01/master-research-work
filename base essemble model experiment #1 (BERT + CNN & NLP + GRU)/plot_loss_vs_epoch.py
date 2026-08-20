import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load training & validation loss data
# -------------------------------------------------
file_path = "Final experiment\\base essemble model experiment #2 (BERT + LSTM & NLP + GRU)\\epoch_LSTM_BERT_model\\LSTM_BERT_Training_Losses.xlsx"
df = pd.read_excel(file_path)

# -------------------------------------------------
# Identify required columns
# -------------------------------------------------
epoch_col = None
train_loss_col = None
val_loss_col = None

for col in df.columns:
    col_lower = col.lower()
    if "epoch" in col_lower:
        epoch_col = col
    elif "train" in col_lower and "loss" in col_lower:
        train_loss_col = col
    elif "val" in col_lower and "loss" in col_lower:
        val_loss_col = col

if epoch_col is None or train_loss_col is None or val_loss_col is None:
    raise ValueError("Required columns (Epoch, Training Loss, Validation Loss) not found.")

# -------------------------------------------------
# Plot Training & Validation Loss
# -------------------------------------------------
plt.figure(figsize=(8,5))

plt.plot(
    df[epoch_col],
    df[train_loss_col],
    marker='o',
    markersize=6,
    linewidth=2,
    color='tab:blue',
    label='Training Loss'
)

plt.plot(
    df[epoch_col],
    df[val_loss_col],
    marker='o',
    markersize=6,
    linewidth=2,
    color='tab:orange',
    label='Validation Loss'
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM-BERT Model Loss over Epochs ")

plt.legend(loc="upper left")
plt.grid(True, linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()