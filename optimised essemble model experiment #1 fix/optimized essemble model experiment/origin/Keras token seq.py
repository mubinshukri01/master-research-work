# ===============================================================
# Create Keras Validation Sequences
# Output: keras_val_sequences.pt
# ===============================================================

import torch
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
np.random.seed(SEED)

# -----------------------
# Load dataset
# -----------------------
df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

# -----------------------
# Encode labels
# -----------------------
le = LabelEncoder()
y = le.fit_transform(labels)

# -----------------------
# Train / Validation split
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    texts, y, test_size=0.15, random_state=42, stratify=y
)

# -----------------------
# Keras tokenizer
# -----------------------
VOCAB_SIZE = 20000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)   # IMPORTANT: fit ONLY on training text

# -----------------------
# Convert validation texts
# -----------------------
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_val_seq = pad_sequences(X_val_seq, maxlen=MAX_LEN)

# -----------------------
# Save as PyTorch tensor
# -----------------------
torch.save(torch.tensor(X_val_seq, dtype=torch.long), "essemble model experiment/keras_val_sequences.pt")

print("essemble model experiment/keras_val_sequences.pt saved successfully")
