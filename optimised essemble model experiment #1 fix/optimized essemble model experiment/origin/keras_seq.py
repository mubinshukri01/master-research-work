# ===============================================================
# Keras Word Tokenizer → keras_sequences.pt
# ===============================================================

import torch
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# -----------------------
# Load dataset
# -----------------------
df = pd.read_excel("original data\TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

# -----------------------
# Keras Tokenizer
# -----------------------
VOCAB_SIZE = 20000
MAX_LEN = 50

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Convert to sequences
sequences = tokenizer.texts_to_sequences(texts)
sequences = pad_sequences(sequences, maxlen=MAX_LEN)

# Convert to torch tensor
sequences_tensor = torch.tensor(sequences, dtype=torch.long)

# Save
torch.save(sequences_tensor, "essemble model experiment/keras_sequences.pt")
print("Saved essemble model experiment/keras_sequences.pt")
