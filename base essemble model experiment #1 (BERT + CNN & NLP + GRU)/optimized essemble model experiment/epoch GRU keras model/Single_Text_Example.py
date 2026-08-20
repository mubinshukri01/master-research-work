# =============================================================== 
# GRU + Keras Tokenizer - Single Text Example
# Demonstrates the pipeline with one example text
# ===============================================================

import torch
import torch.nn as nn
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------
# Reproducibility
# -----------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# ===============================================================
# Setup Parameters
# ===============================================================

VOCAB_SIZE = 20000
MAX_LEN = 50
NUM_CLASSES = 3  # assuming 3 sentiment classes

# ===============================================================
# Single Example Text
# ===============================================================

# Example: a single text to process
example_text = "OK LA tu selamat la kiranya, saya sangkut lagi sini, office, business, tanah, rumah sendiri dan lain lain semua sini, no way out"
print(f"\n--- Input Text ---")
print(f"Text: '{example_text}'")

# ===============================================================
# Step 1: Create and fit Tokenizer (on training corpus)
# ===============================================================

# In practice, tokenizer is fit on training data
# For this demo, we'll fit it on a sample corpus
training_corpus = [
    "This is a great product and I really enjoyed using it",
    "I did not like this product at all",
    "The quality is amazing and delivery was fast",
    "Terrible experience and waste of money",
    "Best purchase I have made this year",
    "Not worth the price"
]

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(training_corpus)

print(f"\n--- Tokenizer Info ---")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Total unique words in training corpus: {len(tokenizer.word_index)}")

# ===============================================================
# Step 2: Convert text to sequence
# ===============================================================

# Tokenize the example text
sequence = tokenizer.texts_to_sequences([example_text])[0]
print(f"\n--- Tokenization ---")
print(f"Sequence (indices): {sequence}")

# Show word-to-index mapping for this example
words = example_text.split()
print(f"\nWord-to-Index mapping:")
for word in words:
    idx = tokenizer.word_index.get(word.lower())
    if idx:
        print(f"  '{word}' → {idx}")

# ===============================================================
# Step 3: Pad sequence
# ===============================================================

padded = pad_sequences([sequence], maxlen=MAX_LEN)[0]
print(f"\n--- Padding ---")
print(f"Original sequence length: {len(sequence)}")
print(f"Padded sequence length: {len(padded)}")
print(f"Padded sequence: {padded}")

# Convert to tensor
X_example = torch.tensor([padded], dtype=torch.long).to(device)
print(f"\nTensor shape: {X_example.shape}")

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
        # Embedding layer: (batch, seq_len) → (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        print(f"  After embedding: {embedded.shape}")
        
        # GRU layer: (batch, seq_len, embedding_dim) → (batch, seq_len, hidden_dim)
        gru_out, hidden = self.gru(embedded)
        print(f"  After GRU: {gru_out.shape}")
        print(f"  GRU hidden state: {hidden.shape}")
        
        # Take hidden state: (1, batch, hidden_dim) → (batch, hidden_dim)
        h = hidden.squeeze(0)
        print(f"  After squeeze: {h.shape}")
        
        # Fully connected layer: (batch, hidden_dim) → (batch, num_classes)
        logits = self.fc(h)
        print(f"  After FC layer (logits): {logits.shape}")
        
        return logits

# ===============================================================
# Step 4: Forward pass through model
# ===============================================================

print(f"\n--- Model Forward Pass ---")
model = GRU_Model().to(device)
model.eval()

with torch.no_grad():
    logits = model(X_example)

print(f"\nRaw logits: {logits}")

# Convert logits to probabilities
probabilities = torch.softmax(logits, dim=1)
print(f"Probabilities: {probabilities}")

# Get prediction
predicted_class = torch.argmax(probabilities, dim=1).item()
print(f"Predicted class: {predicted_class}")
print(f"Confidence: {probabilities[0, predicted_class].item():.4f}")

# ===============================================================
# Step 5: Inspect embeddings
# ===============================================================

print(f"\n--- Embedding Layer Inspection ---")
embedding_matrix = model.embedding.weight.detach().cpu().numpy()
print(f"Embedding matrix shape: {embedding_matrix.shape}")
print(f"Embedding for padding (index 0): {embedding_matrix[0][:5]}")

# Get embeddings for words in the sequence
print(f"\nEmbeddings for words in sequence:")
for idx in sequence[:5]:  # Show first 5 words
    if idx > 0:
        word = [w for w, i in tokenizer.word_index.items() if i == idx]
        if word:
            print(f"  Word '{word[0]}' (idx {idx}): {embedding_matrix[idx][:5]} ...")

print(f"\n✓ Single text example completed successfully!")
