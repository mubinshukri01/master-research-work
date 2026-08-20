# =======================
# SET 1: Keras Tokenizer
# =======================

import os, random, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoModelForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_PATH = "original data/TweeterManglishDS - filtered.xlsx"
df = pd.read_excel(DATA_PATH)[['comment/tweet','majority_sent']].dropna()
df.columns = ['text','label']

le = LabelEncoder()
df['y'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['y'], stratify=df['y'], test_size=0.2, random_state=SEED
)

# -------- Keras Tokenizer --------
tk = Tokenizer(num_words=1000, oov_token="<OOV>")
tk.fit_on_texts(X_train)

X_train_seq = pad_sequences(tk.texts_to_sequences(X_train), maxlen=100)
X_test_seq  = pad_sequences(tk.texts_to_sequences(X_test), maxlen=100)

# -------- Dataset --------
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(TextDataset(X_train_seq, y_train), batch_size=32, shuffle=True)
test_loader  = DataLoader(TextDataset(X_test_seq, y_test), batch_size=32)

# -------- Models --------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv = nn.Conv1d(
            in_channels=128,
            out_channels=128,
            kernel_size=5
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)          # (B, L, E)
        x = x.transpose(1, 2)          # (B, E, L) ✅ required
        x = torch.relu(self.conv(x))   # (B, 128, L')
        x = self.pool(x).squeeze(-1)   # (B, 128)
        return self.fc(x)


class RNNBase(nn.Module):
    def __init__(self, cell_type, vocab_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)

        if cell_type == "LSTM":
            self.rnn = nn.LSTM(128, 128, batch_first=True)
            self.is_lstm = True
        elif cell_type == "GRU":
            self.rnn = nn.GRU(128, 128, batch_first=True)
            self.is_lstm = False
        else:  # RNN
            self.rnn = nn.RNN(128, 128, batch_first=True)
            self.is_lstm = False

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)

        if self.is_lstm:
            _, (h_n, _) = self.rnn(x)   # h_n: (1, B, H)
        else:
            _, h_n = self.rnn(x)        # h_n: (1, B, H)

        h_n = h_n.squeeze(0)            # (B, H) ✅ tensor now
        return self.fc(h_n)



models = {
    "CNN": TextCNN(vocab_size=1000, num_classes=num_classes),
    "LSTM": RNNBase("LSTM", vocab_size=1000, num_classes=num_classes),
    "GRU":  RNNBase("GRU",  vocab_size=1000, num_classes=num_classes),
    "RNN":  RNNBase("RNN",  vocab_size=1000, num_classes=num_classes),
    "mBERT": AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=num_classes
    )
}



results = []
sentiment_dist = {}

for name, model in models.items():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(3):
        for x,y in train_loader:
            optimizer.zero_grad()
            out = model(x) if name!="mBERT" else model(input_ids=x).logits
            loss = loss_fn(out,y)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x,y in test_loader:
            out = model(x) if name!="mBERT" else model(input_ids=x).logits
            p = out.argmax(1)
            preds.extend(p.numpy())
            trues.extend(y.numpy())

    sentiment_dist[name] = np.bincount(preds, minlength=num_classes)
    results.append([
        name,
        accuracy_score(trues,preds),
        f1_score(trues,preds,average="micro"),
        f1_score(trues,preds,average="macro"),
        f1_score(trues,preds,average="weighted")
    ])

# df_res = pd.DataFrame(results, columns=["Model","Acc","F1-Micro","F1-Macro","F1-Weighted"])
# df_res.to_excel("Phase 3\result\sentiment distributionresultSET1_KerasTokenizer.xlsx", index=False)

# pd.DataFrame(sentiment_dist, index=le.classes_).plot(kind="bar")
# plt.title("SET 1 – Sentiment Distribution (Keras Tokenizer)")
# plt.show()
