import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_excel("original data/TweeterManglishDS - filtered.xlsx")
texts = df["comment/tweet"].astype(str).tolist()
labels = df["majority_sent"].tolist()

le = LabelEncoder()
y = le.fit_transform(labels)
NUM_CLASSES = len(le.classes_)

train_texts, val_texts, y_train, y_val = train_test_split(
    texts, y, test_size=0.2, stratify=y, random_state=42
)

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

class BertDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

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
            self.labels[idx]
        )

    def __len__(self):
        return len(self.texts)

train_loader = DataLoader(BertDataset(train_texts, y_train), batch_size=16, shuffle=True)
val_loader = DataLoader(BertDataset(val_texts, y_val), batch_size=32)

class LSTM_BERT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, ids, mask):
        x = self.bert(ids, attention_mask=mask).last_hidden_state
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(h)

model = LSTM_BERT(NUM_CLASSES).to(device)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for ids, mask, labels in train_loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(ids, mask), labels)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "essemble model experiment #2/lstm_bert_model.pt")
print("LSTM-BERT model saved.")
