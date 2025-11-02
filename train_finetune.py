import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "./depaura_model"  # existing model
DATA_PATH = "data/mbti_1.csv"
EPOCHS = 3                      # fine-tune for 2–3 epochs
BATCH_SIZE = 8
LR = 1e-5                       # smaller learning rate for fine-tuning
MAX_LEN = 256

# -------------------------------
# Load tokenizer & model
# -------------------------------
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------------
# Dataset Preparation
# -------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Total samples in original dataset: {len(df)}")

# Clean text
df["posts"] = df["posts"].str.replace(r'http\S+', '', regex=True)
df["posts"] = df["posts"].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df["posts"] = df["posts"].str.lower()

# Label encoding
types = sorted(df["type"].unique())
label2id = {t: i for i, t in enumerate(types)}
df["label"] = df["type"].map(label2id)

# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["posts"].tolist(), df["label"].tolist(), test_size=0.1, random_state=42
)

# -------------------------------
# Custom Dataset
# -------------------------------
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_len, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = MBTIDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = MBTIDataset(val_texts, val_labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = AdamW(model.parameters(), lr=LR)

# -------------------------------
# Fine-tuning
# -------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch in tqdm(train_loader, desc="Fine-tuning"):
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")

    # -------------------------------
    # Validation
    # -------------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            _, preds = torch.max(outputs.logits, dim=1)
            correct += (preds == inputs["labels"]).sum().item()
            total += inputs["labels"].size(0)

    val_acc = correct / total
    print(f"Validation Accuracy after Epoch {epoch+1}: {val_acc:.4f}")

# -------------------------------
# Save fine-tuned model
# -------------------------------
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("\n✅ Fine-tuning completed! Model updated at ./depaura_model")
