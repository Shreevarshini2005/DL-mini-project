import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
df = pd.read_csv("data/mbti_1.csv")
print(f"Total samples in original dataset: {len(df)}")
print(df['type'].value_counts())

# -------------------------------
# 2️⃣ Balance Dataset (resample)
# -------------------------------
min_count = df['type'].value_counts().min()
balanced_df = (
    df.groupby('type', group_keys=False)
    .apply(lambda x: resample(x, replace=True, n_samples=min_count, random_state=42))
)
print(f"Number of samples after balancing: {len(balanced_df)}")
print("Sample labels after balancing:")
print(balanced_df['type'].value_counts())

# -------------------------------
# 3️⃣ Encode labels
# -------------------------------
label2id = {label: idx for idx, label in enumerate(sorted(balanced_df['type'].unique()))}
id2label = {idx: label for label, idx in label2id.items()}
balanced_df["label"] = balanced_df["type"].map(label2id)

# -------------------------------
# 4️⃣ Train/Validation Split
# -------------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    balanced_df["posts"].tolist(),
    balanced_df["label"].tolist(),
    test_size=0.2,
    random_state=42
)

# -------------------------------
# 5️⃣ Tokenizer & Dataset
# -------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class MBTIDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,  
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

train_dataset = MBTIDataset(train_texts, train_labels)
val_dataset = MBTIDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# -------------------------------
# 6️⃣ Model Setup
# -------------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id)
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze BERT layers to focus on classifier (helps small dataset)
for param in model.bert.parameters():
    param.requires_grad = False

# -------------------------------
# 7️⃣ Weighted Loss (handle rare MBTI types)
# -------------------------------
class_counts = balanced_df['label'].value_counts().sort_index()
weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
weights = weights.to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

# -------------------------------
# 8️⃣ Optimizer & Scheduler
# -------------------------------
optimizer = AdamW(model.parameters(), lr=1e-5)  # smaller LR
total_steps = len(train_loader) * 10  # 10 epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# -------------------------------
# 9️⃣ Training Loop (10 epochs)
# -------------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    print(f"\nEpoch {epoch + 1}/{epochs}")
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average Training Loss: {avg_train_loss:.4f}")

    # -------------------------------
    # 🔹 Validation
    # -------------------------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])
    accuracy = correct / total
    print(f"Validation Accuracy after Epoch {epoch + 1}: {accuracy:.4f}")

# -------------------------------
# 10️⃣ Save Model & Tokenizer
# -------------------------------
model.save_pretrained("./depaura_model")
tokenizer.save_pretrained("./depaura_model")

print("\n✅ Training completed and model saved to ./depaura_model")
