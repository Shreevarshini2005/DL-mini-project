import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# 1. Load CSV file
df = pd.read_csv('data/mbti_1.csv')

# 2. Create labels: Introvert = 1, Extrovert = 0
df['label'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)

# 3. Clean the text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)  # remove links
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove symbols/numbers
    text = text.lower()  # lowercase everything
    return text

df['clean_posts'] = df['posts'].apply(clean_text)

# 4. Split into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['clean_posts'], df['label'], test_size=0.2, random_state=42
)

# 5. Use BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(
    list(train_texts), truncation=True, padding=True, max_length=512
)
test_encodings = tokenizer(
    list(test_texts), truncation=True, padding=True, max_length=512
)

print("✅ Preprocessing done.")
