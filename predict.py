from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# -------------------------------
# Load model and tokenizer
# -------------------------------
model = BertForSequenceClassification.from_pretrained("./depaura_model")
tokenizer = BertTokenizer.from_pretrained("./depaura_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# -------------------------------
# ID to Label mapping
# -------------------------------
id2label = {
    0: "ENFJ", 1: "ENFP", 2: "ENTJ", 3: "ENTP",
    4: "INFJ", 5: "INFP", 6: "INTJ", 7: "INTP",
    8: "ISFJ", 9: "ISFP", 10: "ISTJ", 11: "ISTP",
    12: "ESTJ", 13: "ESTP", 14: "ESFJ", 15: "ESFP"
}

# -------------------------------
# Prediction function
# -------------------------------
def predict_personality(text, top_k=3, temperature=1.5):
    """
    temperature > 1 sharpens the probabilities to highlight the top predictions
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze()

    # Apply softmax with temperature to sharpen probabilities
    probs = F.softmax(logits * temperature, dim=0)

    # Get top k predictions
    top_probs, top_indices = torch.topk(probs, top_k)
    top_predictions = [(id2label[idx.item()], prob.item()) for idx, prob in zip(top_indices, top_probs)]

    return top_predictions

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    user_text = input("Enter your text: ")
    top_preds = predict_personality(user_text)

    print("\nTop predictions (with normalized confidence):")
    for label, prob in top_preds:
        print(f"{label}: {prob*100:.2f}%")
