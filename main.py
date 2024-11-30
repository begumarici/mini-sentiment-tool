from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

def analyze_sentiment_multiclass(message):
    # Translate to English
    translated_text = GoogleTranslator(source="auto", target="en").translate(message)

    # Tokenize input
    inputs = tokenizer(translated_text, return_tensors="pt").to(device)

    # Perform sentiment analysis
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).squeeze()

    # Map scores to sentiment
    sentiments = ["Negative", "Neutral", "Positive"]
    max_index = torch.argmax(scores).item()
    sentiment = sentiments[max_index]
    confidence = round(scores[max_index].item() * 100, 2)

    return translated_text, sentiment, confidence

if __name__ == "__main__":
    print("Enter a message to analyze its sentiment (type 'quit' to exit):")
    
    while True:
        user_input = input("Message: ").strip()
        if user_input.lower() == 'quit':
            print("Exiting sentiment analysis. Goodbye!")
            break

        translated_text, sentiment, confidence = analyze_sentiment_multiclass(user_input)
        print(f"Translated Text: {translated_text}")
        print(f"Sentiment: {sentiment} ({confidence}%)")
