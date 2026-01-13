from transformers import pipeline


# Load sentiment analysis pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# Sample text
text = "I am feeling bad!"


# Analyze sentiment
result = sentiment_analyzer(text)


print(f"Sentiment: {result[0]['label']}")
print(f"Confidence Score: {result[0]['score']}")
