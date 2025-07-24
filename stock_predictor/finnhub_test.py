import finnhub
from datetime import datetime, timedelta
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
today = datetime.today()
minus_time = today - timedelta(days=1)

from_date = minus_time.strftime('%Y-%m-%d')
to_date = today.strftime('%Y-%m-%d')

# Setup client
api_key = 'YOUR_API_KEY'
finnhub_client = finnhub.Client(api_key='d21a131r01qkdupigrm0d21a131r01qkdupigrmg')
company_name_keywords = ['Apple', 'AAPL']
# Example: Get news for AAPL
news = finnhub_client.company_news('AAPL', _from=from_date, to=to_date)


filtered_news = [
    article for article in news
    if any(keyword.lower() in (article['headline'] + article['summary']).lower() for keyword in company_name_keywords)
]

filtered_news = filtered_news[:5]

sentences = [news['summary'] for news in filtered_news]

# Load FinBERT tokenizer and model
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# Run model and get logits
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Apply softmax to get probabilities
probabilities = torch.softmax(logits, dim=1).numpy()

# Labels mapping
labels = ['neutral', 'positive', 'negative']

for i, sentence in enumerate(sentences):
    score_array = [0 , 0, 0]#neutral, positive, negative
    probs = probabilities[i]
    pred_label = labels[np.argmax(probs)]
    for i in range(3):
        score_array[i] += probs[i]


print(f'Overall Sentiment Scores: Neutral={score_array[0]:.3f}, Positive={score_array[1]:.3f}, Negative={score_array[2]:.3f}')
print(f'Overall Sentiment: {labels[np.argmax(score_array)]}')