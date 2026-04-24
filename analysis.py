import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from collections import Counter

nltk.download('stopwords')

# Load dataset
df = pd.read_csv(r"E:\ML\text.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = str(text).lower().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned"] = df["text"].apply(clean_text)

# Sentiment Analysis
def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["text"].apply(get_sentiment)

# Tag Extraction
all_words = " ".join(df["cleaned"]).split()
tags = Counter(all_words).most_common(5)

print("\nTop Tags:", tags)

# 🔥 SAVE NEW FILE (IMPORTANT)
df.to_csv(r"E:\ML\processed_text.csv", index=False)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["cleaned"])

# Get feature names
feature_names = tfidf.get_feature_names_out()

# Convert to DataFrame (first few rows)
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print("\nTF-IDF Sample:\n", tfidf_df.head())