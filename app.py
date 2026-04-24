import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')


df = pd.read_csv(r"E:\ML\text.csv")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    words = str(text).lower().split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned"] = df["text"].apply(clean_text)


cv = CountVectorizer()
bow = cv.fit_transform(df["cleaned"])
bow_words = cv.get_feature_names_out()


tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["cleaned"])
tfidf_words = tfidf.get_feature_names_out()


def get_sentiment(text):
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["text"].apply(get_sentiment)


all_words = " ".join(df["cleaned"]).split()
tags = Counter(all_words).most_common(5)


st.title("Text Mining Dashboard")


st.subheader("Dataset")
st.write(df)


st.subheader("Preprocessed Text")
st.write(df[["text", "cleaned"]])

# BoW
st.subheader("Bag of Words (BoW)")
st.write(bow_words[:20])  

# TF-IDF
st.subheader("TF-IDF Features")
st.write(tfidf_words[:20])  

# Dimensionality Reduction
st.subheader("Dimensionality Reduction")
st.write("Not required as dataset size is small.")

# Sentiment
st.subheader("Sentiment Distribution")
st.bar_chart(df["Sentiment"].value_counts())

# Tags
st.subheader("Top Tags")
tag_df = pd.DataFrame(tags, columns=["Word", "Count"])
st.table(tag_df)

# WordCloud
st.subheader("WordCloud")
wordcloud = WordCloud().generate(" ".join(df["cleaned"]))
plt.imshow(wordcloud)
plt.axis("off")
st.pyplot(plt)