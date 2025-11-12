# Day 1-2: Dataset Loading and Preprocessing

import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('fake_job_postings (1).csv')

# -------------------------
# Basic Dataset Info
# -------------------------
print("Total Records:", len(df))
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Count of real vs fake jobs
if 'fraudulent' in df.columns:
    print("\nReal vs Fake Jobs:")
    print(df['fraudulent'].value_counts())

# Display 3 examples of fake jobs
print("\n3 Fake Job Descriptions:")
fake_jobs = df[df['fraudulent'] == 1]['description'].head(3)
for i, desc in enumerate(fake_jobs, 1):
    print(f"{i}:", desc[:300], "\n")

# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()  # lowercase
    text = re.sub(r'<.*?>', ' ', text)  # remove HTML tags
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)  # remove punctuation
    words = [WordNetLemmatizer().lemmatize(w) for w in text.split() if w not in stopwords.words('english')]
    return ' '.join(words)

# Apply cleaning to text columns
text_cols = ['company_profile', 'description', 'requirements']
for col in text_cols:
    df['clean_' + col] = df[col].apply(clean_text)

# -------------------------
# Average words before & after cleaning for 'description'
# -------------------------
df['desc_word_count'] = df['description'].apply(lambda x: len(str(x).split()))
df['clean_desc_word_count'] = df['clean_description'].apply(lambda x: len(str(x).split()))
print("\nAverage words in original descriptions:", df['desc_word_count'].mean())
print("Average words after cleaning:", df['clean_desc_word_count'].mean())

# -------------------------
# Feature Extraction
# -------------------------
# 1️⃣ Bag-of-Words for 'description'
texts = df['clean_description'].tolist()
bow_vectorizer = CountVectorizer(max_features=2000)
X_bow = bow_vectorizer.fit_transform(texts)
print("\nBoW shape:", X_bow.shape)
print("Sample BoW features:", bow_vectorizer.get_feature_names_out()[:10])

# 2️⃣ TF-IDF for 'description'
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)
print("\nTF-IDF shape:", X_tfidf.shape)
print("Sample TF-IDF features:", tfidf_vectorizer.get_feature_names_out()[:10])

# -------------------------
# Top 20 most frequent words in job descriptions (BoW)
# -------------------------
import numpy as np
word_counts = np.sum(X_bow.toarray(), axis=0)
words = bow_vectorizer.get_feature_names_out()
freq_df = pd.DataFrame({'word': words, 'count': word_counts}).sort_values(by='count', ascending=False)
print("\nTop 20 Most Frequent Words in Job Descriptions:")
print(freq_df.head(20))
