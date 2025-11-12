# -------------------------------
# 🧠 Day 5: Fake Job Detection Model
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ Load dataset (make sure the CSV file is in the same folder)
df = pd.read_csv('fake_job_postings.csv')

# Remove missing descriptions
df = df.dropna(subset=['description'])

# 2️⃣ Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['description'])
y = df['fraudulent']

# 3️⃣ Split data into train & test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4️⃣ Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = model.predict(X_test)

# 6️⃣ Evaluate performance
print("\n✅ Model Evaluation Results ✅")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7️⃣ Check example predictions
test_samples = [
    "Work from home! Limited vacancies. Apply now.",
    "We are hiring a data scientist for our Bangalore office."
]
sample_features = vectorizer.transform(test_samples)
predictions = model.predict(sample_features)

print("\n🔍 Sample Predictions:")
for text, pred in zip(test_samples, predictions):
    label = "Fake" if pred == 1 else "Real"
    print(f"Text: {text} → Predicted: {label}")
