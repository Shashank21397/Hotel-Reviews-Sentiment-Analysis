# ===========================
# train_sentiment_model.py
# ===========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# ===========================
# 1️⃣ Prepare Data
# ===========================
data = {
    "text": [
        # Positive
        "The room was clean and comfortable",
        "Staff were very friendly and helpful",
        "Amazing service and great food",
        "I loved my stay here, will come again",
        "The view from the hotel was breathtaking",
        # Negative
        "The room was dirty and smelled bad",
        "Staff were rude and unhelpful",
        "Terrible experience, never coming back",
        "The bed was uncomfortable and noisy",
        "The food was cold and tasteless",
        # Neutral
        "The hotel was okay, nothing special",
        "It was an average stay",
        "The room was fine, not great but not bad",
        "Just a regular hotel experience",
        "The service was acceptable"
    ],
    "sentiment": [
        "Positive","Positive","Positive","Positive","Positive",
        "Negative","Negative","Negative","Negative","Negative",
        "Neutral","Neutral","Neutral","Neutral","Neutral"
    ]
}

df = pd.DataFrame(data)

# ===========================
# 2️⃣ Train-Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)

# ===========================
# 3️⃣ TF-IDF + Logistic Regression
# ===========================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_tfidf, y_train)

# ===========================
# 4️⃣ Evaluate Model
# ===========================
y_pred = model.predict(X_test_tfidf)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ===========================
# 5️⃣ Save Artifacts
# ===========================
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/sentiment_model.joblib")
joblib.dump(vectorizer, "artifacts/vectorizer.joblib")

print("\n🎉 Model training complete! Files saved in 'artifacts/' folder.")
