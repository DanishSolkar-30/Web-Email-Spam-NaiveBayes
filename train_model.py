# ==========================================================
# EMAIL SPAM DETECTION USING NAIVE BAYES
# Model Training and Evaluation
# ==========================================================

# Import Libraries

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("spam.csv", encoding="latin-1")


# ==========================================================
# KEEP REQUIRED COLUMNS
# ==========================================================

df = df[['v1', 'v2']]

df.columns = ['Label', 'Email']


# ==========================================================
# DISPLAY DATASET INFORMATION
# ==========================================================

print("First 5 Records:\n")
print(df.head())

print("\nDataset Shape:", df.shape)


# ==========================================================
# CONVERT LABELS
# ham  -> 0
# spam -> 1
# ==========================================================

df['Label'] = df['Label'].map({
    'ham': 0,
    'spam': 1
})


# ==========================================================
# CHECK CLASS DISTRIBUTION
# ==========================================================

print("\nClass Distribution:")
print(df['Label'].value_counts())

print("\nClass Distribution (%):")
print((df['Label'].value_counts(normalize=True) * 100).round(2))


# ==========================================================
# FEATURES AND TARGET
# ==========================================================

X = df['Email']
y = df['Label']


# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================================
# TEXT VECTORIZATION
# ==========================================================

vectorizer = CountVectorizer(
    stop_words='english'
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)


# ==========================================================
# TRAIN MODEL
# ==========================================================

model = MultinomialNB()

model.fit(X_train, y_train)


# ==========================================================
# PREDICTION
# ==========================================================

y_pred = model.predict(X_test)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==========================================")
print("MODEL PERFORMANCE")
print("==========================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"]
))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================================
# SAVE MODEL PERFORMANCE METRICS
# ==========================================================

from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(
        precision_score(y_test, y_pred) * 100,
        2
    ),
    "recall": round(
        recall_score(y_test, y_pred) * 100,
        2
    ),
    "f1_score": round(
        f1_score(y_test, y_pred) * 100,
        2
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        y_pred
    ).tolist()
}

joblib.dump(metrics, "metrics.pkl")

print("\nMetrics saved successfully!")


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\n==========================================")
print("MODEL SAVED SUCCESSFULLY!")
print("==========================================")

print("\nCreated Files:")
print("1. model.pkl")
print("2. vectorizer.pkl")