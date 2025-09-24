
#importing libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading dataset
dataset_filename = 'IMDB Dataset.csv'
df = pd.read_csv(dataset_filename)

print(f"Dataset '{dataset_filename}' loaded successfully.")
print("\n" + "="*50 + "\n")

#preprocessing
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Text preprocessing complete.")
print("\n" + "="*50 + "\n")


X = df['cleaned_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Data splitting complete.")
print("\n" + "="*50 + "\n")

#training
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Text vectorization with TF-IDF and N-grams complete.")
print(f"New vocabulary size: {len(vectorizer.get_feature_names_out())} features")
print("\n" + "="*50 + "\n")


model = LogisticRegression(max_iter=1000, random_state=42)
print("Training the Logistic Regression model...")
model.fit(X_train_vec, y_train)
print("Model training complete.")
print("\n" + "="*50 + "\n")

#Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")
print("--------------------------------")
