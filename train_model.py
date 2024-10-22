import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Create a sample fake news dataset
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'text': [
        'The government has announced a new plan to build a bridge.',
        'Aliens spotted in New York City!',
        'Scientists develop new energy source from water.',
        'New study reveals the benefits of chocolate.',
        'A new system designed to detect fake news was launched today.',
        'President announces new policy on education.',
        'Mysterious creature found in the Amazon forest.',
        'A group of conspiracy theorists predict that a zombie apocalypse will occur next year.',
        'A recent study shows that consuming chocolate can improve health.',
        'The Mars mission has successfully landed on the planet.'
    ],
    'label': [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]  # 0 = Real, 1 = Fake
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV file
df.to_csv('fake_news.csv', index=False)

# Load the dataset
data = pd.read_csv('fake_news.csv')

# Prepare features (X) and labels (y)
X = data['text']
y = data['label']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
with open('fake_news_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

# Predict on the test set
X_test_tfidf = vectorizer.transform(X_test)
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
