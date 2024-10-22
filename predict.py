import pickle
from utils import preprocess_text

# Load the saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(article):
    # Preprocess the input text
    article_preprocessed = preprocess_text(article)
    
    # Transform the input text using the saved vectorizer
    article_tfidf = vectorizer.transform([article_preprocessed])
    
    # Make prediction
    prediction = model.predict(article_tfidf)
    
    return 'Fake' if prediction[0] == 1 else 'Real'

# Example usage
if __name__ == "__main__":
    article = "Your article text goes here"
    result = predict_news(article)
    print(f"The article is {result}")
