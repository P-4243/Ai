import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model and vectorizer
with open('fake_news_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(article):
    # Transform the input text using the saved vectorizer
    article_tfidf = vectorizer.transform([article])
    
    # Make prediction
    prediction = model.predict(article_tfidf)
    
    return 'Fake' if prediction[0] == 1 else 'Real'

if __name__ == '__main__':
    while True:
        article = input("Please enter the news article text (or 'exit' to quit): ")
        if article.lower() == 'exit':
            break
        result = predict_news(article)
        print(f"The article is {result}.")
