from flask import Flask, request, jsonify
from predict import predict_news

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    article = data.get('article', '')
    
    # Predict if the article is fake or real
    result = predict_news(article)
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
