from flask import Flask, request, render_template, jsonify
from transformers import pipeline

# Inicializando o Flask
app = Flask(__name__)

# Carregando o modelo pré-treinado do Hugging Face
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")


# Rota principal para exibir a página HTML
@app.route('/')
def home():
    return render_template('index.html')

# Endpoint para previsão (HTML)
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not text:
        return render_template('index.html', error="Por favor, insira um texto para análise.")

    result = sentiment_analyzer(text)[0]
    prediction = result['label'].lower()

    return render_template('index.html', prediction=prediction, text=text)

# Endpoint para previsão (API JSON)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Texto ausente na requisição.'}), 400

    text = data['text']
    result = sentiment_analyzer(text)[0]
    prediction = result['label'].lower()

    return jsonify({'text': text, 'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)