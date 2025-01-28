from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, render_template, jsonify
from transformers import pipeline


base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
app = Flask(__name__)

with open('C:/Users/lusca/Universidade/AA/TPFinal/modelo.pkl', 'rb') as file:
    MODELO = pickle.load(file)

with open('C:/Users/lusca/Universidade/AA/TPFinal/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sentiment_analyzer = pipeline("sentiment-analysis", framework="pt")


def filtrarPalavras(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = texto.lower().split()
    palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "API está online."}), 200

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')
    if not text:
        return render_template('index.html', error="Por favor, insira um texto para análise.")

    result = sentiment_analyzer(text)[0]
    prediction = result['label'].lower()

    return render_template('index.html', prediction=prediction, text=text)

@app.route('/v2/predict', methods=['POST'])
def api_predict():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'Texto ausente na requisição.'}), 400

    text = data['text']
    result = sentiment_analyzer(text)[0]
    prediction = result['label'].lower()

    return jsonify({'text': text, 'sentiment': prediction})

@app.route('/sebastiao/previsao', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'O campo "text" é obrigatório'}), 500
        
        text = data['text']
        texto_filtrado = filtrarPalavras(text)
        texto_vetorizado = vectorizer.transform([texto_filtrado]).toarray()
        predicao = MODELO.predict(texto_vetorizado)[0]
        sentimento = "positivo" if predicao == 1 else "negativo"
        return jsonify({"texto": text, "sentimento": sentimento})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 503


if __name__ == '__main__':
    app.run(debug=True)
