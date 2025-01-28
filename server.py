from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
app = Flask(__name__)

with open('C:/Users/lusca/Universidade/AA/TPFinal/modelo.pkl', 'rb') as file:
    MODELO = pickle.load(file)

with open('C:/Users/lusca/Universidade/AA/TPFinal/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)


STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def filtrarPalavras(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    palavras = texto.lower().split()
    palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)


# EP do status da API
@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "API está online."}), 200

# EP de POST da frase
@app.route('/previsao', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'O campo "text" é obrigatório'}), 500
        
        # Tratar input
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
