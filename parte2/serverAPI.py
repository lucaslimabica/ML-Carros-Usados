from flask import Flask, request, render_template, jsonify
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

with open('C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/modelo.pkl', 'rb') as file:
    MODELO = pickle.load(file)

with open('C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

STOPWORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def filtrarPalavras(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'[^\w\s]', '', texto)
    palavras = texto.lower().split()
    palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)

@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "API está online."}), 200

@app.route('/previsao', methods=['POST'])
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

@app.route('/', methods=['GET', 'POST'])
def home():
    response = None
    if request.method == 'POST':
        user_input = request.form['user_input'] 
        texto_filtrado = filtrarPalavras(user_input)
        texto_vetorizado = vectorizer.transform([texto_filtrado]).toarray()
        predicao = MODELO.predict(texto_vetorizado)[0]
        response = "positivo" if predicao == 1 else "negativo"
    
    return render_template("index.html", response=response)

if __name__ == '__main__':
    app.run(debug=True)
