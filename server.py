from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer


base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
app = Flask(__name__)

# Carregar o modelo treinado
with open('modelo.pkl', 'rb') as file:
    MODELO = pickle.load(file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def matrixarFeatures(data_set):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(data_set['filtrada']).toarray()
    return X

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
        X = vectorizer.transform([text]).toarray()
        
        # Prever
        previsao = MODELO.predict(X)
        probabilidade = MODELO.predict_proba(X).tolist()
        
        # Retorno
        return jsonify({
            'prediction': int(previsao[0]),
            'probabilities': probabilidade
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 503

if __name__ == '__main__':
    app.run(debug=True)
