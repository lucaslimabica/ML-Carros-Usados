
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from flask import Flask, request, render_template, jsonify
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask
app = Flask(__name__)

# Load model and tokenizer
try:
    model_path = "sentiment_analysis_model.h5"
    tokenizer_path = "tokenizer.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        logger.error("Model or tokenizer files not found")
        raise FileNotFoundError("Required model files are missing")
        
    sentiment_analyzer = load_model(model_path)
    logger.info("Model loaded successfully!")

    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    logger.info("Tokenizer loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise

MAX_SEQ_LENGTH = 50

def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding="post")
    return padded_sequences

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text = request.form.get('text')
        if not text:
            return render_template('index.html', error="Por favor, insira um texto para análise.")

        processed_text = preprocess_text(text)
        prediction = sentiment_analyzer.predict(processed_text)[0][0]
        sentiment = "positivo" if prediction > 0.5 else "negativo"

        return render_template('index.html', prediction=sentiment, text=text)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return render_template('index.html', error="Erro ao processar o texto.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Texto ausente na requisição.'}), 400

        text = data['text']
        processed_text = preprocess_text(text)
        prediction = sentiment_analyzer.predict(processed_text)[0][0]
        sentiment = "positivo" if prediction > 0.5 else "negativo"

        return jsonify({'text': text, 'sentiment': sentiment})
    except Exception as e:
        logger.error(f"Error in API prediction: {e}")
        return jsonify({'error': 'Erro ao processar o texto.'}), 500

if __name__ == '__main__':
    from waitress import serve
    logger.info("Starting server on port 80...")
    serve(app, host="0.0.0.0", port=80, threads=1, connection_limit=50, url_scheme='http')
