import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
import pickle
import re
from nltk.stem import WordNetLemmatizer

# Setando as stopwords e lematizador
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Função para remover stopwords, pontuação e lematizar
def filtrarPalavras(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'[^\w\s]', '', texto)  # Remove pontuação
    palavras = texto.lower().split()
    palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)

# Função para vetorizar os textos e associar com os rótulos
def matrixar(data_set, vectorizer):
    X = vectorizer.fit_transform(data_set['filtrada']).toarray()
    y = data_set['sentiment']
    return X, y

# Carregar os dados
DATA = pd.read_csv("C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/DATASETCINEMA.csv")

# Verificar se as colunas esperadas estão no dataset
if 'text' not in DATA.columns or 'sentiment' not in DATA.columns:
    raise ValueError("O dataset deve conter as colunas 'text' e 'sentiment'.")

# Remover dados faltantes
DATA.dropna(inplace=True)

# Aplicar pré-processamento nos textos
DATA['filtrada'] = DATA['text'].apply(filtrarPalavras)

# Inicializar o vetor TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Vetorizar os dados e dividir em treino e validação
X, y = matrixar(DATA, vectorizer=vectorizer)
X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.2, random_state=7)

# Configurar e treinar o modelo de rede neural
#model = MLPClassifier(hidden_layer_sizes=(150, 50, 25), max_iter=300, alpha=0.0001, solver='adam', random_state=7)
model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=7)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_v)
print("Accuracy:", accuracy_score(y_v, y_pred))
print(classification_report(y_v, y_pred))

# Validação cruzada
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Acurácia com validação cruzada:", scores.mean())

# Salvar modelo e vetorizador
with open('modelo.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Modelo e vetor TF-IDF salvos com sucesso!")