import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.neural_network import MLPClassifier 
import nltk
import pickle


# Setando as stopwords
nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('portuguese')

# Vetorizando
def filtrarPalavras(texto):
    palavras = texto.lower().split()
    palavras = [palavra for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)


def matrixar():
    pass

DATA = pd.DataFrame({
    'text': [
        'Eu amei este filme!', 
        'Pior coisa do mundo.',
        'Atendimento maravilhoso',
        'Horrível',
        'Fantastico, simplesmente maravilhoso',
        'Não foi o que eu esperava',
        'Muito bom',
        'Desapontado com isto.',
        'Perfeito demais!',
        'Ruim...',
        'Adorei a experiência, recomendo muito!',
        'Comida estava horrível, não volto mais.',
        'Que lugar incrível, foi inesquecível!',
        'O produto não funciona como deveria, uma decepção.',
        'Entrega rápida e sem problemas. Excelente serviço!',
        'Que filme mais chato, quase dormi.',
        'Fiquei surpreso com a qualidade, parabéns!',
        'Péssima qualidade, produto quebrou no primeiro uso.',
        'Amei a apresentação, voltarei mais vezes.',
        'Nunca mais quero passar por isso, foi terrível.',
        'Gostei bastante, superou minhas expectativas!',
        'Um completo desperdício de dinheiro.',
        'Simplesmente fantástico, me emocionei!',
        'Serviço medíocre, me senti ignorado.',
        'Muito feliz com o atendimento, foi impecável.',
        'Inaceitável, um completo desrespeito ao cliente.',
        'Experiência agradável e sem contratempos.',
        'Não gostei, esperava algo muito melhor.',
        'Recomendo a todos, foi perfeito!',
        'Arrependido de ter comprado isso.',
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
})


#print(DATA.head())

DATA['filtrada'] = DATA['text'].apply(filtrarPalavras)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(DATA['filtrada']).toarray() # Tabela valor de "t" em "d" vinculado ao sentimento 
y = DATA['sentiment']

X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.2, random_state=7)

# Modelos

# Random Forest
#rf_model = RandomForestClassifier(n_estimators=1000, random_state=7)
#rf_model.fit(X_train, y_train)

# Avaliação
#y_pred = rf_model.predict(X_v)
#print("Random Forest - Accuracy:", accuracy_score(y_v, y_pred))
#print(classification_report(y_v, y_pred))

# Redes Neuronais
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1, alpha=0.1, solver='sgd', random_state=7)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_v)

# Avaliar a acurácia
print("Redes Neuronais - Accuracy:", accuracy_score(y_v, y_pred))