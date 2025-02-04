import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score, classification_report
import nltk
import pickle
import re
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def filtrarPalavras(texto):
    if not isinstance(texto, str):
        return ''
    texto = re.sub(r'[^\w\s]', '', texto)  
    palavras = texto.lower().split()
    palavras = [lemmatizer.lemmatize(palavra) for palavra in palavras if palavra not in STOPWORDS]
    return ' '.join(palavras)

def matrixar(data_set, vectorizer):
    X = vectorizer.fit_transform(data_set['filtrada']).toarray()
    y = data_set['sentiment']
    return X, y

DATA = pd.read_csv("C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/DATASETCINEMA.csv")

if 'text' not in DATA.columns or 'sentiment' not in DATA.columns:
    raise ValueError("O dataset deve conter as colunas 'text' e 'sentiment'.")

DATA.dropna(inplace=True)

DATA['filtrada'] = DATA['text'].apply(filtrarPalavras)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X, y = matrixar(DATA, vectorizer=vectorizer)
X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.2, random_state=7)

GBR = GradientBoostingClassifier(n_estimators=30, random_state=7)
RF = RandomForestClassifier(n_estimators=80, max_depth=None, random_state=7)

GBR.fit(X_train, y_train)
RF.fit(X_train, y_train)

gbr_preds = GBR.predict(X_v)
rf_preds = RF.predict(X_v)

model = RF

results = {
    "Model": ["Gradient Boosting", "Random Forest"],
    "Accuracy": [
        accuracy_score(y_v, gbr_preds),
        accuracy_score(y_v, rf_preds)
    ],
    "F1 Score": [
        f1_score(y_v, gbr_preds),
        f1_score(y_v, rf_preds)
    ]
}

results_df = pd.DataFrame(results)
print(results_df)

y_pred = rf_preds
print("Accuracy do Random Forest:", accuracy_score(y_v, y_pred))
print(classification_report(y_v, y_pred))

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Acurácia com validação cruzada:", scores.mean())

with open('C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/modelo.pkl', 'wb') as model_file:
    pickle.dump(rf_preds, model_file)

with open('C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Modelo e vetor TF-IDF salvos com sucesso!")