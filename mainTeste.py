import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Modelos
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor



base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
TRAIN = os.path.join(base_dir, "train.csv")
TEST = os.path.join(base_dir, "test.csv")


train = pd.read_csv(TRAIN)
# Remover cols inuteis
train = train.drop(['id'], axis=1)

# LabelEncoder para transformar variáveis categóricas em numéricas
categorical_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le 

X = train.drop(['price'], axis=1) # Features
y = train['price'] # Target

X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.2, random_state=7) # 20% para validação

# Treino
RandomForest = RandomForestRegressor(n_estimators=100, random_state=7)
SVRmodel = SVR(kernel='rbf')
GBR = GradientBoostingRegressor(n_estimators=500, random_state=7)
MLPmodel = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=2000, alpha=0.0001, solver='adam', random_state=7)

models = {
    "Random Forest": RandomForest,
    "SVR": SVRmodel,
    "Gradient Boosting": GBR,
    "MLP Regressor": MLPmodel
}
# Dicionários para armazenar métricas
results = {"Model": [], "RMSE": [], "R2 Score": []}

for name, model in models.items():
    # Treinamento
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_v)
    
    # Calculando métricas
    rmse = np.sqrt(mean_squared_error(y_v, y_pred))
    r2 = r2_score(y_v, y_pred)
    
    # Armazenar os resultados
    results["Model"].append(name)
    results["RMSE"].append(rmse)
    results["R2 Score"].append(r2)

# Transformar em DataFrame
results_df = pd.DataFrame(results)
# Plotando os resultados
plt.figure(figsize=(12, 6))

# RMSE
plt.subplot(1, 2, 1)
plt.bar(results_df["Model"], results_df["RMSE"], color='skyblue')
plt.title("Comparação de RMSE")
plt.ylabel("RMSE")
plt.xticks(rotation=45)

# R2 Score
plt.subplot(1, 2, 2)
plt.bar(results_df["Model"], results_df["R2 Score"], color='lightgreen')
plt.title("Comparação de R2 Score")
plt.ylabel("R2 Score")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
