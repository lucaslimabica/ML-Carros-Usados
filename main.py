import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import numpy as np


base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
TRAIN = os.path.join(base_dir, "train.csv")
TEST = os.path.join(base_dir, "test.csv")


train = pd.read_csv(TRAIN)
train = train.drop(['id'], axis=1)

categorical_cols = ['brand', 'model', 'fuel_type', 'engine', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    label_encoders[col] = le 

X = train.drop(['price'], axis=1)
y = train['price'] 
X_train, X_v, y_train, y_v = train_test_split(X, y, test_size=0.2, random_state=7) 
model = MLPRegressor(hidden_layer_sizes=(100, 50, 25,), max_iter=2000, alpha=0.0001, solver='adam', random_state=7)
model.fit(X_train, y_train)

y_pred = model.predict(X_v)

rmse = mean_squared_error(y_v, y_pred, squared=False)
print(f"RMSE: {rmse:.2f} euros")
print(f"MAE: {mean_absolute_error(y_v, y_pred):.2f} euros")
print(f"RRMSE: {rmse / y.mean() * 100:.2f}%")
print(f"Media dos precos: {y.mean():.2f} euros")
print(f"Mediana dos precos: {y.median():.2f} euros")
print(f"R2: {r2_score(y_v, y_pred):.2f}")
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')


test = pd.read_csv(TEST)
for col in categorical_cols:
    label_encoders[col].classes_ = np.append(label_encoders[col].classes_, 'unknown')
    
    test[col] = test[col].apply(lambda x: x if x in label_encoders[col].classes_ else 'unknown')
    test[col] = label_encoders[col].transform(test[col])

test_predictions = model.predict(test.drop(['id'], axis=1))

submission = pd.DataFrame({
    'Id': test['id'],           
    'price': test_predictions   
})

#submission_file = os.path.join(base_dir, 'submission.csv')
#submission.to_csv(submission_file, index=False)
#print(f"Arquivo de submissão gerado com sucesso: {submission_file}")
