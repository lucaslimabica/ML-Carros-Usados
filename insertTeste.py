import json
import requests

data = {"name": "tESTE2", "accuracy": "1071"}

response = requests.post(
    "http://127.0.0.1:5000/model",
    data=json.dumps(data),
    headers={"Content-Type": "application/json"}
)

# Exibir a resposta do servidor
print(response.status_code)
print(response.json())
