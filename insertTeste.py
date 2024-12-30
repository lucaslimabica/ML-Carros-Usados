import json
import requests

data = {"name": "Lucas", "accuracy": "12"}

response = requests.post(
    "http://127.0.0.1:5000/model",
    data=json.dumps(data),
    headers={"Content-Type": "application/json"}
)

# Exibir a resposta do servidor
print(response.status_code)
print(response.json())
