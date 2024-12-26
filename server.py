from flask import Flask, request, jsonify
import sqlite3
import json
import CRUD # para realizar CRUD na bd


DATABASE = "./ML-Carros-Usados/database.db"
app = Flask(__name__)

def validar_json(json_string, required_keys) -> bool:
    """
    Valida o JSON assegurando que nele estão todas as necessárias chaves.

    Args:
        json_string (str): O JSON como uma string.
        required_keys (list): Array de todas as chaves do JSON.

    Returns:
        bool: True se todas as chaves necessárias estão presentes, caso contrário, False.
    """
    try:
        # Transforma o JSON string em um dicionário
        data = json.loads(json_string)
    except:  # noqa: E722
        return False

    # Verifica se todas as chaves necessárias estão presentes
    for key in required_keys:
        if key not in data:
            return False

    return True

# EP do status da API
@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "API está online."}), 200

# EP para info do modelo específico
@app.route('/model-info/<str:model>', methods=['GET'])
def model_info():
    pass

# EP para postar um modelo
@app.route('/model', methods=['POST'])
def model_post():
    data = request.get_json()
    if validar_json(json.dumps(data), ["name", "accuracy"]):
        name = data["name"]
        accuracy = data["accuracy"]
        insert = CRUD.inserir_modelo(name, accuracy)
        if insert:
            return jsonify(
                {"sucess": "True"},
                {"data": data}
            ), 201
        else:
            return jsonify(
                {"sucess": "False"},
                {"error": "Erro na inserção"}
            ), 500
    else:
        return jsonify(
                {"sucess": "False"},
                {"error": "Erro no JSON inserido"}
            ), 500

# Executar o servidor
if __name__ == "__main__":
    app.run(debug=True)
