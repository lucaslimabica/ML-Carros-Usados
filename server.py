from flask import Flask, request, jsonify
import sqlite3
import json
import os


base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
DATABASE = os.path.join(base_dir, "database.db")
app = Flask(__name__)
def get_db_connection():
    """
    Cria e retorna uma connection à base de dados SQLite.

    Setada no sqlite3.Row,
    Que permite-nos aceder às colunas pelo nome e index.

    Returns:
        sqlite3.Connection: Uma conexão à database.
    """
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    return conn, cursor

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
@app.route('/model-info/<string:model>', methods=['GET'])
def model_info(model):
    conn, cursor = get_db_connection()
    try:
        cursor.execute('''
        SELECT * FROM models WHERE name = ?
        ''', (model,))
        model = cursor.fetchone()
        conn.close()
        if model:
            name = model[1]
            accuracy = model[2]
            return jsonify({
                "success": True,
                "data": {
                    "name": name,
                    "accuracy": accuracy
                }
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": f"Model wasn't found"
            }), 404
    except Exception as e:
        return jsonify({
                "success": False,
                "error": f"Erro no servidor: {str(e)}"
            }), 500

# Endpoint para postar um modelo
@app.route('/model', methods=['POST'])
def model_post():
    data = request.get_json()
    if validar_json(json.dumps(data), ["name", "accuracy"]):
        name = data["name"]
        accuracy = data["accuracy"]
        conn, cursor = get_db_connection()
        try:
            cursor.execute('''
                INSERT INTO models (name, accuracy)
                VALUES (?, ?)
            ''', (name, accuracy))

            conn.commit()
            conn.close()
            return jsonify({
                "success": True,
                "data": data
            }), 201
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Erro no servidor: {str(e)}"
            }), 500
    else:
        return jsonify({
            "success": False,
            "error": "Erro no JSON inserido. Verifique os campos 'name' e 'accuracy'."
        }), 400


# Executar o servidor
if __name__ == "__main__":
    app.run(debug=True)
