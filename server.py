from flask import Flask, request, jsonify, sqlite3


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
    return conn

# Endpoint do status da API
@app.route('/status', methods=['GET'])
def status_check():
    return jsonify({"status": "API está online."}), 200

# Endpoint para info do modelo específico
@app.route('/model-info/<str:model>', methods=['GET'])
def model_info():
    pass

# Executar o servidor
if __name__ == "__main__":
    app.run(debug=True)
