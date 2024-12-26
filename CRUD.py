import sqlite3


DATABASE = "./ML-Carros-Usados/database.db"

def inserir_modelo(name, accuracy):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    try:
        cursor.execute('''
            INSERT INTO models (name, accuracy)
            VALUES (?, ?)
        ''', (name, accuracy))

        conn.commit()
        conn.close()
        return True
    except:
        return False
    
def get_modelo_nome(name):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM models WHERE name = ?
    ''', (name,))
    model = cursor.fetchone()
    conn.close()

    if model:
        return {
            "id": model[0],
            "name": model[1],
            "accuracy": model[2]
        }
    else:
        print(f"Modelo '{name}' não encontrado.")
        return None