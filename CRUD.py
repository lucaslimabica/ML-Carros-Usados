import sqlite3
import os

base_dir = "C:/Users/lusca/Universidade/AA/TPFinal/ML-Carros-Usados"
DATABASE = os.path.join(base_dir, "database.db")

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
    
#inserir_modelo("TEste0", "17")
print(type(get_modelo_nome("Teste")))