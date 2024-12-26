import sqlite3


db_file = "database.db"

# Conectar ao banco de dados (ou criar se não existir)
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
# Criar tabela para armazenar informações dos modelos
cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        accuracy REAL,
    )
''')
conn.commit()
conn.close()
print(f"Banco de dados '{db_file}' e tabela 'models' criados com sucesso.")

