from psycopg import connect

db_uri = "postgresql://postgres:1234@localhost:5432/chatbot_con"

try:
    conn = connect(db_uri)
    print("✅ Connection opened successfully")
    
    # run a simple query
    with conn.cursor() as cur:
        cur.execute("SELECT 1;")
        print("Query result:", cur.fetchone())
    
    conn.close()
    print("✅ Connection closed")
except Exception as e:
    print("❌ Connection failed:", e)
