# src/agents/db_setup.py

from psycopg import connect
from langgraph.checkpoint.postgres import PostgresSaver
import os
from dotenv import load_dotenv
load_dotenv()


DB_URI = os.getenv("DB_URI")


def setup_database():
    print("🚀 Connecting to Postgres...")
    conn = connect(DB_URI)

    print("🧱 Running LangGraph table setup...")
    with PostgresSaver.from_conn_string(DB_URI) as saver:
        saver.setup()

    print("✅ Database setup complete! Tables created successfully.")


if __name__ == "__main__":
    setup_database()

 