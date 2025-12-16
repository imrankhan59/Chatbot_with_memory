from langgraph.checkpoint.postgres import PostgresSaver
import os
from dotenv import load_dotenv

load_dotenv()
DB_URI = os.getenv("DB_URI")

def retrieve_all_threads():
    """
    Retrieve all thread IDs from Postgres memory.
    Opens a fresh connection each time.
    """
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config['configurable']['thread_id'])
        return list(all_threads)

print(retrieve_all_threads())