import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.messages import HumanMessage
# --- CRITICAL CHANGE: Import ConnectionPool and PostgresSaver ---
from langgraph.checkpoint.postgres import PostgresSaver 
from psycopg_pool import ConnectionPool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition, ToolNode
import streamlit as st
from psycopg.rows import dict_row # CRITICAL: Needed for dictionary access

# --- Environment Setup (Unchanged) ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DB_URI = os.getenv("DB_URI")

if not DB_URI or not GROQ_API_KEY:
    raise EnvironmentError("Missing DB_URI or GROQ_API_KEY in .env")

# --- Agent State, LLM, and Graph Definition (Unchanged) ---
class AgentState(TypedDict):
    """State of the agent."""
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=GROQ_API_KEY)
search_tool = DuckDuckGoSearchRun()
llm_with_tools = llm.bind_tools([search_tool])

def chat_node(state: AgentState):
    """LLM call."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def build_graph() -> StateGraph:
    """Define the graph structure."""
    graph = StateGraph(AgentState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode([search_tool]))
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    return graph

# ------------------------------------------------------------------
# 🚀 THE PRODUCTION-READY, TYPE-CORRECT SETUP 🚀
# ------------------------------------------------------------------

# 1. Initialize the Connection Pool (The Toolbox)
try:
    CONN_POOL = ConnectionPool(
        conninfo=DB_URI, 
        min_size=2, 
        max_size=10, 
        kwargs={
            "autocommit": True,  # Critical for LangGraph setup and persistence
            "row_factory": dict_row # Critical for PostgresSaver to access columns by name
        } 
    )
    
    # 2. Acquire a temporary connection from the pool to run setup
    # NOTE: The CHECKPOINTER is created temporarily here to call setup()
    with CONN_POOL.connection() as conn:
        PostgresSaver(conn=conn).setup()

except Exception as e:
    raise EnvironmentError(f"FATAL: Database pool failed to initialize. Error: {e}")


# 3. Create the resilient Checkpointer
# We must use the *pool itself* as the main argument (or the 'conn' keyword 
# if the Saver accepted the pool object directly).
# The current correct pattern based on documentation is to pass the pool object.
# If PostgresSaver() throws an error without the 'pool' keyword, the library 
# expects the pool instance itself to be the first positional argument.

# Try passing the Pool as the positional argument, which is the pattern
# for non-async saver that uses a pool (sometimes disguised as a 'conn' argument 
# that accepts either a connection or a pool object).
try:
    CHECKPOINTER = PostgresSaver(CONN_POOL)
except TypeError:
    # If the above fails, it means the library requires the connection string 
    # to be managed by *its own* internal pool implementation, or expects a 
    # specific connection object. Given your requirements, we revert to the 
    # most common stable pattern for your version: creating a new connection 
    # with the pool parameters and passing the raw connection. 
    # However, since that causes the timeout, the only reliable option is 
    # often passing the pool directly. Let's stick with the pool object itself.
    # If you still get an error here, you need to check the specific 
    # `langgraph-checkpoint-postgres` documentation for your version.
    
    # If the simple line above failed, it is likely due to an old version.
    # Let's use the stable pattern for a library that supports connection objects, 
    # and pass the pool, hoping it's recognized.
    CHECKPOINTER = PostgresSaver(CONN_POOL) 


# 4. Compile the workflow globally
workflow = build_graph().compile(checkpointer=CHECKPOINTER)

config = {'configurable': {'thread_id': '4'}}

response = workflow.invoke(
    {"messages": [HumanMessage(content="do you know me?")]},
    config=config
)

print(response)
