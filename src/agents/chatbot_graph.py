import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver 
from psycopg_pool import ConnectionPool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import tools_condition, ToolNode
import streamlit as st
from psycopg.rows import dict_row 


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


# 1️⃣ Create ONE global connection pool
CONN_POOL = ConnectionPool(
    conninfo=DB_URI,
    min_size=2,
    max_size=10,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row
    }
)

# 2️⃣ Create ONE checkpointer from the pool
CHECKPOINTER = PostgresSaver(CONN_POOL)

# 3️⃣ Run setup ONCE
CHECKPOINTER.setup()

# 4. Compile the workflow globally
workflow = build_graph().compile(checkpointer=CHECKPOINTER)

#onfig = {'configurable': {'thread_id': '4'}}

#sponse = workflow.invoke(
#   {"messages": [HumanMessage(content="do you know me?")]},
#   config=config
#

#rint(response)



#def retrieve_all_threads():
#    all_threads = set()
#    for checkpoint in checkpointer.list(None):
#        all_threads.add(checkpoint.config['configurable']['thread_id'])
#    print(list(all_threads))
