import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
import streamlit as st
from psycopg import connect

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
DB_URI = os.getenv("DB_URI")


llm = ChatGroq(model = "openai/gpt-oss-120b", api_key = groq_api_key)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@st.cache_resource
def init_checkpointer():
    # Step 1: Create DB tables if not exist
    with PostgresSaver.from_conn_string(DB_URI) as setup_saver:
        setup_saver.setup()
    # Step 2: Create a persistent connection
    conn = connect(DB_URI)
    return PostgresSaver(conn)  # persistent checkpointer

checkpointer = init_checkpointer()

#checkpointer = PostgresSaver.from_conn_string(DB_URI).__enter__()
#heckpointer.setup()

def chat_node(state: AgentState): 
    response = llm.invoke(state['messages'])
    return {'messages': [response]}



graph = StateGraph(AgentState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)

#response = chatbot.invoke({'messages': HumanMessage(content="tell me about yourself")}, config = {'configurable':{'thread_id': 'init_thread'}})

#print(response)
def retrieve_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None):
        all_thread.add(checkpoint.config['configurable']['thread_id'])
    return list(all_thread)