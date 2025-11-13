import streamlit as st
from src.agents.chatbot_graph import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

# ************************ utility functions *********************************

def generate_thread_id():
    """Generate a unique thread ID."""
    return uuid.uuid4()

def add_thread(thread_id, title="New Chat"):
    """Add a thread ID with a friendly title."""
    if 'chat_thread' not in st.session_state:
        st.session_state['chat_thread'] = []
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)
    if 'thread_titles' not in st.session_state:
        st.session_state['thread_titles'] = {}
    st.session_state['thread_titles'][thread_id] = title

def reset_chat():
    """Start a new chat thread."""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []

def load_conversation(thread_id):
    """Load messages from LangGraph checkpointer."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])

    # Set a friendly title using the first user message
    if messages:
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                st.session_state['thread_titles'][thread_id] = msg.content[:20]  # first 20 chars
                break

    return messages

# ************************ session state *************************************

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if "chat_thread" not in st.session_state:
    threads = retrieve_all_threads() or []
    st.session_state['chat_thread'] = threads

add_thread(st.session_state['thread_id'])

# ************************* Sidebar ******************************************

st.sidebar.title("LangGraph Chat")
if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")
for thread_id in st.session_state['chat_thread'][::-1]:
    title = st.session_state['thread_titles'].get(thread_id, str(thread_id))
    if st.sidebar.button(title):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if hasattr(msg, "type") and msg.type == "human" else "assistant"
            temp_messages.append({'role': role, "content": msg.content})

        st.session_state['message_history'] = temp_messages

# *********************** Display messages ***********************************

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# *********************** User input and AI response *************************

user_input = st.chat_input("Type here...")

if user_input:
    # Add user message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # Stream AI response
    assistant_chunks = []
    with st.chat_message('assistant'):
        for chunk, metadata in chatbot.stream(
            {'messages': HumanMessage(content=user_input)},
            config={'configurable': {"thread_id": st.session_state['thread_id']}},
            stream_mode="messages"
        ):
            st.write(chunk.content)
            assistant_chunks.append(chunk.content)

    ai_message = "".join(assistant_chunks)
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
