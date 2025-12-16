import streamlit as st
from src.agents.chatbot_graph import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid

#************************ utility function *********************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id 
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_thread']:
        st.session_state['chat_thread'].append(thread_id)

#def load_conversation(thread_id):
    #return chatbot.get_state(config = {'configurable':{'thread_id': thread_id}}).values['messages']

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # use .get to avoid KeyError
    return state.values.get('messages', [])



#************************ session state **************************************  1   

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

#if "chat_thread" not in st.session_state:
#   st.session_state['chat_thread'] = retrieve_all_threads()

if "chat_thread" not in st.session_state:
    st.session_state['chat_thread'] = retrieve_all_threads() or []


add_thread(st.session_state['thread_id'])


#*********************** SideBar ******************************************
st.sidebar.title("LangGraph")
if st.sidebar.button("New Chat"):
    reset_chat()
st.sidebar.header("My Conversation")

for thread_id in st.session_state['chat_thread'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({'role': role, "content": msg.content})
        
        st.session_state['message_history'] = temp_messages
            


#*************************** user_input ---- model response ***************************

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here...")

if user_input:
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #response = chatbot.invoke({'messages': HumanMessage(content=user_input)}, config = config)
    #ai_message = response['messages'][-1].content
    
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': HumanMessage(content = user_input)},
                config = {'configurable': {"thread_id" : st.session_state['thread_id']}},
                stream_mode = "messages"
            )
        )
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
