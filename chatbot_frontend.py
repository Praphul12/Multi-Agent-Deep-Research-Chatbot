import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

#*****************************Utilities**************************#

def generate_id():
   thread_id = uuid.uuid4()
   return thread_id


def add_thread(thread_id):
    if thread_id not in st.session_state['thread_history']:
        st.session_state['thread_history'].append(thread_id)

def reset_chat():
    thread_id = generate_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    
 #******************************State management**********************#
 

# session state does not reset when we press the enter button

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []
    

if 'thread_history' not in st.session_state:
    st.session_state['thread_history'] = []
    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_id()
    add_thread(st.session_state['thread_id'])




user_input = st.chat_input('Type here')

CONFIG=  {"configurable": {'thread_id': st.session_state['thread_id']}} 

#********************************Sidebar****************************#


st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New chat'):
    reset_chat()
st.sidebar.header('My conversations')


for thread in st.session_state['thread_history']:
    st.sidebar.text(thread)
    
    
#****************************** Conversation history****************#
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
    
if(user_input):
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    # response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]},config= CONFIG)
    # ai_msg = response['messages'][-1].content
    
    with st.chat_message('assistant'):
        ai_msg = st.write_stream(
            message_chunk for message_chunk,metadata in chatbot.stream(
                input={'messages': [HumanMessage(content=user_input)]},
                config=  CONFIG,
                stream_mode= 'messages'
            )
        )
    st.session_state['message_history'].append({'role':'assistant','content':ai_msg})