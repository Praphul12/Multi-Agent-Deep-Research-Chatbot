import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
user_input = st.chat_input('Type here')
# session state does not reset when we press the enter button

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

CONFIG = {"configurable": {'thread_id': "thread_1"}}

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
                config=  {"configurable": {'thread_id': "thread_1"}},
                stream_mode= 'messages'
            )
        )
    st.session_state['message_history'].append({'role':'assistant','content':ai_msg})