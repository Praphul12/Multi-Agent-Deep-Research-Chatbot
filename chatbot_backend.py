# %%
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from typing import TypedDict,Annotated

# %%
load_dotenv()

# %%
llm = HuggingFaceEndpoint(
    model= "meta-llama/Llama-3.1-8B-Instruct"
)

# %%
model = ChatHuggingFace(llm = llm)

# %%
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage],add_messages]

# %%
conn = sqlite3.connect('checkpoint.db',check_same_thread= False)
checkpoint = SqliteSaver(conn)

# %%
def chat_node(state: chatState):
    msg = state['messages']
    response = model.invoke(msg)
    return {'messages': [response]}

# %%
graph = StateGraph(chatState)
graph.add_node('chatNode',chat_node)
graph.add_edge(START,'chatNode')
graph.add_edge('chatNode',END)

chatbot = graph.compile(checkpointer= checkpoint)

# CONFIG=  {"configurable": {'thread_id': 'thread1'}} 
# result = chatbot.invoke({'messages':'My name is praphul. I am 27 years old'},config= CONFIG)
# res = chatbot.get_state(config={'configurable':{'thread_id': 'thread1'}})

checkpoin ters = checkpoint.list(None)

def get_unique_threads():
    checkpoints = set()
    for checkpoint in checkpointers:
        checkpoints.add(checkpoint.config['configurable']['thread_id'])

    return list(checkpoints)