# %%
from langgraph.graph import StateGraph, START,END
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
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
checkpoint = InMemorySaver()

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
# result = chatbot.invoke({'messages':'hi there'},config= CONFIG)
# res = chatbot.get_state(config={'configurable':{'thread_id': 'thread1'}})

# print(res.values['messages'])


