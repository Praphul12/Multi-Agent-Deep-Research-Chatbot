# backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated,Dict,Any,Optional
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import sqlite3
import requests
import tempfile
import os
load_dotenv()

# -------------------
# 1. LLM
# -------------------
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model= 'text-embedding-3-small')

#-----------------
# PDF retriever store per thread

_THREAD_RETRIEVERS_: Dict[str,Any] = {}
_THREAD_METADATA_: Dict[str,Dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETRIEVERS_:
        return _THREAD_RETRIEVERS_[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """
        Build a FAISS retriever for a particualar thread and then store the 
        retriever aganist the thread_id
    """
    if not file_bytes:
        raise ValueError('No bytes received for ingestion')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
        try:
            loader = PyPDFLoader(file_path=temp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size = 300,
                chunk_overlap = 40,
                separators=["\n\n","\n"," ",""]
            )
            docs_split = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(docs_split,embeddings)
            
            retriever = vector_store.as_retriever(
                search_type = "similarity",
                search_kwargs = {'k':4}
            )
            _THREAD_RETRIEVERS_[str(thread_id)] = retriever
            _THREAD_METADATA_[str(thread_id)] = {
                "filename": filename or os.path.basename(temp_path),
                "documents": len(docs),
                "chunks": len(docs_split)
            }
            
            return {
                "filename": filename or os.path.basename(temp_path),
                "documents": len(docs),
                "chunks": len(docs_split)
            }
            
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            
            
# -------------------
# 2. Tools
# -------------------
# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query:str,thread_id: Optional[str] = None)-> dict:
    """
    Retrieve relevant information from the uploaded pdf for this current chat thread
    Always include the thread_id when calling this tool
    """
    
    retriever = _get_retriever(thread_id)
    
    if retriever is None:
        return{
            'error': 'No document indexes for this chat.Upload a PDF first',
            "query": query
        }
    result = retriever.invoke(query)
    context = [docs.page_content for docs in result]
    metadata = [docs.metadata for docs in result]
    
    return{
        "query":query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA_.get(str(thread_id),{}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator,rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState,config = None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable",{}).get("thread_id")
    system_message = SystemMessage(
       content=(
           "You are a helpful assistant. For questions about the uploaded PDF, call "
           "the `rag_tool` and include the thread_id "
           f"`{thread_id}`. You can also use the web search, stock price, and "
           "calculator tools when helpful. If no document is available, ask the user "
           "to upload a PDF."
       )
    )
    messages = [system_message,*state["messages"]]
    response = llm_with_tools.invoke(messages,config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS_


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA_.get(str(thread_id), {})