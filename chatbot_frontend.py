import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])
    
    # Filter to show only user messages and final AI responses (with content, no tool calls)
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            # Only add AI messages that have actual content and aren't tool calls
            temp_messages.append({"role": "assistant", "content": msg.content})
    
    return temp_messages

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
    
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}
    
add_thread(st.session_state["thread_id"])

# Get current thread info
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")

# Show current thread ID
st.sidebar.markdown(f"**Current Thread:** `{thread_key[:8]}...`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# PDF Upload Section
st.sidebar.header("📄 Document Management")

# Show current document status
if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"✅ **{latest_doc.get('filename')}**\n\n"
        f"📊 {latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages"
    )
else:
    st.sidebar.info("No PDF indexed for this chat yet.")

# File uploader
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat", 
    type=["pdf"],
    help="Upload a PDF to enable document-based Q&A"
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"✓ `{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("🔄 Indexing PDF…", expanded=True) as status_box:
            try:
                summary = ingest_pdf(
                    uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name,
                )
                thread_docs[uploaded_pdf.name] = summary
                status_box.update(
                    label=f"✅ PDF indexed successfully!", 
                    state="complete", 
                    expanded=False
                )
                st.rerun()
            except Exception as e:
                status_box.update(
                    label=f"❌ Error indexing PDF", 
                    state="error", 
                    expanded=True
                )
                st.sidebar.error(f"Error: {str(e)}")

# Conversation history
st.sidebar.header("💬 My Conversations")
for i, thread_id in enumerate(st.session_state["chat_threads"][::-1], 1):
    # Show if thread has a document
    doc_meta = thread_document_metadata(str(thread_id))
    label = f"Chat {i}"
    if doc_meta:
        label += f" 📄"
    
    if st.sidebar.button(label, key=thread_id, use_container_width=True):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()

# ============================ Main UI ============================
st.title("🤖 Multi-Utility Chatbot")

# Show document info in main area if available
doc_meta = thread_document_metadata(thread_key)
if doc_meta:
    st.info(
        f"📄 **Active Document:** {doc_meta.get('filename')} "
        f"({doc_meta.get('chunks')} chunks, {doc_meta.get('documents')} pages)"
    )

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask about your document or use other tools...")

if user_input:
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}
        collected_content = []  # Collect all content chunks

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Handle tool usage indicators
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant content (not tool calls)
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    collected_content.append(message_chunk.content)
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize tool status if used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save the complete assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )