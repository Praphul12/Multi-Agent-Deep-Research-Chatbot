import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads, ingest_pdf, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid
from datetime import date
from pathlib import Path
import re
import zipfile
from io import BytesIO
import json
from typing import Dict, Any, List, Tuple, Iterator, Optional
import pandas as pd

# Import blog writer
try:
    from bwa_backend import app as blog_app
    BLOG_WRITER_AVAILABLE = True
except ImportError:
    BLOG_WRITER_AVAILABLE = False
    st.error("Blog writer backend not available. Make sure bwa_backend.py is in the same directory.")

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
    
    # Filter to show only user messages and final AI responses
    temp_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            temp_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            temp_messages.append({"role": "assistant", "content": msg.content})
    
    return temp_messages

# Blog writer utilities
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()

def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Stream graph progress if available; else invoke."""
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    
    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception:
        pass
    
    out = graph_app.invoke(inputs)
    yield ("final", out)

def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            inner = next(iter(step_payload.values()))
            current_state.update(inner)
        else:
            current_state.update(step_payload)
    return current_state

_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")

def _resolve_image_path(src: str) -> Path:
    src = src.strip().lstrip("./")
    return Path(src).resolve()

def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        alt = (m.group("alt") or "").strip()
        src = (m.group("src") or "").strip()
        parts.append(("img", f"{alt}|||{src}"))
        last = m.end()

    tail = md[last:]
    if tail:
        parts.append(("md", tail))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    rest = "\n".join(nxt.splitlines()[1:])
                    parts[i + 1] = ("md", rest)

        if src.startswith("http://") or src.startswith("https://"):
            st.image(src, caption=caption or (alt or None), width='content')
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), width='content')
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1

# ======================= Session Initialization ===================
if "app_mode" not in st.session_state:
    st.session_state["app_mode"] = "chat"

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()
    
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "last_blog_out" not in st.session_state:
    st.session_state["last_blog_out"] = None

if "blog_logs" not in st.session_state:
    st.session_state["blog_logs"] = []
    
add_thread(st.session_state["thread_id"])

# Get current thread info
thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})

# ============================ Page Config ============================
st.set_page_config(page_title="Multi-Agent Assistant", layout="wide", page_icon="🤖")

# ============================ Sidebar ============================
st.sidebar.title("🤖 Multi-Agent Assistant")

# Mode Selector
mode = st.sidebar.radio(
    "Select Mode:",
    ["💬 Chat Mode", "📝 Research Mode"],
    key="mode_selector",
    help="Chat Mode: Talk with AI and analyze PDFs\nResearch Mode: Generate in-depth blog posts"
)

if mode == "💬 Chat Mode":
    st.session_state["app_mode"] = "chat"
else:
    st.session_state["app_mode"] = "research"

st.sidebar.divider()

# ============================ CHAT MODE UI ============================
if st.session_state["app_mode"] == "chat":
    # Chat sidebar controls
    st.sidebar.markdown(f"**Current Thread:** `{thread_key[:8]}...`")

    if st.sidebar.button("New Chat", width='content'):
        reset_chat()
        st.rerun()

    # PDF Upload Section
    st.sidebar.header("📄 Document Management")

    if thread_docs:
        latest_doc = list(thread_docs.values())[-1]
        st.sidebar.success(
            f"✅ **{latest_doc.get('filename')}**\n\n"
            f"📊 {latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages"
        )
    else:
        st.sidebar.info("No PDF indexed for this chat yet.")

    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF for this chat", 
        type=["pdf"],
        help="Upload a PDF to enable document-based Q&A"
    )

    if uploaded_pdf:
        if uploaded_pdf.name in thread_docs:
            st.sidebar.info(f"✓ `{uploaded_pdf.name}` already processed.")
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
                        label="✅ PDF indexed!", 
                        state="complete", 
                        expanded=False
                    )
                    st.rerun()
                except Exception as e:
                    status_box.update(
                        label="❌ Error indexing PDF", 
                        state="error", 
                        expanded=True
                    )
                    st.sidebar.error(f"Error: {str(e)}")

    # Conversation history
    st.sidebar.header("💬 My Conversations")
    for i, thread_id in enumerate(st.session_state["chat_threads"][::-1], 1):
        doc_meta = thread_document_metadata(str(thread_id))
        label = f"Chat {i}"
        if doc_meta:
            label += " 📄"
        
        if st.sidebar.button(label, key=f"thread_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(thread_id)
            st.rerun()

    # Main Chat UI
    st.title("🤖 Multi-Utility Chatbot")

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
        st.session_state["message_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        CONFIG = {
            "configurable": {"thread_id": st.session_state["thread_id"]},
            "metadata": {"thread_id": st.session_state["thread_id"]},
            "run_name": "chat_turn",
        }

        with st.chat_message("assistant"):
            status_holder = {"box": None}
            collected_content = []

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
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

                    if isinstance(message_chunk, AIMessage) and message_chunk.content:
                        collected_content.append(message_chunk.content)
                        yield message_chunk.content

            ai_message = st.write_stream(ai_only_stream())

            if status_holder["box"] is not None:
                status_holder["box"].update(
                    label="✅ Tool finished", state="complete", expanded=False
                )

        st.session_state["message_history"].append(
            {"role": "assistant", "content": ai_message}
        )

# ============================ RESEARCH MODE UI ============================
else:
    if not BLOG_WRITER_AVAILABLE:
        st.error("Blog writer backend not available.")
        st.stop()

    st.title("📝 Deep Research Agent")
    st.caption("Generate comprehensive, well-researched blog posts on any topic")

    # Research controls in sidebar
    st.sidebar.header("🔬 Research Settings")
    topic = st.sidebar.text_area(
        "Research Topic",
        height=120,
        placeholder="Enter a topic to research and write about..."
    )
    as_of = st.sidebar.date_input("As-of date", value=date.today())
    
    run_btn = st.sidebar.button("🚀 Generate Research Blog", type="primary", use_container_width=True)

    # Main area with tabs
    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
        ["🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
    )

    logs: List[str] = []

    def log(msg: str):
        logs.append(msg)

    if run_btn:
        if not topic.strip():
            st.warning("Please enter a research topic.")
            st.stop()

        inputs: Dict[str, Any] = {
            "topic": topic.strip(),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "as_of": as_of.isoformat(),
            "recency_days": 7,
            "sections": [],
            "merged_md": "",
            "md_with_placeholders": "",
            "image_specs": [],
            "final": "",
        }

        status = st.status("Running research agent…", expanded=True)
        progress_area = st.empty()

        current_state: Dict[str, Any] = {}
        last_node = None

        for kind, payload in try_stream(blog_app, inputs):
            if kind in ("updates", "values"):
                node_name = None
                if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                    node_name = next(iter(payload.keys()))
                if node_name and node_name != last_node:
                    status.write(f"➡️ Node: `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "queries": current_state.get("queries", [])[:5] if isinstance(current_state.get("queries"), list) else [],
                    "evidence_count": len(current_state.get("evidence", []) or []),
                    "tasks": len((current_state.get("plan") or {}).get("tasks", [])) if isinstance(current_state.get("plan"), dict) else None,
                    "images": len(current_state.get("image_specs", []) or []),
                    "sections_done": len(current_state.get("sections", []) or []),
                }
                progress_area.json(summary)

                log(f"[{kind}] {json.dumps(payload, default=str)[:1200]}")

            elif kind == "final":
                out = payload
                st.session_state["last_blog_out"] = out
                status.update(label="✅ Research Complete", state="complete", expanded=False)
                log("[final] received final state")

    # Render last result
    out = st.session_state.get("last_blog_out")
    if out:
        with tab_plan:
            st.subheader("Research Plan")
            plan_obj = out.get("plan")
            if not plan_obj:
                st.info("No plan found in output.")
            else:
                if hasattr(plan_obj, "model_dump"):
                    plan_dict = plan_obj.model_dump()
                elif isinstance(plan_obj, dict):
                    plan_dict = plan_obj
                else:
                    plan_dict = json.loads(json.dumps(plan_obj, default=str))

                st.write("**Title:**", plan_dict.get("blog_title"))
                cols = st.columns(3)
                cols[0].write("**Audience:** " + str(plan_dict.get("audience")))
                cols[1].write("**Tone:** " + str(plan_dict.get("tone")))
                cols[2].write("**Type:** " + str(plan_dict.get("blog_kind", "")))

                tasks = plan_dict.get("tasks", [])
                if tasks:
                    df = pd.DataFrame(
                        [
                            {
                                "id": t.get("id"),
                                "title": t.get("title"),
                                "target_words": t.get("target_words"),
                                "requires_research": t.get("requires_research"),
                                "requires_citations": t.get("requires_citations"),
                            }
                            for t in tasks
                        ]
                    ).sort_values("id")
                    st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_evidence:
            st.subheader("Research Evidence")
            evidence = out.get("evidence") or []
            if not evidence:
                st.info("No evidence collected (closed-book mode).")
            else:
                rows = []
                for e in evidence:
                    if hasattr(e, "model_dump"):
                        e = e.model_dump()
                    rows.append(
                        {
                            "title": e.get("title"),
                            "published_at": e.get("published_at"),
                            "source": e.get("source"),
                            "url": e.get("url"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), width='content', hide_index=True)

        with tab_preview:
            st.subheader("Blog Preview")
            final_md = out.get("final") or ""
            if not final_md:
                st.warning("No final markdown generated yet.")
            else:
                render_markdown_with_local_images(final_md)

                plan_obj = out.get("plan")
                if hasattr(plan_obj, "blog_title"):
                    blog_title = plan_obj.blog_title
                elif isinstance(plan_obj, dict):
                    blog_title = plan_obj.get("blog_title", "blog")
                else:
                    blog_title = "blog"

                md_filename = f"{safe_slug(blog_title)}.md"
                st.download_button(
                    "⬇️ Download Markdown",
                    data=final_md.encode("utf-8"),
                    file_name=md_filename,
                    mime="text/markdown",
                )

                bundle = bundle_zip(final_md, md_filename, Path("images"))
                st.download_button(
                    "📦 Download Bundle (MD + images)",
                    data=bundle,
                    file_name=f"{safe_slug(blog_title)}_bundle.zip",
                    mime="application/zip",
                )

        with tab_images:
            st.subheader("Generated Images")
            specs = out.get("image_specs") or []
            images_dir = Path("images")

            if not specs and not images_dir.exists():
                st.info("No images generated for this research.")
            else:
                if specs:
                    st.write("**Image specifications:**")
                    st.json(specs)

                if images_dir.exists():
                    files = [p for p in images_dir.iterdir() if p.is_file()]
                    if files:
                        for p in sorted(files):
                            st.image(str(p), caption=p.name, width='content')

        with tab_logs:
            st.subheader("Agent Logs")
            if logs:
                st.session_state["blog_logs"].extend(logs)
            st.text_area("Event log", value="\n\n".join(st.session_state["blog_logs"][-80:]), height=520)
    else:
        st.info("Enter a research topic in the sidebar and click **Generate Research Blog** to begin.")