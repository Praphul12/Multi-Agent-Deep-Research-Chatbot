# chatbot_backend_enhanced.py - CORRECTED VERSION WITH WORKING IMAGE GENERATION

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Dict, Any, Optional, List, Literal
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun, TavilySearchResults
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt, Command, Send
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field
from pathlib import Path
import sqlite3
import requests
import tempfile
import os
import json
from datetime import datetime
import operator
import base64
import re
load_dotenv()

# =================== EXISTING CHATBOT CODE ===================
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# PDF retriever store per thread
_THREAD_RETRIEVERS_: Dict[str, Any] = {}
_THREAD_METADATA_: Dict[str, Dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available"""
    if thread_id and thread_id in _THREAD_RETRIEVERS_:
        return _THREAD_RETRIEVERS_[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None):
    """Build a FAISS retriever for a particular thread"""
    if not file_bytes:
        raise ValueError('No bytes received for ingestion')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
        
        try:
            loader = PyPDFLoader(file_path=temp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=40,
                separators=["\n\n", "\n", " ", ""]
            )
            docs_split = splitter.split_documents(docs)
            vector_store = FAISS.from_documents(docs_split, embeddings)
            
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 4}
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

# Tools
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers."""
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
    """Fetch latest stock price for a given symbol."""
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int, total_price: int) -> dict:
    """Simulate purchasing stock with human approval."""
    decision = interrupt(
        f"Approve before buying {quantity} shares of {symbol} for ${total_price}"
    )
    
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f'Purchase order placed for {quantity} shares of {symbol}',
            "symbol": symbol,
            "quantity": quantity,
            "total_price": total_price
        }
    else:
        return {
            "status": "cancelled",
            "message": f'Purchase order for {quantity} shares of {symbol} declined by human'
        }

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """Retrieve relevant information from the uploaded PDF."""
    retriever = _get_retriever(thread_id)
    
    if retriever is None:
        return {
            'error': 'No document indexed for this chat. Upload a PDF first',
            "query": query
        }
    result = retriever.invoke(query)
    context = [docs.page_content for docs in result]
    metadata = [docs.metadata for docs in result]
    
    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA_.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool, purchase_stock]
llm_with_tools = llm.bind_tools(tools)

# Chat State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and purchase stock "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )
    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Checkpointer
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS_

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA_.get(str(thread_id), {})

def check_for_interrupt(thread_id: str) -> dict:
    """Check if the graph is currently interrupted."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config=config)
    
    if state.next and len(state.next) > 0:
        if state.tasks:
            for task in state.tasks:
                if hasattr(task, 'interrupts') and task.interrupts:
                    return {
                        'pending': True,
                        "question": task.interrupts[0].value,
                        "task": task
                    }
    return {'pending': False}

def resume_with_decision(thread_id: str, decision: str):
    """Resume the interrupted graph with human decision."""
    config = {"configurable": {"thread_id": thread_id}}
    result = chatbot.invoke(Command(resume=decision), config=config)
    return result


# =================== BLOG GENERATOR CODE ===================

# Schemas
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="What the reader should understand after this section.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False

class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]

class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None

class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book", "rag_grounded"]
    queries: List[str] = Field(default_factory=list)

class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)

class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"

class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class BlogState(TypedDict):
    topic: str
    thread_id: Optional[str]
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    rag_context: List[str]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str
    progress: str

# Blog generation nodes
ROUTER_SYSTEM =  """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def blog_router_node(state: BlogState) -> dict:
    topic = state["topic"]
    thread_id = state.get("thread_id")
    
    # Check if we have RAG context available
    has_rag = thread_id and thread_has_document(str(thread_id))
    
    router_hint = ""
    if has_rag:
        router_hint = "\n\nNote: User has uploaded a document. Consider 'rag_grounded' mode to use document as primary source."
    
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}{router_hint}"),
    ])

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "progress": f"✓ Routing: {decision.mode} mode"
    }

def blog_route_next(state: BlogState) -> str:
    mode = state.get("mode", "")
    thread_id = state.get("thread_id")
    
    # If RAG grounded mode and we have a document, go to RAG first
    if mode == "rag_grounded" and thread_id and thread_has_document(str(thread_id)):
        return "rag_retrieval"
    # Otherwise check if we need web research
    elif state.get("needs_research"):
        return "research"
    else:
        return "orchestrator"

def rag_retrieval_node(state: BlogState) -> dict:
    """Retrieve context from uploaded PDF for blog generation."""
    thread_id = state.get("thread_id")
    topic = state["topic"]
    
    if not thread_id or not thread_has_document(str(thread_id)):
        return {"rag_context": [], "progress": "⚠ No document available"}
    
    retriever = _get_retriever(str(thread_id))
    
    # Generate multiple queries to get comprehensive context
    queries = [
        topic,
        f"key concepts related to {topic}",
        f"examples and details about {topic}",
    ]
    
    all_context = []
    seen = set()
    
    for query in queries:
        results = retriever.invoke(query)
        for doc in results:
            content = doc.page_content
            if content not in seen:
                seen.add(content)
                all_context.append(content)
    
    return {
        "rag_context": all_context[:15],  # Top 15 chunks
        "progress": f"✓ Retrieved {len(all_context)} relevant chunks from document"
    }

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})
    
    normalized: List[dict] = []
    for r in results or []:
        normalized.append({
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "snippet": r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source": r.get("source"),
        })
    return normalized

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.
Produce a deduplicated list of EvidenceItem objects from web search results.
Only include items with non-empty URLs. Prefer authoritative sources."""

def blog_research_node(state: BlogState) -> dict:
    queries = state.get("queries", [])[:10]
    max_results = 6
    
    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))
    
    if not raw_results:
        return {"evidence": [], "progress": "⚠ No research results found"}
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])
    
    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    
    evidence_list = list(dedup.values())
    return {
        "evidence": evidence_list,
        "progress": f"✓ Researched {len(evidence_list)} sources"
    }

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def blog_orchestrator_node(state: BlogState) -> dict:
    planner = llm.with_structured_output(Plan)
    
    evidence = state.get("evidence", [])
    rag_context = state.get("rag_context", [])
    mode = state.get("mode", "closed_book")
    
    context_info = ""
    if rag_context:
        context_info = f"\n\nDocument Context (PRIMARY SOURCE):\n" + "\n---\n".join(rag_context[:10])
    
    evidence_info = ""
    if evidence:
        evidence_info = f"\n\nWeb Evidence (SECONDARY):\n" + str([e.model_dump() for e in evidence[:12]])
    
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"{context_info}"
            f"{evidence_info}"
        )),
    ])

    return {"plan": plan, "progress": f"✓ Created outline with {len(plan.tasks)} sections"}

def blog_fanout(state: BlogState):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "rag_context": state.get("rag_context", []),
        })
        for task in state["plan"].tasks
    ]

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def blog_worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    rag_context = payload.get("rag_context", [])
    mode = payload.get("mode", "closed_book")
    
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    
    rag_text = ""
    if rag_context:
        rag_text = "\n\nDocument Excerpts (PRIMARY):\n" + "\n---\n".join(rag_context[:8])
    
    evidence_text = ""
    if evidence:
        evidence_text = "\n\nWeb Sources:\n" + "\n".join(
            f"- {e.title} | {e.url}" for e in evidence[:15]
        )
    
    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Mode: {mode}\n\n"
            f"Section: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Bullets:{bullets_text}\n"
            f"{rag_text}"
            f"{evidence_text}"
        )),
    ]).content.strip()
    
    return {"sections": [(task.id, section_md)]}

def blog_merge_content(state: BlogState) -> dict:
    """Merge sections without images (images handled in next node)"""
    plan = state["plan"]
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    
    return {
        "merged_md": merged_md,
        "progress": "✓ Merged all sections"
    }

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images_node(state: BlogState) -> dict:
    """Decide where to place images and generate prompts"""
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Blog kind: {plan.blog_kind}\n"
            f"Topic: {state['topic']}\n\n"
            "Insert placeholders + propose image prompts.\n\n"
            f"{merged_md}"
        )),
    ])

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
        "progress": f"✓ Planned {len(image_plan.images)} images"
    }

# ============ CORRECTED GEMINI IMAGE GENERATION ============
def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Returns raw image bytes generated by Gemini.
    Requires: pip install google-genai
    Env var: GOOGLE_API_KEY
    
    IMPORTANT: Uses correct model name and import structure from working code
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    # CORRECTED: Use the working model name from your code
    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",  # Changed from gemini-2.5-flash-exp
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Extract image bytes from response (using working code's logic)
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")

# ============ FIXED IMAGE GENERATION WITH BASE64 ============

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images_node(state: BlogState) -> dict:
    """Generate images and embed them as base64 in markdown for Streamlit compatibility"""
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Add metadata footer
    mode_label = state.get("mode", "unknown")
    rag_used = "✓" if state.get("rag_context") else "✗"
    web_used = "✓" if state.get("evidence") else "✗"
    footer = f"\n\n---\n\n*Generated with: {mode_label} mode | Document: {rag_used} | Web: {web_used}*"

    # If no images requested, just finalize
    if not image_specs:
        final_md = md + footer
        return {
            "final": final_md,
            "progress": "✓ Blog complete (no images)"
        }

    # Create images directory (for backup storage)
    images_dir = BLOGS_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    generated_count = 0
    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # Generate image if needed
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
                generated_count += 1
            except Exception as e:
                # Graceful fallback: keep doc usable
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {str(e)}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        # ============ THIS IS THE KEY FIX ============
        try:
            # Read the generated image
            img_bytes = out_path.read_bytes()
            
            # Convert to base64
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            
            # Create inline data URI (THIS is what makes it work in Streamlit!)
            img_md = f"![{spec['alt']}](data:image/png;base64,{b64_img})\n*{spec.get('caption', '')}*"
            md = md.replace(placeholder, img_md)
            
        except Exception as e:
            # Fallback: show error but keep blog functional
            error_block = (
                f"> **[IMAGE ENCODING FAILED]** {spec.get('caption', '')}\n>\n"
                f"> **Alt:** {spec.get('alt', '')}\n>\n"
                f"> **Error:** {str(e)}\n>\n"
                f"> Image saved to: `{out_path}`\n"
            )
            md = md.replace(placeholder, error_block)

    final_md = md + footer
    return {
        "final": final_md,
        "progress": f"✓ Generated {generated_count} images, blog complete!"
    }


# Build blog reducer graph with image generation
blog_reducer_graph = StateGraph(BlogState)
blog_reducer_graph.add_node("merge", blog_merge_content)
blog_reducer_graph.add_node("decide_images", decide_images_node)
blog_reducer_graph.add_node("generate_images", generate_and_place_images_node)

blog_reducer_graph.add_edge(START, "merge")
blog_reducer_graph.add_edge("merge", "decide_images")
blog_reducer_graph.add_edge("decide_images", "generate_images")
blog_reducer_graph.add_edge("generate_images", END)

blog_reducer = blog_reducer_graph.compile()

blog_graph = StateGraph(BlogState)
blog_graph.add_node("router", blog_router_node)
blog_graph.add_node("rag_retrieval", rag_retrieval_node)
blog_graph.add_node("research", blog_research_node)
blog_graph.add_node("orchestrator", blog_orchestrator_node)
blog_graph.add_node("worker", blog_worker_node)
blog_graph.add_node("reducer", blog_reducer)

blog_graph.add_edge(START, "router")
blog_graph.add_conditional_edges("router", blog_route_next, {
    "rag_retrieval": "rag_retrieval",
    "research": "research",
    "orchestrator": "orchestrator"
})
blog_graph.add_edge("rag_retrieval", "orchestrator")
blog_graph.add_edge("research", "orchestrator")
blog_graph.add_conditional_edges("orchestrator", blog_fanout, ["worker"])
blog_graph.add_edge("worker", "reducer")
blog_graph.add_edge("reducer", END)

blog_generator = blog_graph.compile()


# =================== BLOG STORAGE ===================

BLOGS_DIR = Path("generated_blogs")
BLOGS_DIR.mkdir(exist_ok=True)

def save_blog(blog_content: str, title: str, thread_id: Optional[str] = None) -> dict:
    """Save a generated blog to disk."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_title = safe_title.replace(' ', '_')[:50]
    
    filename = f"{timestamp}_{safe_title}.md"
    filepath = BLOGS_DIR / filename
    
    # Add metadata header
    metadata = {
        "title": title,
        "generated_at": datetime.now().isoformat(),
        "thread_id": thread_id,
    }
    
    content_with_meta = f"<!--- {json.dumps(metadata)} --->\n\n{blog_content}"
    filepath.write_text(content_with_meta, encoding="utf-8")
    
    return {
        "filename": filename,
        "filepath": str(filepath),
        "title": title,
        "timestamp": timestamp
    }

def list_saved_blogs() -> List[dict]:
    """List all saved blogs."""
    blogs = []
    for filepath in sorted(BLOGS_DIR.glob("*.md"), reverse=True):
        content = filepath.read_text(encoding="utf-8")
        
        # Extract metadata
        if content.startswith("<!---"):
            meta_end = content.find("--->\n")
            if meta_end != -1:
                try:
                    meta_json = content[5:meta_end].strip()
                    metadata = json.loads(meta_json)
                    blogs.append({
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "title": metadata.get("title", filepath.stem),
                        "generated_at": metadata.get("generated_at", ""),
                    })
                except:
                    pass
    return blogs

def load_blog(filename: str) -> Optional[str]:
    """Load a saved blog."""
    filepath = BLOGS_DIR / filename
    if filepath.exists():
        content = filepath.read_text(encoding="utf-8")
        # Strip metadata
        if content.startswith("<!---"):
            meta_end = content.find("--->\n")
            if meta_end != -1:
                return content[meta_end + 5:].strip()
        return content
    return None

def delete_blog(filename: str) -> bool:
    """Delete a saved blog."""
    filepath = BLOGS_DIR / filename
    if filepath.exists():
        filepath.unlink()
        return True
    return False


# =================== BLOG GENERATION RUNNER ===================

def generate_blog(topic: str, thread_id: Optional[str] = None, progress_callback=None) -> dict:
    """
    Generate a blog post with progress tracking.
    
    Args:
        topic: Blog topic
        thread_id: Optional thread ID to use RAG context
        progress_callback: Function to call with progress updates
    """
    initial_state = {
        "topic": topic,
        "thread_id": thread_id,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "rag_context": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "progress": "Starting..."
    }
    
    # Stream progress
    for event in blog_generator.stream(initial_state):
        if progress_callback and "progress" in event.get(list(event.keys())[0], {}):
            node_name = list(event.keys())[0]
            node_state = event[node_name]
            if "progress" in node_state:
                progress_callback(node_state["progress"])
    
    # Get final result
    result = blog_generator.invoke(initial_state)
    
    return {
        "content": result["final"],
        "title": result["plan"].blog_title if result.get("plan") else topic,
        "mode": result.get("mode", "unknown"),
        "sections_count": len(result.get("sections", [])),
        "used_rag": bool(result.get("rag_context")),
        "used_web": bool(result.get("evidence")),
    }