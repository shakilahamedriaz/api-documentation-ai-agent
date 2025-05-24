import os
import asyncio
from pathlib import Path
from typing import List, AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

from phi.assistant import Assistant
from phi.llm.groq import Groq
from phi.knowledge.base import KnowledgeBase
from phi.vectordb.chroma import ChromaDb
from phi.embedder.huggingface import HuggingFaceEmbedder
from phi.document import Document
from phi.document.reader.markdown import MarkdownReader
from phi.utils.log import logger

# --- Environment and Configuration ---
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY not found in .env file.")
    raise ValueError("GROQ_API_KEY not found. Please ensure it's set in the .env file.")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
DOCS_KB_DIR = BASE_DIR.parent / "docs_kb"
CHROMA_DB_DIR = BASE_DIR.parent / "chroma_db_store"

CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"API Documentation Agent starting...")
logger.info(f"Documentation directory: {DOCS_KB_DIR}")
logger.info(f"ChromaDB persist directory: {CHROMA_DB_DIR}")

# --- Knowledge Base Initialization ---
try:
    hf_embedder = HuggingFaceEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
    api_knowledge_base = KnowledgeBase(
        vector_db=ChromaDb(
            collection_name="api_docs_rag_v2", # Changed collection name for clarity if upgrading
            persist_directory=str(CHROMA_DB_DIR),
            embedder=hf_embedder,
        ),
        embedder=hf_embedder,
    )
except Exception as e:
    logger.error(f"Failed to initialize HuggingFace Embedder or ChromaDb: {e}", exc_info=True)
    raise

def load_and_process_docs_if_needed():
    """Loads and processes documents if the knowledge base is empty or needs update."""
    if not DOCS_KB_DIR.exists() or not any(DOCS_KB_DIR.iterdir()):
        logger.warning(f"Documentation directory {DOCS_KB_DIR} is empty or not found. Skipping doc loading.")
        return

    # Simple check: if the collection is empty, load.
    # ChromaDb/KnowledgeBase load is idempotent if `upsert=True` and document IDs are consistent.
    # Phidata's Document class generates IDs, or they can be derived from file name + content hash.
    # For this prototype, we process all files. ChromaDB handles deduplication/updates via `upsert=True`.

    logger.info("Processing documentation files...")
    documents_to_load: List[Document] = []
    # Using RecursiveCharacterTextSplitter with sensible defaults for Markdown.
    # Smaller chunks can sometimes provide more targeted context.
    doc_reader = MarkdownReader(chunk_by="RecursiveCharacterTextSplitter", chunk_size=600, chunk_overlap=60)

    for md_file in DOCS_KB_DIR.glob("*.md"):
        logger.debug(f"Reading and chunking: {md_file.name}")
        try:
            docs_from_file = doc_reader.read_file(md_file)
            for doc_chunk in docs_from_file:
                # Phidata's Document automatically gets an ID.
                # We can add metadata for better source tracking if needed.
                doc_chunk.meta_data = {"source": md_file.name}
            documents_to_load.extend(docs_from_file)
        except Exception as e:
            logger.error(f"Error processing file {md_file.name}: {e}", exc_info=True)

    if documents_to_load:
        logger.info(f"Loading {len(documents_to_load)} document chunks into the knowledge base...")
        try:
            api_knowledge_base.load_documents(documents_to_load, upsert=True)
            logger.info("Knowledge base updated successfully.")
        except Exception as e:
            logger.error(f"Error loading documents into ChromaDB: {e}", exc_info=True)
    else:
        logger.info("No new documents processed or found for the knowledge base.")

# Load documents on application startup
load_and_process_docs_if_needed()

# --- AI Assistant Initialization ---
try:
    api_assistant = Assistant(
        name="APIDocAssistant",
        llm=Groq(model="llama3-8b-8192", api_key=GROQ_API_KEY), # Or "mixtral-8x7b-32768"
        knowledge_base=api_knowledge_base,
        description=(
            "You are a specialized AI assistant for answering questions about a specific API. "
            "Your knowledge comes exclusively from the provided API documentation. "
            "Answer concisely and accurately based *only* on the retrieved context segments."
        ),
        instructions=[
            "IMPORTANT: Base your answers *solely* on the information present in the retrieved document snippets.",
            "If the answer is not found in the provided context, clearly state that the documentation does not cover the query.",
            "Do not invent information or use external knowledge.",
            "When referring to API endpoints, mention their HTTP method, path, and key parameters if available in the context.",
            "For code examples or configurations, present them clearly, ideally in Markdown code blocks.",
            "If the user's question is ambiguous, ask for clarification before attempting to answer.",
            "Be friendly and helpful within your role as a documentation assistant."
        ],
        add_references_to_prompt=True,  # Crucial for RAG
        num_documents_to_retrieve=3,   # Number of context chunks to retrieve
        # show_tool_calls=True, # Enable for debugging to see what context is retrieved
        debug_mode=False # Set to True for more verbose logging from Phidata
    )
except Exception as e:
    logger.error(f"Failed to initialize Phidata Assistant: {e}", exc_info=True)
    raise

# --- FastAPI Application Setup ---
app = FastAPI(
    title="Interactive API Documentation Agent",
    version="1.0.0",
    description="An AI agent that answers questions about an API using its documentation, with streaming responses.",
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

class QueryRequest(BaseModel):
    question: str

# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_chat_interface(request: Request):
    """Serves the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

async def stream_llm_response(question: str) -> AsyncGenerator[str, None]:
    """Streams the LLM response token by token."""
    if not question.strip():
        yield "Error: Question cannot be empty."
        return

    logger.info(f"Streaming response for question: {question}")
    full_response_for_log = []
    try:
        # stream=True returns a generator from phidata.assistant.run
        for chunk_obj in api_assistant.run(question, stream=True):
            chunk_str = str(chunk_obj) # Ensure it's a string
            full_response_for_log.append(chunk_str)
            yield chunk_str
            await asyncio.sleep(0.01) # Small delay to simulate token-by-token and allow client to catch up
    except Exception as e:
        logger.error(f"Error during LLM stream for question '{question}': {e}", exc_info=True)
        yield f"Error: Could not generate response. {str(e)}"
    finally:
        logger.debug(f"Full streamed response for '{question}': {''.join(full_response_for_log)}")
        logger.info(f"Finished streaming for question: {question}")


@app.post("/ask_stream")
async def ask_question_streaming(query: QueryRequest):
    """Handles a user's question and streams the assistant's answer."""
    return StreamingResponse(stream_llm_response(query.question), media_type="text/plain")

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server for API Doc Agent on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)