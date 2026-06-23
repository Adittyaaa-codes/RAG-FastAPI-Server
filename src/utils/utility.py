import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)      
logging.getLogger("httpcore").setLevel(logging.WARNING)

genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize MongoDB connection for token tracking
from motor.motor_asyncio import AsyncIOMotorClient
mongo_url = os.getenv("MONGO_URL")
if not mongo_url:
    logger.warning("MONGO_URL not set in environment. Token tracking will be disabled.")
    mongo_client = None
    mongo_db = None
else:
    mongo_client = AsyncIOMotorClient(mongo_url)
    try:
        mongo_db = mongo_client.get_database() # Uses DB from connection string
    except Exception:
        mongo_db = mongo_client.get_database("test") # Fallback if URL lacks DB name


def make_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None  
    return QdrantClient(url=url, api_key=api_key)

def ensure_collection(client: QdrantClient, name: str):
    try:
        existing = [c.name for c in client.get_collections().collections]

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(
                        size=384, 
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"[QDRANT] Created hybrid collection '{name}'")

        client.create_payload_index(name, "subject", PayloadSchemaType.KEYWORD)
        client.create_payload_index(name, "chapter", PayloadSchemaType.KEYWORD)
    except Exception as e:
        logger.error(f"Qdrant collection/index error: {str(e)}")
        raise

def load_and_chunk(path: str) -> list:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      
    chunk_overlap=50,    
    separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return splitter.split_documents(docs)

import json
import os
from langchain_groq import ChatGroq
from src.constants.constants import SECONDARY_MODEL,TEMPARATURE

def generate_rolling_summary(old_messages: list) -> str:
    if not old_messages:
        return ""
    try:
        llm = ChatGroq(model=SECONDARY_MODEL, temperature=TEMPARATURE)
        text_to_summarize = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in old_messages])
        prompt = (
            "Summarize the following chat history. Focus on key facts, user preferences, "
            "and topics discussed. The summary MUST be a maximum of 200 words.\n\n"
            f"{text_to_summarize}"
        )
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Failed to generate rolling summary: {e}")
        return ""

def extract_text(content) -> str:
    """
    Safely extract a plain string from any LangChain message content.
    Handles: str, list[str], list[{"type":"text","text":"..."}], etc.
    """
    if not content:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)

def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"

def is_ai_message(msg) -> bool:
    t = type(msg).__name__
    if "AI" in t or "Assistant" in t:
        return True
    if isinstance(msg, dict):
        return msg.get("role") in ("assistant", "ai")
    return False

def is_tool_message(msg) -> bool:
    t = type(msg).__name__
    if "Tool" in t:
        return True
    if isinstance(msg, dict):
        return msg.get("role") == "tool"
    return False
