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
logging.getLogger("httpx").setLevel(logging.WARNING)       # suppress noisy HTTP logs
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

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072

def retry_config(name):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(f"Retrying {name}... attempt {retry_state.attempt_number}"),
        reraise=True
    )

@retry_config("Gemini Embedding (Single)")
def embed_text(text: str) -> list[float]:
    try:
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        embedding = result.embeddings[0].values
        assert len(embedding) == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM}, got {len(embedding)}"
        return list(embedding)
    except Exception as e:
        logger.error(f"Failed embedding single text: {str(e)}")
        raise

@retry_config("Gemini Embedding (Batch)")
def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        embeddings = [list(e.values) for e in result.embeddings]
        for i, emb in enumerate(embeddings):
            assert len(emb) == EMBEDDING_DIM, f"Chunk {i}: Expected {EMBEDDING_DIM}, got {len(emb)}"
        return embeddings
    except Exception as e:
        logger.error(f"Failed embedding batch: {str(e)}")
        raise

def make_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None  
    return QdrantClient(url=url, api_key=api_key)

@retry_config("Qdrant Ensure Collection")
def ensure_collection(client: QdrantClient, name: str):
    try:
        existing = [c.name for c in client.get_collections().collections]

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": VectorParams(
                        size=384,                  # sentence-transformers/all-MiniLM-L6-v2
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

