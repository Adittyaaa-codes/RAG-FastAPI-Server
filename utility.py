import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import openai
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from qdrant_client.models import VectorParams, Distance, PayloadSchemaType
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton Clients
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def retry_config(name):
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=lambda retry_state: logger.warning(f"Retrying {name}... attempt {retry_state.attempt_number}"),
        reraise=True
    )

@retry_config("OpenAI Embedding (Single)")
def embed_text(text: str) -> list[float]:
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embedding = response.data[0].embedding
        assert len(embedding) == 1536, f"Expected 1536, got {len(embedding)}"
        return embedding
    except Exception as e:
        logger.error(f"Failed embedding single text: {str(e)}")
        raise

@retry_config("OpenAI Embedding (Batch)")
def embed_texts(texts: list[str]) -> list[list[float]]:
    try:
        response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
        )
        embeddings = [item.embedding for item in response.data]

        for i, emb in enumerate(embeddings):
            assert len(emb) == 1536, f"Chunk {i}: Expected 1536, got {len(emb)}"

        return embeddings
    except Exception as e:
        logger.error(f"Failed embedding single text: {str(e)}")
        raise


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )

@retry_config("Qdrant Ensure Collection")
def ensure_collection(client: QdrantClient, name: str):
    try:
        existing = [c.name for c in client.get_collections().collections]

        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
            logger.info(f"[QDRANT] Created collection '{name}'")

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